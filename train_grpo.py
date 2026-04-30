"""
GRPO (Group Relative Policy Optimization) training for Qwen3.6-35B-A3B.

Builds on the LoRA adapter from SFT training. Uses rule-based reward functions
to reinforce good domain-specific reasoning without a separate reward model.

Model: Qwen/Qwen3.6-35B-A3B + LoRA adapter from SFT
Hardware: 1x NVIDIA B200 (192GB VRAM) on fat partition
Method: GRPO with rule-based rewards (no reward model needed)

VRAM budget (higher than SFT due to multiple generations per prompt):
  - Base model (BF16): ~67GB
  - LoRA adapters: ~1GB
  - Generations (num_generations=4, max_completion=512): ~20GB
  - Optimizer + gradients: ~8GB
  - Total: ~96GB — within 192GB B200

Usage:
  python train_grpo.py --adapter-path ./output/qlora-v1/adapter --data-dir ./data

On HPC:
  sbatch train_grpo.sh
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# --- Reward Functions ---
# GRPO uses rule-based rewards: each function scores a completion on one axis.
# TRL 1.3.0 calls: reward_func(prompts=..., completions=..., completion_ids=..., **kwargs)
# IMPORTANT: For chat models, completions is list[list[dict]] where each item is
# [{"role": "assistant", "content": "..."}]. We extract the text content.


def _extract_text(completion) -> str:
    """Extract text content from a completion (handles both chat-format and plain string).

    TRL 1.3.0 passes completions differently depending on model type:
    - Chat models: list[list[dict]] -> [{"role": "assistant", "content": "..."}]
    - Causal models: list[str]

    This helper handles both, preventing the common 'list' object has no attribute 'lower' error.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat format: [{"role": "assistant", "content": "..."}]
        texts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return " ".join(texts)
    return str(completion)


def reward_structure(completions, **kwargs) -> list[float]:
    """Reward structured reasoning: sections like ## Approach, ## Reasoning, ## Changes."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        text_lower = text.lower()
        if "## approach" in text_lower or "approach" in text_lower[:100]:
            score += 0.3
        if "## reasoning" in text_lower or "reasoning" in text_lower[:200]:
            score += 0.3
        if "## config" in text_lower or "config change" in text_lower:
            score += 0.4
        rewards.append(score)
    return rewards


def reward_config_format(completions, **kwargs) -> list[float]:
    """Reward properly formatted configuration change proposals."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        # Look for param change format: `param`: value -> value
        param_changes = re.findall(r'`[\w_]+`\s*[:=]?\s*\S+\s*[->]+\s*\S+', text)
        if param_changes:
            score += min(len(param_changes) * 0.15, 0.6)
        # Look for bullet-point config items
        config_bullets = re.findall(r'^\s*[-*]\s+`[\w_]+`', text, re.MULTILINE)
        if config_bullets:
            score += min(len(config_bullets) * 0.1, 0.4)
        rewards.append(min(score, 1.0))
    return rewards


def reward_criteria_awareness(completions, **kwargs) -> list[float]:
    """Reward mentioning evaluation criteria with correct thresholds."""
    # Define your domain-specific evaluation gates here
    gate_terms = {
        "accuracy": ["accuracy", "r-squared", "r2", "r_squared"],
        "convergence": ["convergence", "divergen"],
        "reliability": ["reliability", "trust_score", "trust score"],
        "sample_quality": ["ess", "ess_bulk", "effective sample"],
    }
    thresholds = {
        "accuracy": ["0.85", ".85", "85%"],
        "convergence": ["0"],
        "reliability": ["50"],
        "sample_quality": ["100"],
    }
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        text_lower = text.lower()
        score = 0.0
        gates_mentioned = 0
        thresholds_correct = 0
        for gate, terms in gate_terms.items():
            if any(t in text_lower for t in terms):
                gates_mentioned += 1
                if any(t in text for t in thresholds.get(gate, [])):
                    thresholds_correct += 1
        score += min(gates_mentioned * 0.15, 0.5)
        score += min(thresholds_correct * 0.15, 0.4)
        rewards.append(min(score, 1.0))
    return rewards


def reward_evidence_reasoning(completions, **kwargs) -> list[float]:
    """Reward evidence-grounded reasoning (citations, quantitative arguments)."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        # Quantitative reasoning
        quant_patterns = re.findall(r'\d+\.?\d*\s*(?:%|GB|MB|samples?|chains?|draws?|tune)', text)
        score += min(len(quant_patterns) * 0.08, 0.3)
        # Causal reasoning words
        causal_terms = ["because", "since", "therefore", "indicates", "suggests",
                        "evidence", "prior", "posterior", "hypothesis", "observed"]
        causal_count = sum(1 for t in causal_terms if t in text.lower())
        score += min(causal_count * 0.06, 0.3)
        # Domain-specific terminology (customize for your use case)
        domain_terms = ["bayesian", "prior", "posterior", "mcmc",
                        "hierarchical", "sampling", "convergence",
                        "hyperparameter", "optimization", "configuration"]
        domain_count = sum(1 for t in domain_terms if t in text.lower())
        score += min(domain_count * 0.05, 0.4)
        rewards.append(min(score, 1.0))
    return rewards


def reward_domain_correctness(completions, **kwargs) -> list[float]:
    """Penalize known-bad patterns; reward known-good patterns.

    Customize this for your domain. Example: penalize suggesting
    incompatible library versions, reward mentioning best practices.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        text_lower = text.lower()
        # Example: penalize suggesting a known-broken version
        if re.search(r"library.*(0\.19|0\.20)", text_lower):
            rewards.append(-1.0)
            continue
        # Reward mentioning pinned/stable version
        if "0.18" in text_lower:
            score += 0.3
        # Reward mentioning best practices
        if any(w in text_lower for w in ["sampling", "chains", "convergence"]):
            score += 0.2
        # Penalize known anti-patterns
        if "normalize" in text_lower and "manually" in text_lower:
            score -= 0.3
        rewards.append(min(max(score, -1.0), 1.0))
    return rewards


def reward_length(completions, **kwargs) -> list[float]:
    """Penalize very short or excessively long responses."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        word_count = len(text.split())
        if word_count < 30:
            score = -0.5  # Too short — likely degenerate
        elif word_count < 80:
            score = 0.0   # Short but might be OK
        elif word_count <= 500:
            score = 0.3   # Good length
        elif word_count <= 800:
            score = 0.1   # Getting long
        else:
            score = -0.2  # Too verbose
        rewards.append(score)
    return rewards


class VRAMLogCallback(TrainerCallback):
    """Log GPU VRAM usage at key training events."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._log_vram("train_begin")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 2 == 0:
            self._log_vram(f"step_{state.global_step}")

    def _log_vram(self, tag):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[VRAM {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")


def load_prompts(data_dir: Path) -> Dataset:
    """
    Load SFT data and extract just the prompts (system + user messages).
    GRPO generates its own completions — we only need the prompts.
    TRL 1.3.0 expects 'prompt' as a list of message dicts for chat models.
    """
    sft_path = data_dir / "sft_train.jsonl"
    records = []
    with open(sft_path) as f:
        for line in f:
            record = json.loads(line)
            messages = record["messages"]
            # Keep only system + user messages (GRPO generates the assistant response)
            prompt_messages = [m for m in messages if m["role"] in ("system", "user")]
            records.append({"prompt": prompt_messages})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="GRPO training for domain-adapted LLM")
    parser.add_argument("--model-path", type=str,
                        default="./models/qwen3.6-35b-a3b",
                        help="Path to base model")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter from SFT (optional)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with sft_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./output/grpo-v1",
                        help="Output directory for GRPO adapter weights")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (lower than SFT)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt for group comparison")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="Max tokens per generated completion")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training: Qwen3.6-35B-A3B")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"SFT Adapter: {args.adapter_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max completion length: {args.max_completion_length}")
    print()

    # --- 1. Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",  # Left padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load prompts dataset ---
    print("Loading prompts...")
    data_dir = Path(args.data_dir)
    dataset = load_prompts(data_dir)
    print(f"Prompts: {len(dataset)} examples")

    # --- 3. LoRA config (GRPO applies LoRA via peft_config param) ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- 4. GRPO config ---
    # IMPORTANT: reward_weights MUST be in GRPOConfig, NOT set on trainer after init.
    # Setting trainer.reward_weights = [...] causes "list has no attribute .to()" error.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        top_p=0.9,
        beta=0.01,  # KL penalty coefficient
        loss_type="grpo",
        scale_rewards="group",
        reward_weights=[0.15, 0.20, 0.20, 0.20, 0.15, 0.10],  # MUST be here, not on trainer
        log_completions=True,
        num_completions_to_print=2,
        # Model loading kwargs (GRPOTrainer loads model from string path)
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        },
    )

    # --- 5. Reward functions ---
    reward_funcs = [
        reward_structure,
        reward_config_format,
        reward_criteria_awareness,
        reward_evidence_reasoning,
        reward_domain_correctness,
        reward_length,
    ]

    # --- 6. Initialize GRPO trainer ---
    # GRPOTrainer accepts model as a string path + model_init_kwargs
    # and peft_config to apply LoRA internally
    print("Initializing GRPOTrainer...")

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[VRAMLogCallback()],
    )

    # --- 7. Train ---
    print("\n" + "=" * 60)
    print("GRPO TRAINING STARTED")
    print(f"Reward functions: {[f.__name__ for f in reward_funcs]}")
    print("=" * 60 + "\n")

    result = trainer.train()

    print("\n" + "=" * 60)
    print("GRPO TRAINING COMPLETE")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics.get('train_runtime', 0):.0f}s")
    print("=" * 60)

    # --- 8. Save adapter ---
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nGRPO adapter saved to: {adapter_path}")

    # --- 9. Save training metadata ---
    meta = {
        "base_model": args.model_path,
        "sft_adapter": args.adapter_path,
        "method": "GRPO",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "reward_functions": [f.__name__ for f in reward_funcs],
        "dataset_size": len(dataset),
        "final_loss": result.training_loss,
        "training_steps": result.global_step,
    }
    with open(output_dir / "grpo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
