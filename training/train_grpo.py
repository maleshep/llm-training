"""
GRPO (Group Relative Policy Optimization) training for Qwen3.6-35B-A3B.

Builds on the LoRA adapter from SFT training. Uses rule-based reward functions
to reinforce good MMM optimization reasoning without a separate reward model.

Model: Qwen/Qwen3.6-35B-A3B + LoRA adapter from SFT
Hardware: 1× NVIDIA B200 (192GB VRAM) on fat partition
Method: GRPO with rule-based rewards (no reward model needed)

VRAM budget (higher than SFT due to multiple generations per prompt):
  - Base model (BF16): ~67GB
  - LoRA adapters: ~0.5GB
  - Generations (num_generations=8, max_completion=1024): ~50GB
  - Optimizer + gradients: ~8GB
  - Total: ~126GB — within 192GB B200

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
# For chat models, completions is list[list[dict]] where each item is
# [{"role": "assistant", "content": "..."}]. We extract the text content.


def _extract_text(completion) -> str:
    """Extract text content from a completion (handles both chat-format and plain string)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat format: [{"role": "assistant", "content": "..."}]
        texts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return " ".join(texts)
    return str(completion)


def reward_structure(completions, **kwargs) -> list[float]:
    """Reward structured reasoning — format-agnostic.

    Accepts BOTH markdown (## Approach) AND think-mode (<think> blocks).
    Scores on semantic content: does the response contain structured reasoning
    with identifiable approach, reasoning, and proposed changes sections?
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        text_lower = text.lower()

        # --- Approach/strategy section (markdown OR plain text) ---
        has_approach = (
            "## approach" in text_lower
            or re.search(r"approach\s*[-:–—]", text_lower)
            or re.search(r"<think>.*?(strategy|approach|plan)\b", text_lower, re.DOTALL)
            or re.search(r"^(my |the )?(approach|strategy|plan)\b", text_lower, re.MULTILINE)
        )
        if has_approach:
            score += 0.3

        # --- Reasoning/evidence section ---
        has_reasoning = (
            "## reasoning" in text_lower
            or re.search(r"reasoning\s*[-:–—]", text_lower)
            or re.search(r"(because|since|evidence|rationale|justification)", text_lower)
            or re.search(r"<think>.*?(because|therefore|since)\b", text_lower, re.DOTALL)
        )
        if has_reasoning:
            score += 0.3

        # --- Config/parameter change proposals ---
        has_config = (
            "## config" in text_lower
            or "config change" in text_lower
            or re.search(r"(param|config|setting).*?(change|update|modify|set)", text_lower)
            or re.search(r"[\w_]+\s*[:=]\s*\S+\s*[→\->]+\s*\S+", text)
            or re.search(r"(increase|decrease|set|change)\s+[\w_.]+\s+(to|from)", text_lower)
        )
        if has_config:
            score += 0.4

        rewards.append(score)
    return rewards


def reward_config_format(completions, **kwargs) -> list[float]:
    """Reward config change proposals — format-agnostic.

    Accepts BOTH markdown (`param`: old → new) AND plain-text parameter changes
    (param = value, set param to value, change param from X to Y).
    Scores on whether the response proposes concrete, identifiable parameter changes.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0

        # --- Markdown backtick param format: `param`: value -> value ---
        param_changes_md = re.findall(r'`[\w_.]+`\s*[:=]?\s*\S+\s*[→\->]+\s*\S+', text)
        if param_changes_md:
            score += min(len(param_changes_md) * 0.12, 0.5)

        # --- Plain text arrow format: param_name: old → new OR param = old -> new ---
        param_changes_plain = re.findall(
            r'[\w_.]+\s*[:=]\s*\S+\s*(?:→|->|-->)\s*\S+', text
        )
        if param_changes_plain:
            score += min(len(param_changes_plain) * 0.12, 0.5)

        # --- Verbal change format: "set X to Y", "change X from A to B", "increase X to Y" ---
        verbal_changes = re.findall(
            r'(set|change|update|increase|decrease|reduce|raise)\s+[\w_.]+\s+(?:to|from)\s+\S+',
            text, re.IGNORECASE
        )
        if verbal_changes:
            score += min(len(verbal_changes) * 0.1, 0.4)

        # --- Bullet-point config items (markdown or plain) ---
        config_bullets = re.findall(r'^\s*[-*]\s+[\w_.]+', text, re.MULTILINE)
        if config_bullets:
            score += min(len(config_bullets) * 0.05, 0.2)

        rewards.append(min(score, 1.0))
    return rewards


def reward_gate_awareness(completions, **kwargs) -> list[float]:
    """Reward mentioning evaluation gates with correct thresholds."""
    gate_terms = {
        "email": ["email", "email_attribution", "email attr"],
        "f2f": ["f2f", "face-to-face", "f2f_plausibility"],
        "r_squared": ["r²", "r-squared", "r2", "r_squared"],
        "divergences": ["divergen"],
        "trust": ["trust_score", "trust score", "trust"],
        "ess": ["ess", "ess_bulk", "effective sample"],
    }
    thresholds = {
        "email": ["30", "30%"],
        "f2f": ["75", "75%", "85"],
        "r_squared": ["0.85", ".85"],
        "divergences": ["0"],
        "trust": ["50"],
        "ess": ["100"],
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
        score += min(gates_mentioned * 0.1, 0.4)
        score += min(thresholds_correct * 0.1, 0.3)
        # Bonus for mentioning pymc-marketing version constraint
        if "0.18" in text or "pymc-marketing" in text_lower:
            score += 0.15
        # Bonus for mentioning locked parameters
        if "opens" in text_lower and "email" in text_lower:
            score += 0.1
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
        # Domain-specific terminology
        domain_terms = ["adstock", "saturation", "half-life", "hierarchical",
                        "pymc", "bayesian", "prior", "posterior", "mcmc",
                        "channel", "attribution", "carryover", "diminishing"]
        domain_count = sum(1 for t in domain_terms if t in text.lower())
        score += min(domain_count * 0.05, 0.4)
        rewards.append(min(score, 1.0))
    return rewards


def reward_domain_correctness(completions, **kwargs) -> list[float]:
    """Penalize known-bad patterns; reward known-good patterns."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        text_lower = text.lower()
        # Fatal: suggesting pymc-marketing 0.19.x (breaks F2F)
        if re.search(r"pymc.marketing.*(0\.19|0\.20)", text_lower):
            rewards.append(-1.0)
            continue
        # Good: pin 0.18.0
        if "0.18.0" in text_lower or "0.18" in text_lower:
            score += 0.3
        # Good: per-channel adstock
        if "per-channel" in text_lower or "per_channel" in text_lower:
            score += 0.2
        # Good: adequate sampling
        if any(w in text_lower for w in ["tune", "sampling", "chains", "ess"]):
            score += 0.15
        # Good: mentions specific channel names
        channels = ["calls_f2f", "emails_mass", "events_medical",
                     "calls_remote", "emails_rte", "events_commercial"]
        ch_count = sum(1 for c in channels if c in text_lower)
        score += min(ch_count * 0.1, 0.3)
        # Bad: suggesting normalization (library handles it)
        if "normalize spend" in text_lower or "log transform" in text_lower:
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
    parser = argparse.ArgumentParser(description="GRPO training for Qwen3.6-35B-A3B MMM agent")
    parser.add_argument("--model-path", type=str,
                        default="/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b",
                        help="Path to base model")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter from SFT (for merging before GRPO)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with sft_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./output/grpo-v1",
                        help="Output directory for GRPO adapter weights")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (lower than SFT)")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions per prompt for group comparison")
    parser.add_argument("--max-completion-length", type=int, default=1024,
                        help="Max tokens per generated completion")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (16 for small datasets)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (keep alpha/r = 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training: Qwen3.6-35B-A3B -> Pharma MMM Agent")
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
    # For MoE with small datasets, only target shared attention layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )

    # --- 4. GRPO config ---
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
        reward_weights=[0.20, 0.20, 0.15, 0.20, 0.15, 0.10],
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
        reward_gate_awareness,
        reward_evidence_reasoning,
        reward_domain_correctness,
        reward_length,
    ]

    # --- 6. Initialize GRPO trainer ---
    # GRPOTrainer can accept model as a string path + model_init_kwargs
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
