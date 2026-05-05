"""
DPO (Direct Preference Optimization) training for Qwen3.6-35B-A3B.

Third stage after SFT + GRPO. Uses preference pairs (chosen vs rejected responses)
to further align the model toward high-quality MMM optimization proposals.

Model: Qwen/Qwen3.6-35B-A3B + LoRA adapter from GRPO
Hardware: 1x NVIDIA B200 (192GB VRAM) on fat partition
Method: DPO with TRL's DPOTrainer + LoRA

VRAM budget:
  - Base model (BF16): ~67GB
  - LoRA adapters: ~0.5GB
  - Reference model (shared weights, no extra VRAM with peft_ref_model): ~0GB
  - Optimizer + gradients: ~8GB
  - Activations (batch=1, seq=4096, grad ckpt): ~8GB
  - Total: ~84GB — within 192GB B200

Usage:
  python train_dpo.py --adapter-path ./output/grpo-v1/adapter --data-dir ./data

On HPC:
  sbatch train_dpo.sh
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


class VRAMLogCallback(TrainerCallback):
    """Log GPU VRAM usage at key training events."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._log_vram("train_begin")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            self._log_vram(f"step_{state.global_step}")

    def _log_vram(self, tag):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[VRAM {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")


def load_dpo_data(data_dir: Path) -> Dataset:
    """
    Load DPO JSONL into a HuggingFace Dataset.

    Expected format per line:
    {
      "prompt": [{"role": "system", ...}, {"role": "user", ...}],
      "chosen": [{"role": "assistant", "content": "..."}],
      "rejected": [{"role": "assistant", "content": "..."}]
    }

    DPOTrainer expects: prompt (list[dict]), chosen (list[dict]), rejected (list[dict])
    """
    dpo_path = data_dir / "dpo_train.jsonl"
    records = []
    with open(dpo_path) as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="DPO training for Qwen3.6-35B-A3B MMM agent")
    parser.add_argument("--model-path", type=str,
                        default="/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b",
                        help="Path to base model")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter from GRPO (loaded before DPO)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with dpo_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./output/dpo-v1",
                        help="Output directory for DPO adapter weights")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (lower than GRPO for stability)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty strength)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (16 for small datasets)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (keep alpha/r = 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("DPO Training: Qwen3.6-35B-A3B -> Pharma MMM Agent (Stage 3)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"GRPO Adapter: {args.adapter_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print()

    # --- 1. Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load DPO dataset ---
    print("Loading DPO preference pairs...")
    data_dir = Path(args.data_dir)
    dataset = load_dpo_data(data_dir)
    print(f"DPO pairs: {len(dataset)} examples")

    # --- 3. LoRA config ---
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

    # --- 4. DPO config ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
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
        # DPO-specific
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=2048,
        loss_type="sigmoid",  # Standard DPO loss
        # Model loading kwargs
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        },
    )

    # --- 5. Initialize DPO trainer ---
    print("Initializing DPOTrainer...")

    trainer = DPOTrainer(
        model=args.model_path,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[VRAMLogCallback()],
    )

    # --- 6. Train ---
    print("\n" + "=" * 60)
    print("DPO TRAINING STARTED")
    print(f"  Preference pairs: {len(dataset)}")
    print(f"  Beta (KL penalty): {args.beta}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 60 + "\n")

    result = trainer.train()

    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics.get('train_runtime', 0):.0f}s")
    print("=" * 60)

    # --- 7. Save adapter ---
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nDPO adapter saved to: {adapter_path}")

    # --- 8. Save training metadata ---
    meta = {
        "base_model": args.model_path,
        "grpo_adapter": args.adapter_path,
        "method": "DPO",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "max_length": args.max_length,
        "dataset_size": len(dataset),
        "final_loss": result.training_loss,
        "training_steps": result.global_step,
    }
    with open(output_dir / "dpo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
