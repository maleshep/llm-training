"""
LoRA fine-tuning of Qwen3.6-35B-A3B for domain adaptation.

Model: Qwen/Qwen3.6-35B-A3B (MoE — 35B total, 3B active per token)
Hardware: 1x NVIDIA B200 (192GB VRAM) on fat partition
Method: BF16 base (no quantization) + LoRA r=64 adapters

VRAM budget:
  - Base model (BF16): ~67GB
  - LoRA adapters: ~1GB
  - Optimizer states: ~4GB
  - Activations (batch=1, seq=4096, grad ckpt): ~8GB
  - Total: ~80GB — well within 192GB B200

Note: 4-bit quantization + multi-GPU device_map is incompatible with this
model's linear attention layers (gradient checkpointing causes NLL assertions).
Single B200 with BF16 avoids all quantization/multi-device issues.

Usage:
  python train_qlora.py --data-dir ./data --output-dir ./output/lora-v1

On HPC:
  sbatch train_qlora.sh  (see companion Slurm script)
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


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


def load_sft_data(data_dir: Path) -> Dataset:
    """Load SFT JSONL into a HuggingFace Dataset. Only keep 'messages' field."""
    sft_path = data_dir / "sft_train.jsonl"
    records = []
    with open(sft_path) as f:
        for line in f:
            record = json.loads(line)
            # Only keep messages — metadata causes pyarrow type conflicts
            records.append({"messages": record["messages"]})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.6-35B-A3B")
    parser.add_argument("--model-path", type=str,
                        default="./models/qwen3.6-35b-a3b",
                        help="Path to base model")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with sft_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./output/qlora-v1",
                        help="Output directory for adapter weights")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Fine-Tuning: Qwen3.6-35B-A3B")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, eff_batch={args.batch_size * args.grad_accum}")
    print()

    # --- 1. Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load BF16 model on single B200 (192GB VRAM) ---
    # No quantization needed: BF16 model (67GB) + LoRA + optimizer fits easily.
    # Note: 4-bit + multi-GPU causes NLL assertion failures with this model's
    # linear attention layers during gradient checkpointing recomputation.
    print(f"Loading BF16 model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    print(f"Model loaded. Trainable params before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 3. LoRA config ---
    # Target the attention + MLP layers in the MoE model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP/experts
        ],
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- 4. Load dataset ---
    print("Loading training data...")
    data_dir = Path(args.data_dir)
    dataset = load_sft_data(data_dir)
    print(f"Dataset: {len(dataset)} examples")

    # --- 5. Training arguments ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=args.max_seq_len,  # TRL 1.3.0: max_length, NOT max_seq_length
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
    )

    # --- 6. Trainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 1.3.0: processing_class, NOT tokenizer
        args=training_args,
        train_dataset=dataset,
        callbacks=[VRAMLogCallback()],
    )

    # --- 7. Train ---
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60 + "\n")

    result = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics.get('train_runtime', 0):.0f}s")
    print("=" * 60)

    # --- 8. Save adapter ---
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to: {adapter_path}")

    # --- 9. Save training metadata ---
    meta = {
        "base_model": args.model_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "trainable_params": trainable,
        "total_params": total,
        "dataset_size": len(dataset),
        "final_loss": result.training_loss,
        "training_steps": result.global_step,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
