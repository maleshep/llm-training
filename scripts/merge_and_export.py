"""
Merge LoRA adapters (SFT + GRPO) into base model and export for serving.

Pipeline:
  1. Load base model (BF16) — Qwen3.6-35B-A3B
  2. Load and merge SFT adapter
  3. Load and merge GRPO adapter (stacked on top)
  4. Save merged model to output directory
  5. (Optional) Quantize to FP8 for SGLang serving on L40S

The merged model can be served directly via SGLang with --dtype auto
(BF16, ~67GB on B200) or quantized to FP8 (~18GB on L40S).

Usage:
  python scripts/merge_and_export.py \
    --base /shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b \
    --sft-adapter /shared/project/tdr-mmm-hpc/llm/training/output/qlora-20260430-1213/adapter \
    --grpo-adapter /shared/project/tdr-mmm-hpc/llm/training/output/grpo-20260430-1234/adapter \
    --output /shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b-mmm

On HPC:
  sbatch scripts/merge_and_export.sh
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(model, adapter_path: Path, adapter_name: str):
    """Load a LoRA adapter and merge it into the base model weights."""
    print(f"\n  Loading {adapter_name} adapter from: {adapter_path}")
    t0 = time.perf_counter()

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    elapsed = time.perf_counter() - t0
    print(f"  Merged {adapter_name} in {elapsed:.1f}s")
    return model


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base", type=str,
                        default="/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b",
                        help="Path to base model")
    parser.add_argument("--sft-adapter", type=str, default=None,
                        help="Path to SFT LoRA adapter (merged first)")
    parser.add_argument("--grpo-adapter", type=str, default=None,
                        help="Path to GRPO LoRA adapter (merged second)")
    parser.add_argument("--dpo-adapter", type=str, default=None,
                        help="Path to DPO LoRA adapter (merged third, if available)")
    parser.add_argument("--output", type=str,
                        default="/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b-mmm",
                        help="Output path for merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Output dtype (bfloat16 for B200, float16 for L40S)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip copying tokenizer (if already at output)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect adapters in order
    adapters = []
    if args.sft_adapter:
        adapters.append(("SFT", Path(args.sft_adapter)))
    if args.grpo_adapter:
        adapters.append(("GRPO", Path(args.grpo_adapter)))
    if args.dpo_adapter:
        adapters.append(("DPO", Path(args.dpo_adapter)))

    if not adapters:
        print("ERROR: At least one adapter path required (--sft-adapter, --grpo-adapter, or --dpo-adapter)")
        return

    # Validate paths
    for name, path in adapters:
        if not path.exists():
            print(f"ERROR: {name} adapter not found at {path}")
            return
        if not (path / "adapter_config.json").exists():
            print(f"ERROR: {name} adapter missing adapter_config.json at {path}")
            return

    print("=" * 60)
    print("MERGE & EXPORT: Qwen3.6-35B-A3B + LoRA Adapters")
    print("=" * 60)
    print(f"  Base model: {args.base}")
    for name, path in adapters:
        print(f"  {name} adapter: {path}")
    print(f"  Output: {output_path}")
    print(f"  Dtype: {args.dtype}")
    print()

    # --- 1. Load base model ---
    print("Step 1: Loading base model (this takes ~30s on B200)...")
    t0 = time.perf_counter()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Base model loaded in {elapsed:.1f}s")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {vram:.1f} GB")

    # --- 2. Merge adapters sequentially ---
    print("\nStep 2: Merging adapters...")
    for name, path in adapters:
        model = merge_adapter(model, path, name)

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"\n  VRAM after merge: {vram:.1f} GB")

    # --- 3. Save merged model ---
    print(f"\nStep 3: Saving merged model to {output_path}...")
    t0 = time.perf_counter()

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="4GB",
    )

    elapsed = time.perf_counter() - t0
    print(f"  Model saved in {elapsed:.1f}s")

    # --- 4. Copy tokenizer ---
    if not args.skip_tokenizer:
        print("\nStep 4: Copying tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base, trust_remote_code=True
        )
        tokenizer.save_pretrained(str(output_path))
        print("  Tokenizer saved")

    # --- 5. Write metadata ---
    meta = {
        "base_model": args.base,
        "adapters_merged": [
            {"stage": name, "path": str(path)} for name, path in adapters
        ],
        "output_dtype": args.dtype,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": "qwen3.6-35b-a3b-mmm",
        "description": "Qwen3.6-35B-A3B fine-tuned for pharma MMM optimization (SFT+GRPO merged)",
    }
    with open(output_path / "merge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Summary ---
    total_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors"))
    total_gb = total_size / 1e9

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print(f"  Output: {output_path}")
    print(f"  Size: {total_gb:.1f} GB ({len(list(output_path.glob('*.safetensors')))} shards)")
    print(f"  Dtype: {args.dtype}")
    print(f"  Adapters merged: {' → '.join(n for n, _ in adapters)}")
    print()
    print("  To serve with SGLang (BF16 on B200):")
    print(f"    python -m sglang.launch_server --model-path {output_path} --dtype auto --port 8100")
    print()
    print("  To quantize for L40S (FP8):")
    print(f"    python scripts/quantize_fp8.py --input {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
