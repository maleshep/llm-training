# Model Architecture: Qwen3.6-35B-A3B

Technical notes on the Qwen3.6-35B-A3B architecture and its implications for fine-tuning.

## Overview

| Property | Value |
|----------|-------|
| Architecture | Mixture of Experts (MoE) |
| Total parameters | 34.7B |
| Active parameters per token | ~3B |
| Attention mechanism | Linear (torch_chunk_gated_delta_rule) |
| Vocabulary size | 248,320 |
| Max context length | 131,072 tokens |
| Weight format | BF16 safetensors |
| Number of shards | 1,026 files |
| Disk size (BF16) | ~67 GB |
| VRAM (loaded) | ~67 GB |

## Mixture of Experts (MoE)

The model uses a sparse MoE architecture where only a subset of "expert" feed-forward networks are activated for each token. This gives:

- **35B total parameters** for knowledge capacity
- **~3B active parameters** per forward pass for inference speed
- Higher training FLOPS efficiency per parameter vs. dense models

### Expert Layers

Each MoE layer contains:
- A **router** (gate) that selects top-k experts per token
- Multiple **expert** FFN blocks (gate_proj, up_proj, down_proj)
- Shared attention layers (q_proj, k_proj, v_proj, o_proj)

LoRA targets both the shared attention layers AND the expert FFN projections:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # shared attention
    "gate_proj", "up_proj", "down_proj",       # expert FFN (all experts)
]
```

This means LoRA adapts ALL experts, not just the ones active for a given token.

## Linear Attention (torch_chunk_gated_delta_rule)

This is the critical architectural element that causes most training failures.

### What It Is

Instead of standard softmax attention (`softmax(QK^T/sqrt(d))V`), this model uses a **linear attention** variant with:
- Chunked processing (processes sequences in fixed-size chunks)
- Gated updates (controls information flow between chunks)
- Delta rule learning (updates attention state based on prediction error)

### Why It Matters for Training

The `torch_chunk_gated_delta_rule` kernel:

1. **Maintains internal hidden state** across chunks within a sequence
2. **Is not device-boundary safe** — state becomes corrupted when layers span multiple GPUs
3. **Requires exact numerical reproducibility** during gradient checkpointing recomputation
4. **Has custom CUDA kernels** that are incompatible with DeepSpeed ZeRO and FSDP sharding

### Practical Implications

| Technique | Compatible? | Why |
|-----------|-------------|-----|
| Single GPU BF16 | Yes | No device boundaries, consistent precision |
| Gradient checkpointing (single GPU) | Yes | Recomputation happens on same device |
| Multi-GPU device_map | **NO** | Hidden state crosses device boundaries |
| Gradient checkpointing + multi-GPU | **NO** | Recomputation on wrong device → corrupted state |
| DeepSpeed ZeRO-3 | **NO** | Weight sharding breaks kernel's assumptions |
| FSDP | **NO** | Same issue — can't shard layers with stateful kernels |
| 4-bit quantization (single GPU) | Likely no | 35B 4-bit still ~10GB; with overhead exceeds 48GB |
| 4-bit quantization (multi-GPU) | **NO** | Triggers multi-device issues |
| FP8 training | **NO** | No backward pass kernels for this attention type |
| FP8 inference | Possibly | Forward-only may work, not tested |

## Memory Layout

### SFT Training (single B200, 192GB)

```
┌─────────────────────────────────────────┐
│ B200 VRAM: 192 GB (183 GB usable)       │
├─────────────────────────────────────────┤
│ Base model (BF16)          │  67 GB      │
│ LoRA adapters              │   1 GB      │
│ Optimizer (AdamW states)   │   4 GB      │
│ Activations + gradients    │   8 GB      │
├─────────────────────────────────────────┤
│ TOTAL                      │ ~80 GB      │
│ Headroom                   │ ~103 GB     │
└─────────────────────────────────────────┘
```

### GRPO Training (single B200, 192GB)

```
┌─────────────────────────────────────────┐
│ B200 VRAM: 192 GB (183 GB usable)       │
├─────────────────────────────────────────┤
│ Base model (BF16)          │  67 GB      │
│ LoRA adapters              │   1 GB      │
│ Generations (4×512 tokens) │  20 GB      │
│ Optimizer + gradients      │   8 GB      │
│ KV cache + buffers         │  44 GB      │
├─────────────────────────────────────────┤
│ TOTAL                      │ ~140 GB     │
│ Headroom                   │  ~43 GB     │
└─────────────────────────────────────────┘
```

## LoRA Configuration

```python
LoraConfig(
    r=64,                    # Rank — higher than typical (8-16) because MoE
    lora_alpha=128,          # Alpha = 2*r (standard ratio)
    lora_dropout=0.05,       # Light regularization
    bias="none",             # No bias adaptation needed
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

**Why r=64?** MoE models have more total parameters but fewer active per token. Higher rank compensates by giving each expert more adaptation capacity. With 7 target modules across many experts, this yields ~33.4M trainable parameters (0.10% of total).

## Tokenizer Notes

- Vocabulary: 248,320 tokens (larger than typical — good for multilingual)
- Requires `trust_remote_code=True` (custom tokenizer implementation)
- Pad token not set by default — set to EOS: `tokenizer.pad_token = tokenizer.eos_token`
- For generation (GRPO): use `padding_side="left"`
- For training (SFT): use `padding_side="right"`

## Weight Sharding

The model ships as 1,026 safetensors files. This unusual number of shards is due to:
- Each expert's weights stored separately
- Shared layers stored in their own shards
- Allows partial loading (inference can load only needed experts)

**Loading performance**: ~14 seconds on NVMe, 2-5 minutes on network filesystem.
Always prefer local/NVMe storage for model weights.
