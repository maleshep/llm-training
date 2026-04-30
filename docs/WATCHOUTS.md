# Critical Watchouts: Qwen3.6-35B-A3B Training

10 hard-won lessons from 22 failed jobs before finding a working configuration.

## The Failures

| # | What We Tried | Why It Failed | GPU | Time Wasted |
|---|---------------|---------------|-----|-------------|
| 1 | 4-bit quantization + multi-GPU | NLL assertion: recomputed logits diverge from forward pass | L40S x4 | 45 min |
| 2 | FP8 quantization | No backward pass support for FP8 on this architecture | L40S x4 | 15 min |
| 3 | BF16 + multi-GPU (device_map="auto") | Same NLL assertion — gradient checkpointing across device boundaries | L40S x4 | 30 min |
| 4 | 4-bit + single L40S | OOM — 35B model doesn't fit in 48GB even quantized | L40S x1 | 5 min |
| 5 | DeepSpeed ZeRO-3 | Incompatible with model's custom attention kernel | L40S x4 | 10 min |
| 6 | FSDP | Same custom kernel issue — can't shard linear attention state | L40S x4 | 5 min |

**Root cause**: The model uses `torch_chunk_gated_delta_rule` for linear attention. This kernel maintains internal state that becomes corrupted when:
- Weights are split across GPUs (device_map)
- Gradient checkpointing recomputes on a different device than the forward pass
- Quantization reduces precision below BF16

## The 10 Watchouts

### 1. DO NOT use multi-GPU for training this model

**Symptom**: `AssertionError: nll < 0` or wildly divergent loss after step 1.

**Cause**: Linear attention (`torch_chunk_gated_delta_rule`) stores chunked hidden state that must remain on a single device. When `device_map="auto"` splits layers across GPUs, the recomputation pass during backward produces different logits than the forward pass.

**Fix**: Single GPU with enough VRAM (B200 192GB).

### 2. DO NOT quantize for training

**Symptom**: OOM (4-bit still needs multi-GPU for 35B) or no gradient flow (FP8).

**Cause**: 
- 4-bit: Reduces per-weight memory but you still need ~20GB for activations/optimizer. Total for 35B 4-bit + LoRA + optimizer ≈ 55-60GB — too much for 48GB L40S, so you need multi-GPU, which triggers watchout #1.
- FP8: Forward-only — no backward pass kernels exist for this model.

**Fix**: BF16 on a single large-VRAM GPU. The 67GB base model fits on a 192GB B200 with 100GB+ headroom.

### 3. DO NOT use gradient checkpointing with multi-device

**Symptom**: Same NLL assertion as #1, but may take a few steps to manifest.

**Cause**: Gradient checkpointing discards activations during forward and recomputes them during backward. If the model spans multiple devices, recomputation happens on a different device than the original forward, causing numerical divergence in the attention state.

**Fix**: Gradient checkpointing is FINE on a single device. Only problematic when combined with device_map across multiple GPUs.

### 4. TRL 1.3.0 uses `max_length`, not `max_seq_length`

**Symptom**: `TypeError: __init__() got an unexpected keyword argument 'max_seq_length'`

**Cause**: TRL renamed the parameter between versions. If you're on TRL 1.3.0+, the SFTConfig parameter is `max_length`.

**Fix**: `SFTConfig(max_length=4096, ...)` not `SFTConfig(max_seq_length=4096, ...)`

### 5. TRL 1.3.0 uses `processing_class`, not `tokenizer`

**Symptom**: `TypeError: __init__() got an unexpected keyword argument 'tokenizer'`

**Cause**: Both SFTTrainer and GRPOTrainer in TRL 1.3.0 renamed `tokenizer` to `processing_class`.

**Fix**: `SFTTrainer(processing_class=tokenizer, ...)` not `SFTTrainer(tokenizer=tokenizer, ...)`

### 6. GRPO reward functions receive chat-format completions

**Symptom**: `AttributeError: 'list' object has no attribute 'lower'`

**Cause**: For chat models, TRL 1.3.0 passes completions as `list[list[dict]]` where each completion is `[{"role": "assistant", "content": "..."}]`. Reward functions that call `.lower()` or `.split()` directly on the completion will crash.

**Fix**: Use a helper to extract text:
```python
def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        texts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return " ".join(texts)
    return str(completion)
```

### 7. GRPO reward_weights MUST be in GRPOConfig, not set on trainer

**Symptom**: `AttributeError: 'list' object has no attribute 'to'` during training

**Cause**: Setting `trainer.reward_weights = [...]` after initialization creates a plain Python list. TRL expects a tensor (set via config).

**Fix**: Pass weights in the config:
```python
GRPOConfig(reward_weights=[0.15, 0.20, 0.20, 0.20, 0.15, 0.10], ...)
```

### 8. GRPO prompts must be message-list format for chat models

**Symptom**: Generation produces garbage or empty completions.

**Cause**: GRPOTrainer for chat models expects `prompt` to be `list[dict]` (message format), not a plain string.

**Fix**: 
```python
# Keep system + user messages; GRPO generates the assistant response
prompt_messages = [m for m in messages if m["role"] in ("system", "user")]
records.append({"prompt": prompt_messages})
```

### 9. This model has 1,026 weight shards — use NVMe storage

**Symptom**: Model load takes 5+ minutes, or OOM during loading on network filesystem.

**Cause**: 1,026 safetensors shards × sequential reads on slow storage = very slow loading. Network filesystems may also buffer all shards in RAM.

**Fix**: Store the model on NVMe-backed storage. Load time drops to ~14 seconds.

### 10. Use `low_cpu_mem_usage=True` and `device_map={"": 0}` together

**Symptom**: OOM during model loading (before training even starts).

**Cause**: Without `low_cpu_mem_usage`, PyTorch first loads the full model to CPU RAM, then copies to GPU — requiring 67GB CPU RAM + 67GB GPU VRAM simultaneously.

**Fix**:
```python
AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0},        # Everything on GPU 0
    low_cpu_mem_usage=True,    # Stream weights directly to GPU
    torch_dtype=torch.bfloat16,
)
```

## The Working Configuration

```
Hardware: 1x NVIDIA B200 (192GB VRAM)
Model:    BF16 (no quantization)
Device:   Single GPU (device_map={"": 0})
LoRA:     r=64, alpha=128, target all attention + MLP projections
Gradient checkpointing: ON (single device = safe)
SDPA attention: ON (flash attention compatible)

SFT:  ~80GB VRAM peak (43% of available)
GRPO: ~140GB VRAM peak (76% of available)
```

## Quick Decision Flowchart

```
Can you fit the BF16 model on ONE GPU?
├── YES (>=128GB VRAM) → Use BF16 + LoRA + gradient checkpointing. Done.
├── MAYBE (80-128GB) → Try BF16. If OOM, reduce batch/seq_len first.
└── NO (<80GB) → DO NOT attempt this model's training without larger GPU.
                  Inference-only with quantization may work, but training won't.
```
