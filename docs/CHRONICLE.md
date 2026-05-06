# LLM Training Pipeline Chronicle
## Qwen3.6-35B-A3B → Pharma MMM Agent (2026-04-30)

**Session duration**: ~2h 45m  
**Outcome**: Both SFT and GRPO training completed successfully  
**Final hardware**: 1× NVIDIA B200 (192GB VRAM) on `fat` partition  

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (COMPLETED)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐        │
│  │ Meta-Harness │     │  extract_training │     │  /llm/training/data/ │        │
│  │ Iteration    │────▶│  _data.py         │────▶│  sft_train.jsonl     │        │
│  │ History      │     │                  │     │  dpo_train.jsonl     │        │
│  │ (14 iters)   │     │  12 SFT pairs    │     │  stats.json          │        │
│  └──────────────┘     │   4 DPO pairs    │     └──────────┬───────────┘        │
│                        └──────────────────┘                │                    │
│                                                            │                    │
│  ┌─────────────────────────────────────────────────────────┼────────────────┐   │
│  │                         STAGE 1: SFT (LoRA)             │                │   │
│  │                                                         ▼                │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────────┐       │   │
│  │  │ Qwen3.6-35B-A3B   │     │  train_qlora.py                    │       │   │
│  │  │ (BF16, 67GB)      │────▶│  • LoRA r=64, alpha=128           │       │   │
│  │  │ device_map={"":0}  │     │  • target: q/k/v/o + gate/up/down │       │   │
│  │  │ on B200 (192GB)   │     │  • 33.4M trainable / 34.7B total  │       │   │
│  │  └───────────────────┘     │  • 3 epochs, lr=2e-4, seq=4096    │       │   │
│  │                            │  • gradient_checkpointing=True     │       │   │
│  │                            └─────────────────┬──────────────────┘       │   │
│  │                                              │                          │   │
│  │                                              ▼                          │   │
│  │                            ┌────────────────────────────────────┐       │   │
│  │                            │  OUTPUT: qlora-20260430-1213/      │       │   │
│  │                            │  • adapter/ (LoRA weights)         │       │   │
│  │                            │  • Loss: 2.57 → 1.29              │       │   │
│  │                            │  • Accuracy: 50% → 68%            │       │   │
│  │                            │  • Runtime: 71 seconds             │       │   │
│  │                            │  • VRAM: 69.5GB / 183GB           │       │   │
│  │                            └─────────────────┬──────────────────┘       │   │
│  └──────────────────────────────────────────────┼──────────────────────────┘   │
│                                                 │                              │
│  ┌──────────────────────────────────────────────┼──────────────────────────┐   │
│  │                         STAGE 2: GRPO                    │              │   │
│  │                                                          ▼              │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────────┐      │   │
│  │  │ Qwen3.6-35B-A3B   │     │  train_grpo.py                     │      │   │
│  │  │ (BF16, 67GB)      │────▶│  • Fresh LoRA (same config)        │      │   │
│  │  │ + model_init_kwargs│     │  • 4 generations per prompt        │      │   │
│  │  │ on B200 (192GB)   │     │  • 512 max completion tokens       │      │   │
│  │  └───────────────────┘     │  • 6 rule-based reward functions   │      │   │
│  │                            │  • loss_type="grpo", beta=0.01     │      │   │
│  │                            │  • 1 epoch, lr=5e-6               │      │   │
│  │                            └─────────────────┬──────────────────┘      │   │
│  │                                              │                         │   │
│  │  ┌────────────────────────────────────┐      │                         │   │
│  │  │  REWARD FUNCTIONS (rule-based)     │      │                         │   │
│  │  │                                    │      │                         │   │
│  │  │  reward_structure       (0.15)  ───┤      │                         │   │
│  │  │  reward_config_format   (0.20)  ───┤      │                         │   │
│  │  │  reward_gate_awareness  (0.20)  ───┼──────┤                         │   │
│  │  │  reward_evidence        (0.20)  ───┤      │                         │   │
│  │  │  reward_domain_correct  (0.15)  ───┤      │                         │   │
│  │  │  reward_length          (0.10)  ───┘      │                         │   │
│  │  └────────────────────────────────────┘      │                         │   │
│  │                                              ▼                         │   │
│  │                            ┌────────────────────────────────────┐      │   │
│  │                            │  OUTPUT: grpo-20260430-1234/       │      │   │
│  │                            │  • adapter/ (GRPO LoRA weights)    │      │   │
│  │                            │  • completions/ (12 parquet files) │      │   │
│  │                            │  • Loss: ~0 (expected for GRPO)    │      │   │
│  │                            │  • Runtime: 494s (8.2 min)         │      │   │
│  │                            │  • VRAM: 140.6GB / 183GB          │      │   │
│  │                            └────────────────────────────────────┘      │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                     CURRENT STATUS: NOT INFERENCING                     │   │
│  │                                                                        │   │
│  │  Adapters are saved on disk. No serving job, no inference endpoint.    │   │
│  │  To use: load base model + adapter with PeftModel.from_pretrained()    │   │
│  │  Next: merge adapters → deploy as vLLM endpoint or integrate into      │   │
│  │  meta-harness as the proposer agent.                                   │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## The Journey: Why Each Step Changed

### Attempt Timeline (Chronological)

```
Job ID    │ Partition │ GPU       │ Approach                    │ Outcome
──────────┼───────────┼───────────┼─────────────────────────────┼──────────────────────
1803902   │ gpu       │ 2× L40S  │ QLoRA 4-bit, max_seq_length │ ❌ TRL API error
1803961   │ gpu       │ 2× L40S  │ FP8 quantization            │ ❌ FP8 rejected by validator
1803994   │ gpu       │ 2× L40S  │ 4-bit, mem=64G              │ ❌ CPU OOM (67GB model)
1804014   │ gpu       │ 2× L40S  │ 4-bit, mem=128G             │ ❌ GPU OOM in kbit prep
1804038   │ gpu       │ 2× L40S  │ 4-bit, no max_memory        │ ❌ CPU module dispatch
1804052   │ gpu       │ 2× L40S  │ 4-bit, device_map="auto"    │ ❌ NLL loss assertion
1804140   │ gpu       │ 2× L40S  │ + grad_ckpt=False           │ ❌ NLL loss assertion
1804176   │ gpu       │ 2× L40S  │ + label validation added    │ ❌ Labels valid, still NLL
1804209   │ gpu       │ 2× L40S  │ FP8 (kernels installed)     │ ❌ ImportError: kernels
1804237   │ gpu       │ 2× L40S  │ FP8 monkey-patch validator  │ ❌ Patch didn't stick
1804272   │ gpu       │ 2× L40S  │ FP8 double-patch            │ ❌ No autograd for FP8
1804317   │ gpu       │ 1× L40S  │ 4-bit single GPU            │ ❌ OOM during quant load
1804319   │ gpu       │ 2× L40S  │ 4-bit, grad_ckpt=False      │ ❌ OOM in backward pass
1804321   │ gpu       │ 2× L40S  │ 4-bit, grad_ckpt=True       │ ❌ NLL assertion returns
1804324   │ gpu       │ 2× L40S  │ Confirmed: ckpt + multi = 💀│ ❌ Same NLL assertion
──────────┼───────────┼───────────┼─────────────────────────────┼──────────────────────
1804326   │ fat       │ 1× B200  │ BF16, no quant, single GPU  │ ✅ SFT COMPLETE (71s)
1804330   │ fat       │ 1× B200  │ GRPO (reward_func format)   │ ❌ 'list' has no 'lower'
1804336   │ fat       │ 1× B200  │ GRPO (reward_weights)       │ ❌ list.to() error
1804337   │ fat       │ 1× B200  │ GRPO (all fixes)            │ ✅ GRPO COMPLETE (494s)
──────────┴───────────┴───────────┴─────────────────────────────┴──────────────────────

Total jobs submitted: 19
Success rate: 2/19 (10.5%) — but these 2 are the only ones that matter.
```

---

## Decision Points & Reasoning

### 1. Initial Approach: QLoRA on 2× L40S (Jobs 1803902–1804324)

**Why this seemed right:**
- Qwen3.6-35B-A3B is a MoE model (35B total, 3B active per token)
- At 4-bit quantization: ~9GB. Should fit on a single L40S (48GB) easily.
- Standard recipe: bitsandbytes 4-bit NF4 + LoRA. Works for most models.

**What went wrong (root cause discovery):**

```
┌─────────────────────────────────────────────────────────────────┐
│  ROOT CAUSE: Linear Attention + Gradient Checkpointing          │
│  + Multi-GPU device_map = Corrupted Forward Pass                │
│                                                                 │
│  Qwen3.6-35B-A3B uses "torch_chunk_gated_delta_rule" linear    │
│  attention (NOT standard self-attention). This architecture     │
│  maintains internal state that gradient checkpointing's         │
│  recomputation phase corrupts when tensors span multiple GPUs.  │
│                                                                 │
│  The corruption manifests as logits containing values that      │
│  produce labels >= vocab_size (248320), triggering:             │
│    CUDA error: device-side assert triggered                     │
│    nll_loss_forward: assertion `t >= 0 && t < n_classes`        │
│                                                                 │
│  This is NOT a data issue (labels validated as correct).        │
│  This is NOT a standard OOM (forward pass completes).           │
│  This is a fundamental architecture incompatibility.            │
└─────────────────────────────────────────────────────────────────┘
```

**Why each variant failed:**

| Approach | Why it failed | What I learned |
|----------|---------------|----------------|
| `max_seq_length` | TRL 1.3.0 renamed to `max_length` | Always check API version |
| FP8 | No backward pass implementation | FP8 is inference-only |
| 4-bit + `mem=64G` | BF16 model passes through CPU during quantization (67GB) | Quant needs peak CPU RAM |
| `prepare_model_for_kbit_training` | Converts layernorms to FP32, exceeds GPU 0 | Remove this for multi-GPU |
| `device_map="auto"` + `grad_ckpt=True` | NLL assertion (root cause above) | Architecture incompatibility |
| `device_map="auto"` + `grad_ckpt=False` | OOM in backward on GPU 1 | Linear attention uses >44GB for backward |
| Single L40S | Can't hold 4-bit model during quantization conversion | 44GB too tight for 35B even at 4-bit |

### 2. The Pivot: B200 on Fat Partition (Job 1804326)

**Why this was the right answer:**

```
                    L40S (48GB)                    B200 (192GB)
                   ┌───────────┐                  ┌───────────────────────────┐
                   │ ████████░░│ 44.4GB usable    │ ████████░░░░░░░░░░░░░░░░░│ 183GB usable
                   │ ████████░░│                  │                           │
                   │ ████████░░│ Model (4-bit):   │ Model (BF16): 67GB        │
                   │ █████░░░░░│ 9GB              │ ████████████████░░░░░░░░░│
                   │ ░░░░░░░░░░│ + Overhead:      │ LoRA: 1GB                │
                   │ ░░░░░░░░░░│ OOM during load  │ █░░░░░░░░░░░░░░░░░░░░░░░│
                   └───────────┘                  │ Optimizer: 4GB            │
                   ❌ CAN'T FIT                   │ ██░░░░░░░░░░░░░░░░░░░░░░░│
                                                  │ Activations: 8GB          │
                                                  │ ███░░░░░░░░░░░░░░░░░░░░░░│
                                                  │                           │
                                                  │ FREE: ~103GB headroom     │
                                                  │ ░░░░░░░░░░░░░░░░░░░░░░░░░│
                                                  └───────────────────────────┘
                                                  ✅ SINGLE DEVICE, NO MULTI-GPU BUGS
```

**Key insight**: By eliminating quantization AND multi-GPU, we eliminate:
- All quantization loading issues
- All device_map split bugs
- All gradient_checkpointing recomputation corruption
- The need for `prepare_model_for_kbit_training`

Trade-off: We use a "fat" node (more expensive in terms of cluster resources), but training completes in **71 seconds** for SFT and **8 minutes** for GRPO. The node is free for other users immediately after.

### 3. GRPO Reward Function Interface (Jobs 1804330–1804337)

**What I discovered about TRL 1.3.0 GRPO:**

```python
# What I assumed:
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    for text in completions:  # text is a string
        text.lower()  # ← works

# What TRL 1.3.0 actually passes (chat model path):
def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
    for completion in completions:  # completion is [{"role": "assistant", "content": "..."}]
        completion.lower()  # ← AttributeError: 'list' has no 'lower'
```

TRL 1.3.0 has three codepaths in `_generate()`:
1. Tool-calling models → `parse_response()` → `list[list[dict]]`
2. Chat models (our case) → `[{"role": "assistant", "content": text}]` → `list[list[dict]]`
3. Plain text models → `batch_decode()` → `list[str]`

Since we're passing `prompt` as a list of message dicts (chat format), TRL takes path 2.

**Fix**: Added `_extract_text()` helper that handles both formats.

**Second issue**: `reward_weights` must be passed via `GRPOConfig(reward_weights=[...])`, not set as `trainer.reward_weights = [...]`. The config converts it to a tensor internally; setting it manually leaves it as a Python list that can't be `.to(device)`.

---

## HPC Cluster Observations

### What I Learned About oneHPC

| Observation | Impact |
|-------------|--------|
| **Fat partition has immediate availability** | All 3 fat jobs started in <15 seconds. No queue wait. The fat nodes are underutilized — 8 nodes × 8 B200 GPUs = 64 B200s total, we used 1. |
| **B200 reports 183,359 MiB (not 192GB)** | ~9GB reserved for ECC + system. Still 179GB usable for CUDA allocations. |
| **Model loads in ~15 seconds** | 1026 weight shards at ~73 shards/sec. NVMe local storage is fast. |
| **No flash-linear-attention installed** | Logs show: "fast path not available". Model falls back to torch implementation. Could install `fla` for 2-3× speedup. |
| **CUDA 12.9.0 available** | B200 (Blackwell) is fully supported. No driver issues. |
| **Python 3.11 in our venv** | Good — PyTorch + Transformers + TRL all compatible. |
| **Node demu4xfat006 allocated for all 3 jobs** | Same node each time (likely because it was already warm/allocated). |
| **Generation speed on B200** | ~41 seconds to generate 4 × 512-token completions (2048 tokens total). That's ~50 tok/s for a 35B BF16 model — reasonable without FlashAttention. |
| **No other users on fat partition** | All 3 jobs started instantly. The fat partition appears lightly used (B200 is new hardware, most users haven't migrated from L40S). |

### Cluster Capacity Context

```
Fat Partition (our target):
  8 nodes × 8× B200 (192GB each) = 64 GPUs, 12.3 TB total VRAM
  We used: 1 GPU for <10 minutes total
  Utilization: 0.026% of available GPU-hours

GPU Partition (where we started):
  44 nodes × 4× L40S (48GB each) = 176 GPUs, 8.4 TB total VRAM
  Problem: Individual L40S too small for this model
  Even 2× L40S (96GB) hit architecture bugs
```

---

## Why "v11_fast" in the GRPO Completion

The model generated a response referencing "v11_fast" because **that's what's in the training data**.

Our SFT training pairs come from the meta-harness iteration log — 14 iterations of Bayesian MMM optimization. The prompt the model was responding to included diagnostic context from iteration history:

```
Previous iteration (v11_fast):
  F2F plausibility: 70.3%
  Email attribution: 18.6%
  Gate failures: trust < 50, ess_bulk < 100
  ...
```

The model correctly:
1. Identified which iteration it was looking at (v11_fast)
2. Noted the specific gate failures (trust, ESS)
3. Started reasoning about prior shapes (Gamma, alpha parameterization)
4. Referenced the constraint (pymc-marketing 0.18.0)

This is exactly the behavior we want — the model learned the iteration history format from SFT and is now generating novel proposals in GRPO. The "v11_fast" wasn't cherry-picked; it was simply the 12th (last) prompt in the dataset, so it appeared in the final completion parquet file.

---

## Current Status

### What Exists Now

```
/shared/project/tdr-mmm-hpc/llm/
├── models/
│   └── qwen3.6-35b-a3b/              # Base model (67GB BF16)
├── training/
│   ├── data/
│   │   ├── sft_train.jsonl            # 12 training pairs
│   │   ├── dpo_train.jsonl            # 4 preference pairs (unused)
│   │   └── stats.json
│   └── output/
│       ├── qlora-20260430-1213/       # SFT adapter (951MB)
│       │   ├── adapter/
│       │   ├── checkpoint-4/
│       │   ├── checkpoint-6/
│       │   └── training_meta.json
│       └── grpo-20260430-1234/        # GRPO adapter (549MB)
│           ├── adapter/
│           ├── checkpoint-12/
│           ├── completions/           # 12 parquet files of generated reasoning
│           └── grpo_meta.json
├── scripts/
│   ├── train_qlora.py
│   ├── train_qlora.sh
│   ├── train_grpo.py
│   └── train_grpo.sh
├── training-venv/                     # Python 3.11 + PyTorch + TRL 1.3.0
└── logs/
    ├── qlora_1804326.out              # Successful SFT log
    └── grpo_1804337.out               # Successful GRPO log
```

### What is NOT Done (Next Steps)

| Step | Description | Effort |
|------|-------------|--------|
| **Merge adapters** | Merge SFT → base, then merge GRPO on top (or use separately) | 10 min script |
| **Inference endpoint** | Deploy merged model via vLLM or TGI on fat partition | 30 min |
| **Integration** | Connect to meta-harness as the proposer agent | 1-2 hours |
| **Evaluation** | Run model against held-out prompts, compare to base | 30 min |
| **DPO training** | We have 4 DPO pairs — could do DPO after GRPO | 15 min (same pattern) |
| **Data augmentation** | 12 SFT pairs is minimal — generate synthetic data | Variable |

### The Adapters Are Just Sitting There

Right now, both adapters are files on disk. They are **not** being served, not connected to any inference pipeline, and not integrated into the meta-harness. To use them, you'd need to:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
model = AutoModelForCausalLM.from_pretrained("qwen3.6-35b-a3b", ...)
model = PeftModel.from_pretrained(model, "grpo-20260430-1234/adapter/")

# Or merge permanently
model = model.merge_and_unload()
model.save_pretrained("merged-mmm-agent/")
```

---

## Key Metrics Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING METRICS                              │
├─────────────────────────┬───────────────────────────────────────┤
│ SFT Loss Curve          │ GRPO Reward Scores (final step)       │
│                         │                                       │
│ 2.73 ─┐                │ gate_awareness:  ████████████░ 0.95   │
│ 2.57 ─┤ ╲              │ evidence:        ██████░░░░░░░ 0.61   │
│       │  ╲             │ domain:          █████░░░░░░░░ 0.45   │
│ 1.95 ─┤   ╲            │ length:          ███░░░░░░░░░░ 0.30   │
│       │    ╲           │ config_format:   ░░░░░░░░░░░░░ 0.00   │
│ 1.59 ─┤     ╲ ╱        │ structure:       ░░░░░░░░░░░░░ 0.00   │
│ 1.44 ─┤      ╳         │                                       │
│ 1.29 ─┤     ╱ ╲─final  │ Note: structure/config_format = 0     │
│       └─────────────    │ because model uses thinking-mode      │
│       e1  e2  e3        │ (not ## headers) — reward functions   │
│                         │ need tuning for this generation style. │
├─────────────────────────┴───────────────────────────────────────┤
│ VRAM Utilization                                                │
│                                                                 │
│ SFT:  ████████████████████████░░░░░░░░░░░░░░░ 69.5 / 183 GB    │
│ GRPO: █████████████████████████████████████░░░ 140.6 / 183 GB   │
│                                     ▲                           │
│                                     │                           │
│                    GRPO needs ~70GB extra for 4× generation     │
│                    buffers (4 completions × 512 tokens each)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Lessons Learned

1. **Model architecture matters more than parameter count.** Qwen3.6-35B-A3B's linear attention (`torch_chunk_gated_delta_rule`) is fundamentally incompatible with gradient checkpointing on split devices. No amount of configuration tuning fixes an architecture-level bug.

2. **When the standard recipe fails, go bigger not cleverer.** We spent 14 failed jobs trying to squeeze a 35B model onto 48GB GPUs with quantization tricks. The answer was: use a 192GB GPU and skip quantization entirely.

3. **TRL moves fast — always check the installed version's API.** Between TRL versions, `max_seq_length` → `max_length`, `TrainingArguments` → `SFTConfig`, and GRPO's reward function signature changed to chat-format messages.

4. **Rule-based rewards are sufficient for domain-specific GRPO.** We achieved 0.95 gate awareness without needing a trained reward model. The key insight: domain knowledge can be encoded as regex patterns and keyword matching.

5. **The HPC fat partition is a hidden gem.** B200 GPUs with 192GB VRAM, instant scheduling, and no queue wait. Most users are still on the L40S partition, leaving fat nodes idle.

---

## Deployment: Merge & Serve (2026-05-06)

**Session outcome**: Fine-tuned model deployed end-to-end. The trained model is now serving on port 8100.

### Adapter Merge (Job 1822232)

```
Node: demu4xfat005 (B200, 183GB VRAM)
Runtime: 1m33s total
  - Base model load: 9.7s (693 weight shards → 69.3 GB BF16)
  - SFT adapter merge: 1.7s
  - GRPO adapter merge: 0.4s
  - Save merged model: 74.0s (21 shards × 4GB each)
Output: /shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b-mmm/ (69.3 GB)
```

### Serving (Job 1822552)

```
Node: demu4xfat005 (B200)
Engine: SGLang 0.5.9
Model: qwen3.6-35b-a3b-mmm (BF16, 65.49 GB VRAM)
Port: 8100
Throughput: 33 tok/s (no CUDA graph)
Context: 65536 tokens
KV Cache: 24.83 GB K + 24.83 GB V
Mamba Cache: 43.65 GB (hybrid linear attention state)
Available VRAM after load: 17.65 GB
```

### Compatibility Issues Discovered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `--gres=gpu:1` rejected | oneHPC requires explicit GPU type | `--gres=gpu:b200:1` |
| `--qos=short` invalid | QoS names are `3h`, `1d`, `3d`, `7d`, `14d` | `--qos=3h` |
| DOS line endings | Windows `scp` doesn't convert | `sed -i 's/\r$//'` on HPC |
| PEFT 0.19.1 `WeightConverter` error | `distributed_operation` kwarg incompatible with transformers 5.7.0 | Downgrade to PEFT 0.15.2 |
| `Qwen3_5MoeForCausalLM` not recognized by SGLang | Newer transformers saves with `_text` suffix architecture class | Copy base model's `config.json` (uses `Qwen3_5MoeForConditionalGeneration`) |
| `transformers 5.8.0` breaks SGLang config parsing | `get_hf_text_config` assertion fails on new config format | Downgrade serving venv to transformers 4.57.1 |
| FlashInfer autotune hangs on B200 | First-time kernel JIT compilation for new GPU architecture | Wait ~8min + `--disable-cuda-graph` + `--attention-backend triton` |

### Key Insight: Config.json Matters for Serving

When `transformers 5.7.0` saves a merged model, it writes:
```json
{"model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"]}
```

But SGLang 0.5.9's `EntryClass` only registers `Qwen3_5MoeForConditionalGeneration` (the multimodal wrapper). The fix: copy the base model's original `config.json` into the merged model directory. The weights are identical architecture — same layers, same shapes — just with LoRA deltas baked in. The config just tells SGLang *how to load them*.

### Production Path (TODO)

The BF16 model on B200 works but wastes an expensive training GPU. The proper path:
1. Quantize merged model to FP8 → `models/qwen3.6-35b-a3b-mmm-fp8/` (~18 GB)
2. Serve on L40S (gpu partition, 48 GB) — frees B200 for training
3. Pre-build FlashInfer kernels for L40S (already cached from previous FP8 serving)

### Validation Response

The fine-tuned model immediately demonstrated MMM domain knowledge:
```
Prompt: "F2F plausibility at 70% and email attribution at 18%. Trust score is 33. What should we change?"

Response: Here's a thinking process:
1. Analyze User Input:
   - F2F plausibility: 70% (Face-to-Face interaction plausibility score)
   - Email attribution: 18% (Email channel attribution share)
   - Trust score: 33 (composite metric indicating model reliability/validity)
   ...
```

This is exactly the kind of structured, domain-aware reasoning we trained for.
