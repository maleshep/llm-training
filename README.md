# LLM Training: Qwen3.6-35B-A3B Domain Fine-Tuning

Fine-tuning a 35B Mixture-of-Experts model for domain-specific optimization using LoRA + GRPO (Group Relative Policy Optimization) on a single NVIDIA B200 GPU (192GB VRAM).

**From 22 failed jobs to successful training in one session** — a complete chronicle of debugging a novel model architecture on HPC infrastructure.

## What's Here

```
.
├── train_qlora.py          # SFT (Supervised Fine-Tuning) with LoRA
├── train_qlora.sh          # Slurm job script for SFT
├── train_grpo.py           # GRPO reinforcement learning with rule-based rewards
├── train_grpo.sh           # Slurm job script for GRPO
├── training_chronicle.html # Interactive visualization of the full training journey
├── extract_training_data.py # Extracts training pairs from iteration history
├── setup_training_env.sh   # Environment setup for HPC cluster
├── data/
│   └── example_sft.jsonl   # Example training data format (synthetic)
└── docs/
    ├── WATCHOUTS.md        # 10 critical lessons learned
    └── ARCHITECTURE.md     # Model architecture notes
```

## The Journey

| Phase | Duration | Jobs | Outcome |
|-------|----------|------|---------|
| L40S attempts (4-bit, FP8, multi-GPU) | 110 min | 22 | All failed |
| **Pivot to B200 (BF16, single GPU)** | 5 min | — | Decision point |
| SFT training | 71 sec | 1 | Loss 2.57 → 1.29, token acc 68% |
| GRPO debugging | 2 min | 2 | API format issues |
| **GRPO training** | 8.2 min | 1 | 12 steps, 6 reward functions |

**Total GPU-hours consumed: 0.82** (mostly failed attempts on the wrong hardware)

## Key Insight

**Qwen3.6-35B-A3B uses `torch_chunk_gated_delta_rule` linear attention** — a novel architecture that is fundamentally incompatible with gradient checkpointing across GPU boundaries. When model weights are split across GPUs, the recomputation pass during backward produces corrupted logits, triggering NLL assertion failures.

The only working solution: load the full BF16 model (67GB) on a single GPU with enough VRAM (192GB B200). No quantization, no multi-GPU, no gradient checkpoint issues.

## Model Details

| Property | Value |
|----------|-------|
| Model | Qwen3.6-35B-A3B (MoE) |
| Total params | 34.7B |
| Active params/token | 3B |
| Attention | Linear (chunk gated delta rule) |
| Vocab size | 248,320 |
| Weight shards | 1,026 files |
| BF16 size on disk | ~67 GB |

## Training Configuration

### SFT (LoRA Fine-Tuning)

```python
# LoRA config
r=64, alpha=128, dropout=0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
# Result: 33.4M trainable / 34.7B total (0.10%)

# Training
epochs=3, lr=2e-4, batch=1, grad_accum=8
gradient_checkpointing=True, bf16=True
# VRAM peak: ~80 GB (43.6% of 183 GB usable)
```

### GRPO (Reinforcement Learning)

```python
# Group generation
num_generations=4, max_completion_length=512
temperature=0.7, top_p=0.9

# Policy optimization
lr=5e-6, beta=0.01 (KL penalty)
loss_type="grpo", scale_rewards="group"

# 6 rule-based reward functions (no reward model needed)
reward_weights = [0.15, 0.20, 0.20, 0.20, 0.15, 0.10]
# VRAM peak: ~140 GB (76.5% of 183 GB usable)
```

## GRPO Reward Functions

The key innovation: **rule-based rewards** that score completions on multiple axes without a separate reward model:

| Function | Weight | What it rewards |
|----------|--------|----------------|
| `reward_structure` | 0.15 | Structured reasoning (sections, headers) |
| `reward_config_format` | 0.20 | Properly formatted configuration proposals |
| `reward_gate_awareness` | 0.20 | Mentioning evaluation criteria with correct thresholds |
| `reward_evidence_reasoning` | 0.20 | Quantitative, causal, domain-specific reasoning |
| `reward_domain_correctness` | 0.15 | Known-good patterns; penalizes known-bad |
| `reward_length` | 0.10 | Penalizes too-short or too-long responses |

## Hardware

- **GPU**: NVIDIA B200 (192 GB VRAM, 183 GB usable after ECC)
- **Partition**: HPC "fat" nodes (8 nodes × 8 B200s = 64 GPUs)
- **Queue wait**: Zero (partition drastically underutilized)
- **Storage**: NVMe-backed shared filesystem (1026 shards load in 14 seconds)
- **Scheduler**: Slurm 24.11

## Critical Watchouts

See [`docs/WATCHOUTS.md`](docs/WATCHOUTS.md) for the full list. Top 3:

1. **DO NOT use multi-GPU for this model's training** — linear attention + gradient checkpointing + device split = corrupted logits
2. **DO NOT quantize for training** — FP8 has no backward pass; 4-bit requires multi-GPU (see #1)
3. **TRL 1.3.0 GRPO passes chat completions as `list[list[dict]]`** — reward functions must handle `[{"role": "assistant", "content": "..."}]` format, not plain strings

## Visualization

Open `training_chronicle.html` in a browser for the full interactive chronicle with:
- VRAM allocation diagrams
- Decision flow showing why each pivot happened
- Complete 26-job timeline with error classifications
- GRPO reward score bars
- HPC cluster utilization observations

## Software Stack

```
torch==2.9.1+cu128
transformers>=4.57
trl==1.3.0
peft>=0.15
datasets>=3.0
```

## License

MIT — educational/reference use. The training scripts are generic and work with any chat-format model.
