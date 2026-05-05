# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

HPC workloads running on oneHPC (Merck's on-premises Slurm cluster, Munich Equinix MU4). Three workloads share the cluster:

1. **Training** — SFT + GRPO fine-tuning of Qwen3.6-35B-A3B into a pharma MMM optimization agent (completed)
2. **Serving** — Qwen3.6-35B-A3B-FP8 inference via SGLang on L40S (port 8100)
3. **WisprFlow** — Self-hosted ASR (Qwen3-ASR-1.7B, port 8200) + TTS (Qwen3-TTS-1.7B, port 8300) speech pipeline

## HPC Environment

- **Account**: `tdr-mmm-hpc`
- **Login**: `ssh M316235@onehpc.merckgroup.com`
- **Project storage**: `/shared/project/tdr-mmm-hpc/`
- **Scheduler**: Slurm 24.11
- **Partitions**: `gpu` (L40S 48GB, 4/node) for serving; `fat` (B200 192GB, 8/node) for training
- **Python**: 3.11 in venvs
- **CUDA**: 12.9.0 (`module load cuda/12.9.0`)

## Key Commands

### Submit Slurm Jobs

```bash
# Training (B200 fat partition)
sbatch training/train_qlora.sh   # SFT — ~71s on 1× B200
sbatch training/train_grpo.sh    # GRPO — ~8min on 1× B200

# Serving (L40S gpu partition)
sbatch serving/serve-qwen36-fp8.sh     # SGLang inference, 3-day QoS
sbatch wisprflow/serve-unified.sh      # ASR+TTS+LLM unified, 3-day QoS
```

### SSH Tunnels (run on Windows)

```bash
# LLM only
ssh -L 8100:NODE:8100 -N M316235@onehpc.merckgroup.com

# All services
ssh -L 8100:NODE:8100 -L 8200:NODE:8200 -L 8300:NODE:8300 -N M316235@onehpc.merckgroup.com
```

Node name comes from: `ssh M316235@onehpc.merckgroup.com "cat /shared/project/tdr-mmm-hpc/llm/.serve-state.json"`

### Data Preparation

```bash
python scripts/extract_training_data.py --mmm-root ../../marketing-mix
```

Extracts SFT/DPO pairs from the meta-harness iteration log into `data/sft_train.jsonl` and `data/dpo_train.jsonl`.

### Local Client

```bash
# Pi CLI for LLM
omp --model hpc-qwen/qwen3.6-35b-a3b

# WisprFlow speech client
python wisprflow/client/wisprflow.pyw
```

## Architecture Constraints

### Training (Critical)

- Qwen3.6-35B-A3B uses `torch_chunk_gated_delta_rule` linear attention — **incompatible with gradient checkpointing on multi-GPU**. The NLL assertion crash is an architecture bug, not a data issue.
- Only working recipe: **BF16 on single B200 (192GB), no quantization, `device_map={"": 0}`**. Do not attempt 4-bit/FP8 quantization or multi-GPU splits for training.
- TRL 1.3.0: GRPO reward functions receive `list[list[dict]]` (chat format), not `list[str]`. Use `_extract_text()` helper.
- `reward_weights` must be in `GRPOConfig(...)`, not set on trainer instance.

### Serving

- SGLang 0.5.9 serves FP8-quantized model (~18GB VRAM on L40S)
- OpenAI-compatible API at port 8100, served model name: `qwen3.6-35b-a3b`

### WisprFlow

- ASR and TTS require **separate venvs** (`venv-asr`, `venv-tts`) due to conflicting transformers versions (4.57.6 vs 4.57.3)
- ASR: Qwen3-ASR-1.7B, 4.1GB VRAM, 189ms warm latency
- TTS: Qwen3-TTS-12Hz-1.7B-CustomVoice, 4.2GB VRAM

## Key Dependencies

| Component | Stack |
|-----------|-------|
| Training | PyTorch + Transformers + TRL 1.3.0 + PEFT + bitsandbytes |
| Serving | SGLang 0.5.9 |
| ASR server | FastAPI + Transformers + librosa |
| TTS server | FastAPI + Transformers (qwen-tts) |
| Windows client | pyaudio + keyboard + requests + pywin32 |

## On-Disk Layout (HPC)

```
/shared/project/tdr-mmm-hpc/llm/
├── models/                    # Downloaded model weights
│   ├── qwen3.6-35b-a3b/      # Base (67GB BF16)
│   ├── qwen3.6-35b-a3b-fp8/  # Pre-quantized for serving
│   ├── qwen3-asr-1.7b/
│   └── qwen3-tts-1.7b/
├── training/
│   ├── data/                  # sft_train.jsonl, dpo_train.jsonl
│   └── output/                # Timestamped adapter dirs
├── training-venv/             # Python 3.11 + TRL 1.3.0
├── venv/                      # SGLang serving venv
└── .serve-state.json          # Current serving node/port info
```

## LoRA Configuration (for reference)

Both SFT and GRPO use identical LoRA setup:
- rank=64, alpha=128, dropout=0.05
- Targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 33.4M trainable params / 34.7B total (0.096%)
