# Model Training — HPC Workloads

## Vision: How Can We Use HPC?

This repository captures everything we run on oneHPC (Merck's on-premises Slurm cluster in Munich).
Three workloads, one cluster, maximum leverage:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        oneHPC (Munich, Equinix MU4)                  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  1. TRAIN        │  │  2. SERVE        │  │  3. WISPRFLOW    │  │
│  │                  │  │                  │  │                  │  │
│  │  Pharma Agent    │  │  Qwen3.6 LLM     │  │  ASR + TTS       │  │
│  │  SFT + GRPO     │  │  SGLang FP8      │  │  Qwen2-Audio     │  │
│  │  B200 (192 GB)  │  │  L40S (48 GB)    │  │  + CosyVoice3    │  │
│  │                  │  │  Port 8100       │  │  Ports 8200/8300 │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
│  Access: SSH tunnel from Windows → Pi CLI / WisprFlow client        │
└─────────────────────────────────────────────────────────────────────┘
```

## Workloads

### 1. Pharma Agent Training (DONE)

Fine-tune Qwen3.6-35B-A3B (MoE, 35B total / 3B active params) into a pharma-specific
optimization agent. Two-phase approach:

- **SFT** — Supervised fine-tuning on domain iteration data (config → diagnosis → changes)
- **GRPO** — Group Relative Policy Optimization for reward-driven refinement

Hardware: 1× B200 (192 GB VRAM), fat partition. Single-GPU only — MoE linear attention
(`torch_chunk_gated_delta_rule`) is incompatible with multi-GPU gradient checkpointing.

Status: Adapters trained and on disk. Chronicle documents 19 training jobs and all lessons learned.

### 2. Qwen3.6 Inference Serving (RUNNING)

Serve the trained model for daily use via SGLang:

| Component | Detail |
|-----------|--------|
| Model | Qwen3.6-35B-A3B-FP8 (quantized for inference) |
| Engine | SGLang 0.5.9 |
| GPU | 1× L40S (48 GB), ~18 GB VRAM used |
| Speed | 90–103 tok/s |
| Context | 32K tokens |
| Port | 8100 |
| QoS | 3-day (resubmit weekly) |

Access from Windows via SSH tunnel + Pi CLI (`omp --model hpc-qwen/qwen3.6-35b-a3b`).

### 3. WisprFlow — Speech Pipeline (IN PROGRESS)

Self-hosted speech-to-text and voice cloning on HPC. No Azure, no cloud APIs.

**The flow:**
1. User presses hotkey on Windows
2. Audio streams/sends to HPC
3. Qwen3-ASR-1.7B transcribes (ASR)
4. Text returns to cursor position
5. (Optional) Qwen3-TTS-1.7B generates spoken response (TTS/clone)

**Models (LIVE as of 2026-05-01, Job 1811142 on demu4xgpu006):**
| Model | Task | Params | VRAM | Latency |
|-------|------|--------|------|---------|
| Qwen3-ASR-1.7B | ASR | 1.7B | 4.1 GB | 189ms (warm) |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS/Clone | 1.7B | 4.2 GB | ~7.5s/sentence |

Total: 9.5 GB / 46 GB on L40S — 80% headroom remaining.

## Directory Structure

```
model_training/
├── README.md                 ← You are here
├── training/                 ← SFT + GRPO training scripts and configs
│   ├── train_qlora.py
│   ├── train_qlora.sh
│   ├── train_grpo.py
│   ├── train_grpo.sh
│   └── setup_training_env.sh
├── serving/                  ← SGLang inference + model download
│   ├── serve-qwen36-fp8.sh
│   ├── serve-qwen36-sglang.sh
│   ├── download-fp8.sh
│   ├── fix-cuda-and-serve.sh
│   ├── benchmark.sh
│   ├── test-sglang-fp8.sh
│   └── test-sglang-tp2.sh
├── wisprflow/                ← ASR + TTS speech pipeline (LIVE)
│   ├── README.md
│   ├── serve-unified.sh     ← Slurm job (ASR+TTS on one L40S)
│   ├── asr_server.py        ← FastAPI wrapper for Qwen3-ASR-1.7B
│   ├── tts_server.py        ← FastAPI wrapper for Qwen3-TTS-1.7B
│   ├── download-models.sh   ← One-time model download
│   └── client/              ← Windows-side hotkey client + animation
│       ├── wisprflow.pyw
│       └── requirements.txt
├── data/                     ← Training data (SFT/DPO JSONL)
│   ├── sft_train.jsonl
│   ├── dpo_train.jsonl
│   └── stats.json
├── docs/                     ← Documentation and narrative
│   ├── CHRONICLE.md
│   ├── training_chronicle.html
│   ├── README-pi-setup.md
│   └── ARCHITECTURE.md
├── access/                   ← Pi CLI config + SSH tunnel helpers
│   ├── start-tunnel.sh
│   └── start-pi.bat
└── scripts/                  ← Utilities
    └── extract_training_data.py
```

## Access Pattern

```
Windows (local)                              oneHPC (compute node)
┌─────────────────┐                         ┌─────────────────────────┐
│ Pi CLI (omp)    │───── port 8100 ────────▶│ SGLang (Qwen3.6 FP8)   │
│ WisprFlow client│───── port 8200 ────────▶│ ASR (Qwen2-Audio-7B)   │
│                 │───── port 8300 ────────▶│ TTS (CosyVoice3-0.5B)  │
└─────────────────┘                         └─────────────────────────┘
         ▲                                            ▲
         └──── SSH tunnel ────────────────────────────┘
              ssh -L 8100:NODE:8100 -L 8200:NODE:8200 -L 8300:NODE:8300
```

Quick start:
```bash
# Start tunnel (all services)
ssh -L 8100:demu4xgpu002:8100 \
    -L 8200:demu4xgpu002:8200 \
    -L 8300:demu4xgpu002:8300 \
    -N M316235@onehpc.merckgroup.com

# Use LLM
omp --model hpc-qwen/qwen3.6-35b-a3b

# Use WisprFlow (press Ctrl+Shift+Space, speak, text appears at cursor)
python wisprflow/client/wisprflow.pyw
```

## HPC Details

- **Account**: `tdr-mmm-hpc`
- **Project storage**: `/shared/project/tdr-mmm-hpc/`
- **Login**: `ssh M316235@onehpc.merckgroup.com`
- **Scheduler**: Slurm 24.11
- **GPU partitions**: `gpu` (L40S 48GB ×4/node), `fat` (B200 192GB ×8/node)
