# Pi (oh-my-pi) + HPC Qwen3.6 Setup

## Architecture

```
┌─────────────────────────────┐     SSH Tunnel      ┌─────────────────────────────────┐
│  Windows PC (Local)         │  localhost:8100 ───► │  oneHPC GPU Node                │
│                             │                      │                                 │
│  Pi CLI (omp v14.5.5)      │                      │  SGLang 0.5.9                   │
│  └─ models.yml             │                      │  └─ Qwen3.6-35B-A3B-FP8        │
│     └─ hpc-qwen provider   │                      │     └─ 1× NVIDIA L40S (48GB)   │
│        └─ localhost:8100/v1 │                      │     └─ 90-103 tok/s             │
└─────────────────────────────┘                      └─────────────────────────────────┘
```

## Quick Start

### 1. Start SSH Tunnel (Terminal 1)

```bash
ssh -L 8100:demu4xgpu002:8100 -N M316235@onehpc.merckgroup.com
```

Or use the helper script:
```bash
bash marketing-mix/compute/hpc/llm/start-tunnel.sh
```

> Note: The node name may change if the Slurm job is restarted. Check with:
> `ssh M316235@onehpc.merckgroup.com "cat /shared/project/tdr-mmm-hpc/llm/.serve-state.json"`

### 2. Launch Pi (Terminal 2)

```bash
omp --model hpc-qwen/qwen3.6-35b-a3b
```

Or for a one-shot:
```bash
omp -p --model hpc-qwen/qwen3.6-35b-a3b "What is marketing mix modelling?"
```

Or use the batch file:
```
start-pi.bat
```

### 3. Verify Connection

```bash
curl http://localhost:8100/health
curl http://localhost:8100/v1/models
```

## Configuration Files

| File | Purpose |
|------|---------|
| `~/.omp/agent/models.yml` | Provider & model definitions |
| `~/.omp/agent/config.yml` | Global settings (theme, roles, retry) |
| `%LOCALAPPDATA%\Programs\omp\omp.exe` | Binary (326MB) |
| `%LOCALAPPDATA%\Programs\omp\pi_natives.win32-x64-baseline.node` | Native addon (96MB) |

## HPC Server Details

| Property | Value |
|----------|-------|
| Engine | SGLang 0.5.9 |
| Model | Qwen3.6-35B-A3B-FP8 |
| GPU | 1× NVIDIA L40S 48GB |
| Port | 8100 |
| Context | 32,768 tokens |
| Speed | 90-103 tok/s (single), 148 tok/s (concurrent) |
| TTFT | ~200ms |
| Slurm Account | tdr-mmm-hpc |
| QoS | 3d (3-day wall time) |
| Served Model Name | `qwen3.6-35b-a3b` |

## Switching Models

Pi supports model cycling with Ctrl+P. You can also switch inline:

```bash
omp --model claude-sonnet-4-6    # Use Bedrock Claude
omp --model qwen3.6              # Fuzzy-matches our HPC Qwen (free, fast)
omp --model deepseek.v3          # Use Bedrock DeepSeek
```

## Troubleshooting

**"Connection refused on localhost:8100"**
→ SSH tunnel is not running. Start it first.

**"Model not found"**  
→ Check `~/.omp/agent/models.yml` exists and has correct YAML syntax.

**"Timeout / no response"**
→ HPC job may have expired (3-day max). Resubmit:
```bash
ssh M316235@onehpc.merckgroup.com "sbatch /shared/project/tdr-mmm-hpc/llm/scripts/serve-qwen36-fp8.sh"
```

**Node name changed after restart?**
→ Read the state file:
```bash
ssh M316235@onehpc.merckgroup.com "cat /shared/project/tdr-mmm-hpc/llm/.serve-state.json"
```
Then update your tunnel command with the new node.

## Maintenance Window

oneHPC maintenance: **May 4-26, 2026** — all nodes unavailable.
Current job (1782730) must complete before May 4.
