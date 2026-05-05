#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=llm-download-fp8
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=3h
#SBATCH --time=02:55:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/download_fp8_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/download_fp8_%j.err

set -euo pipefail

LLM_DIR=/shared/project/tdr-mmm-hpc/llm
MODEL_DIR=$LLM_DIR/models/qwen3.6-35b-a3b-fp8

source $LLM_DIR/venv/bin/activate

echo "=== Downloading Qwen3.6-35B-A3B-FP8 ==="
echo "Target: $MODEL_DIR"
echo "Expected size: ~37.5 GB"
echo ""

python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import time

start = time.time()
print("Starting download...")
path = snapshot_download(
    repo_id="Qwen/Qwen3.6-35B-A3B-FP8",
    local_dir="/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b-fp8",
    local_dir_use_symlinks=False,
    resume_download=True
)
elapsed = time.time() - start
print(f"Download complete: {path}")
print(f"Time: {elapsed/60:.1f} minutes")
PYEOF

echo ""
echo "=== Download complete ==="
du -sh $MODEL_DIR
ls -la $MODEL_DIR/*.safetensors | wc -l
echo "safetensor files downloaded"
