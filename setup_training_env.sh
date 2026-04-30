#!/bin/bash
# Setup training environment for Qwen3.6-35B-A3B fine-tuning
#
# Prerequisites:
#   - CUDA 12.8+ capable GPU (B200 recommended for 192GB VRAM)
#   - Python 3.11+
#   - uv (recommended) or pip
#
# Usage:
#   bash setup_training_env.sh [--project-dir /path/to/project]

set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
echo "Setting up training environment in: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Check prerequisites ---
echo ""
echo "Checking prerequisites..."

# Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+."
    exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VERSION"

# CUDA
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU_INFO"
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  Driver: $CUDA_VERSION"
else
    echo "  WARNING: nvidia-smi not found. GPU training may not work."
fi

# --- Create project structure ---
echo ""
echo "Creating project structure..."
mkdir -p models data output logs

# --- Install dependencies ---
echo ""
echo "Installing Python dependencies..."

if command -v uv &>/dev/null; then
    echo "Using uv (fast)..."
    uv init --no-readme 2>/dev/null || true
    uv add \
        "torch>=2.9" \
        "transformers>=4.57" \
        "trl==1.3.0" \
        "peft>=0.15" \
        "datasets>=3.0" \
        "accelerate>=1.0" \
        "bitsandbytes>=0.45"
    echo ""
    echo "Run scripts with: uv run python train_qlora.py ..."
else
    echo "Using pip..."
    python3 -m pip install --upgrade pip
    python3 -m pip install \
        "torch>=2.9" \
        "transformers>=4.57" \
        "trl==1.3.0" \
        "peft>=0.15" \
        "datasets>=3.0" \
        "accelerate>=1.0" \
        "bitsandbytes>=0.45"
    echo ""
    echo "Run scripts with: python train_qlora.py ..."
fi

# --- Verify installation ---
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
import trl
import peft
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- Download model (instructions) ---
echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Download the model:"
echo "   huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir ./models/qwen3.6-35b-a3b"
echo ""
echo "   NOTE: This is ~67GB. On HPC, download to shared storage to avoid quota issues."
echo "   The model has 1,026 weight shards — expect ~14 seconds load time on NVMe."
echo ""
echo "2. Prepare your training data:"
echo "   Place your SFT data in ./data/sft_train.jsonl"
echo "   See ./data/example_sft.jsonl for the expected format."
echo ""
echo "3. Run SFT training:"
echo "   python train_qlora.py --data-dir ./data"
echo "   (or: sbatch train_qlora.sh)"
echo ""
echo "4. Run GRPO (after SFT):"
echo "   python train_grpo.py --adapter-path ./output/qlora-v1/adapter"
echo "   (or: sbatch train_grpo.sh)"
echo ""
echo "IMPORTANT: See docs/WATCHOUTS.md before your first training run."
