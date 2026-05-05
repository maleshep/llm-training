#!/bin/bash
# =============================================================================
# Setup training environment on HPC for QLoRA/GRPO
# Run ONCE on login node (I/O-bound installs are fine there)
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
VENV_DIR=$LLM_DIR/training-venv

echo "=== Setting up LLM training environment ==="

# Load modules
module load cuda/12.9.0
module load cli-tools

# Create venv with uv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install training stack
echo "Installing packages..."
uv pip install \
    torch==2.9.1 \
    transformers>=4.57.0 \
    peft>=0.15.0 \
    trl>=0.18.0 \
    bitsandbytes>=0.45.0 \
    datasets>=3.0.0 \
    accelerate>=1.5.0 \
    flash-attn>=2.7.0 \
    pyyaml \
    sentencepiece \
    protobuf

echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
import transformers
import peft
import trl
import bitsandbytes
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'BitsAndBytes: {bitsandbytes.__version__}')
"

# Create directory structure
echo ""
echo "=== Creating directory structure ==="
mkdir -p $LLM_DIR/training/data
mkdir -p $LLM_DIR/training/output
mkdir -p $LLM_DIR/scripts
mkdir -p $LLM_DIR/logs

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo "Scripts: $LLM_DIR/scripts/"
echo "Data: $LLM_DIR/training/data/"
echo "Output: $LLM_DIR/training/output/"
echo ""
echo "Next steps:"
echo "  1. Upload training scripts: rsync model_training/hpc/llm/*.py M316235@onehpc.merckgroup.com:$LLM_DIR/scripts/"
echo "  2. Upload training data: rsync model_training/hpc/llm/data/ M316235@onehpc.merckgroup.com:$LLM_DIR/training/data/"
echo "  3. Submit QLoRA job: sbatch $LLM_DIR/scripts/train_qlora.sh"
echo "  4. After QLoRA: sbatch $LLM_DIR/scripts/train_grpo.sh"
