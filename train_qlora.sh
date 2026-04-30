#!/bin/bash
#SBATCH --account=my-project
#SBATCH --job-name=sft-qwen
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --qos=3h
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# LoRA fine-tuning of Qwen3.6-35B-A3B on a single B200 (192GB VRAM)
#
# IMPORTANT: This model MUST run on a single GPU with enough VRAM for
# the full BF16 model (~67GB). Multi-GPU and quantization are incompatible
# with the model's linear attention (torch_chunk_gated_delta_rule).
# See docs/WATCHOUTS.md for details.
#
# Usage:
#   sbatch train_qlora.sh
#   sbatch train_qlora.sh --export=ALL,EPOCHS=5,LR=1e-4

set -euo pipefail

# --- Environment ---
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Load modules (adjust for your cluster)
module purge
module load cuda/12.9
module load cli-tools  # provides uv

# Create logs directory
mkdir -p logs

# --- Configuration (overridable via --export) ---
MODEL_PATH="${MODEL_PATH:-./models/qwen3.6-35b-a3b}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/qlora-v1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

echo "============================================================"
echo "SFT Training: Qwen3.6-35B-A3B"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:      $MODEL_PATH"
echo "Data:       $DATA_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Config:     epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
echo "LoRA:       r=$LORA_R, alpha=$LORA_ALPHA"
echo "============================================================"

# --- Pre-flight checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_DIR/sft_train.jsonl" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/sft_train.jsonl"
    exit 1
fi

# Check GPU VRAM (need ~80GB for BF16 + LoRA)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -lt 100000 ]; then
    echo "WARNING: GPU has ${GPU_MEM_MB}MB VRAM. BF16 training needs ~80GB."
    echo "Consider using a GPU with >=192GB (e.g., B200)."
fi

# --- Run training ---
uv run python train_qlora.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA"

echo ""
echo "SFT training complete. Adapter saved to: $OUTPUT_DIR/adapter"
echo "Job $SLURM_JOB_ID finished at $(date)"
