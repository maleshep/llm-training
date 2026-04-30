#!/bin/bash
#SBATCH --account=my-project
#SBATCH --job-name=grpo-qwen
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --qos=3h
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

# GRPO (Group Relative Policy Optimization) on Qwen3.6-35B-A3B
#
# Builds on the LoRA adapter from SFT. Uses 6 rule-based reward functions
# to reinforce good domain-specific reasoning without a reward model.
#
# VRAM budget is HIGHER than SFT (~96GB vs ~80GB) because GRPO generates
# multiple completions per prompt (num_generations=4).
#
# Usage:
#   sbatch train_grpo.sh
#   sbatch train_grpo.sh --export=ALL,NUM_GEN=6,MAX_COMP=256

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
ADAPTER_PATH="${ADAPTER_PATH:-./output/qlora-v1/adapter}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/grpo-v1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"
NUM_GEN="${NUM_GEN:-4}"
MAX_COMP="${MAX_COMP:-512}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

echo "============================================================"
echo "GRPO Training: Qwen3.6-35B-A3B"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:      $MODEL_PATH"
echo "Adapter:    $ADAPTER_PATH"
echo "Data:       $DATA_DIR"
echo "Output:     $OUTPUT_DIR"
echo "GRPO:       num_generations=$NUM_GEN, max_completion=$MAX_COMP"
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

# Check if SFT adapter exists (optional but recommended)
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "WARNING: SFT adapter not found at $ADAPTER_PATH"
    echo "GRPO will train from scratch (no SFT warmstart)."
    ADAPTER_ARG=""
else
    ADAPTER_ARG="--adapter-path $ADAPTER_PATH"
fi

# --- Run GRPO training ---
uv run python train_grpo.py \
    --model-path "$MODEL_PATH" \
    $ADAPTER_ARG \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-generations "$NUM_GEN" \
    --max-completion-length "$MAX_COMP" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA"

echo ""
echo "GRPO training complete. Adapter saved to: $OUTPUT_DIR/adapter"
echo "Job $SLURM_JOB_ID finished at $(date)"
