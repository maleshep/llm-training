#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=merge-adapters
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --time=0-01:00:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/merge_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/merge_%j.err

# =============================================================================
# Merge LoRA Adapters → Single Model for Serving
# =============================================================================
# Loads base Qwen3.6-35B-A3B (BF16) + SFT + GRPO adapters
# Merges into single checkpoint at models/qwen3.6-35b-a3b-mmm/
# Runtime: ~2-3 minutes on B200
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
TRAINING_DIR=$LLM_DIR/training

module load cuda/12.9.0
source $TRAINING_DIR/training-venv/bin/activate

echo "=== ADAPTER MERGE ==="
echo "NODE=$(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# Auto-find latest adapters
SFT_ADAPTER=$(ls -td $TRAINING_DIR/output/qlora-*/adapter 2>/dev/null | head -1)
GRPO_ADAPTER=$(ls -td $TRAINING_DIR/output/grpo-*/adapter 2>/dev/null | head -1)
DPO_ADAPTER=$(ls -td $TRAINING_DIR/output/dpo-*/adapter 2>/dev/null | head -1)

if [ -z "$SFT_ADAPTER" ]; then
    echo "ERROR: No SFT adapter found in $TRAINING_DIR/output/qlora-*/adapter"
    exit 1
fi

echo "SFT adapter:  $SFT_ADAPTER"
echo "GRPO adapter: ${GRPO_ADAPTER:-'(none)'}"
echo "DPO adapter:  ${DPO_ADAPTER:-'(none)'}"
echo ""

# Build command
CMD="python $TRAINING_DIR/../scripts/merge_and_export.py \
    --base $LLM_DIR/models/qwen3.6-35b-a3b \
    --sft-adapter $SFT_ADAPTER \
    --output $LLM_DIR/models/qwen3.6-35b-a3b-mmm"

if [ -n "$GRPO_ADAPTER" ]; then
    CMD="$CMD --grpo-adapter $GRPO_ADAPTER"
fi

if [ -n "$DPO_ADAPTER" ]; then
    CMD="$CMD --dpo-adapter $DPO_ADAPTER"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=== MERGE DONE ==="
echo "Model at: $LLM_DIR/models/qwen3.6-35b-a3b-mmm/"
echo ""
echo "Next: Update serving/serve-qwen36-fp8.sh MODEL path, or run:"
echo "  sbatch serving/serve-mmm.sh"
