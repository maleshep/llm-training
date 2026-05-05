#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=dpo-mmm
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --qos=3h
#SBATCH --time=02:55:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/dpo_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/dpo_%j.err

# =============================================================================
# DPO Training — MMM Proposer Agent on 1x B200 (192GB)
# =============================================================================
# Stage 3: runs AFTER GRPO. Aligns model using preference pairs.
# Requires: GRPO adapter from train_grpo.sh
# Method: BF16 base + LoRA + DPO with chosen/rejected pairs
# VRAM: ~67GB model + ~0.5GB LoRA + ~8GB optimizer + ~8GB activations = ~84GB
# B200 (192GB) handles this easily.
# Expected time: 15-30 min (small DPO dataset)
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/qwen3.6-35b-a3b
DATA_DIR=$LLM_DIR/training/data
OUTPUT_DIR=$LLM_DIR/training/output/dpo-$(date +%Y%m%d-%H%M)

# Find latest GRPO adapter
GRPO_ADAPTER=$(ls -td $LLM_DIR/training/output/grpo-*/adapter 2>/dev/null | head -1)

echo "=== DPO TRAINING STARTED ==="
echo "NODE=$(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
echo "GRPO_ADAPTER=$GRPO_ADAPTER"
echo "DATA=$DATA_DIR"
echo "OUTPUT=$OUTPUT_DIR"
echo "=============================="

module load cuda/12.9.0

# Memory optimization for large model loading
export PYTORCH_ALLOC_CONF=expandable_segments:True

source $LLM_DIR/training-venv/bin/activate

if [ -z "$GRPO_ADAPTER" ]; then
    echo "WARNING: No GRPO adapter found. Training DPO from base model."
    ADAPTER_FLAG=""
else
    echo "Using GRPO adapter: $GRPO_ADAPTER"
    ADAPTER_FLAG="--adapter-path $GRPO_ADAPTER"
fi

# Verify DPO data exists
if [ ! -f "$DATA_DIR/dpo_train.jsonl" ]; then
    echo "ERROR: DPO data not found at $DATA_DIR/dpo_train.jsonl"
    echo "Run extract_training_data.py first"
    exit 1
fi

DPO_COUNT=$(wc -l < "$DATA_DIR/dpo_train.jsonl")
echo "DPO preference pairs: $DPO_COUNT"

mkdir -p "$OUTPUT_DIR"

python3 $LLM_DIR/scripts/train_dpo.py \
    --model-path "$MODEL" \
    $ADAPTER_FLAG \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --lr 5e-7 \
    --beta 0.1 \
    --max-length 4096 \
    --lora-r 16 \
    --lora-alpha 32

echo ""
echo "=== GPU Memory After Training ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== Output Files ==="
ls -la "$OUTPUT_DIR/"
du -sh "$OUTPUT_DIR/"

echo ""
echo "=== DPO TRAINING COMPLETE ==="
echo "Adapter at: $OUTPUT_DIR/adapter/"
echo "Pipeline complete: SFT -> GRPO -> DPO"
