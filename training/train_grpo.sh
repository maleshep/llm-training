#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=grpo-mmm
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --qos=3h
#SBATCH --time=02:55:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/grpo_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/grpo_%j.err

# =============================================================================
# GRPO Training — MMM Proposer Agent on 1× B200 (192GB)
# =============================================================================
# Requires: SFT adapter from train_qlora.sh (optional — can train from base)
# Method: BF16 base + LoRA + GRPO with rule-based rewards
# VRAM: ~67GB model + ~1GB LoRA + ~20GB generations + ~8GB optimizer = ~96GB
# B200 (192GB) handles this with margin for num_generations=4
# Expected time: 30-90 min depending on num_generations and dataset size
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/qwen3.6-35b-a3b
DATA_DIR=$LLM_DIR/training/data
OUTPUT_DIR=$LLM_DIR/training/output/grpo-$(date +%Y%m%d-%H%M)

# Find latest SFT adapter
SFT_ADAPTER=$(ls -td $LLM_DIR/training/output/qlora-*/adapter 2>/dev/null | head -1)

echo "=== GRPO TRAINING STARTED ==="
echo "NODE=$(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
echo "SFT_ADAPTER=$SFT_ADAPTER"
echo "DATA=$DATA_DIR"
echo "OUTPUT=$OUTPUT_DIR"
echo "=============================="

module load cuda/12.9.0

# Memory optimization for large model loading + generation
export PYTORCH_ALLOC_CONF=expandable_segments:True

source $LLM_DIR/training-venv/bin/activate

if [ -z "$SFT_ADAPTER" ]; then
    echo "WARNING: No SFT adapter found. Training GRPO from base model."
    ADAPTER_FLAG=""
else
    echo "Using SFT adapter: $SFT_ADAPTER"
    ADAPTER_FLAG="--adapter-path $SFT_ADAPTER"
fi

mkdir -p "$OUTPUT_DIR"

python3 $LLM_DIR/scripts/train_grpo.py \
    --model-path "$MODEL" \
    $ADAPTER_FLAG \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-generations 4 \
    --epochs 1 \
    --lr 5e-6 \
    --max-completion-length 512 \
    --lora-r 64 \
    --lora-alpha 128

echo ""
echo "=== GPU Memory After Training ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== Output Files ==="
ls -la "$OUTPUT_DIR/"
du -sh "$OUTPUT_DIR/"

echo ""
echo "=== GRPO TRAINING COMPLETE ==="
echo "Adapter at: $OUTPUT_DIR/adapter/"
