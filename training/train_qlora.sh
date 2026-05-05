#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=qlora-mmm
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --qos=3h
#SBATCH --time=02:55:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/qlora_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/qlora_%j.err

# =============================================================================
# LoRA Fine-Tuning — Qwen3.6-35B-A3B on 1× B200 (192GB)
# =============================================================================
# Method: BF16 base (no quantization) + LoRA r=64 adapters
# VRAM: ~67GB model + ~1GB LoRA + ~4GB optimizer + ~8GB activations = ~80GB
# B200 (192GB) handles this easily with room for longer sequences.
# Expected time: 20-60 min depending on dataset size
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/qwen3.6-35b-a3b
DATA_DIR=$LLM_DIR/training/data
OUTPUT_DIR=$LLM_DIR/training/output/qlora-$(date +%Y%m%d-%H%M)

echo "=== QLoRA TRAINING STARTED ==="
echo "NODE=$(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
echo "DATA=$DATA_DIR"
echo "OUTPUT=$OUTPUT_DIR"
echo "=============================="

# Load CUDA
module load cuda/12.9.0

# Memory optimization for large model loading
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate training venv
source $LLM_DIR/training-venv/bin/activate

# Verify data exists
if [ ! -f "$DATA_DIR/sft_train.jsonl" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/sft_train.jsonl"
    echo "Run extract_training_data.py first"
    exit 1
fi

SFT_COUNT=$(wc -l < "$DATA_DIR/sft_train.jsonl")
echo "SFT training examples: $SFT_COUNT"

mkdir -p "$OUTPUT_DIR"

# Run QLoRA training
python3 $LLM_DIR/scripts/train_qlora.py \
    --model-path "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --max-seq-len 4096 \
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
echo "=== TRAINING COMPLETE ==="
echo "Adapter at: $OUTPUT_DIR/adapter/"
echo "To serve with adapter: merge weights or load with --lora-path"
