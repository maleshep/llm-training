#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=vllm-serve-qwen36
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3h
#SBATCH --time=02:55:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/serve_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/serve_%j.err

# =============================================================================
# vLLM Inference Server — Qwen3.6-35B-A3B on 1× L40S
# =============================================================================
# Port: 8100 (OpenAI-compatible API)
# Model: Qwen/Qwen3.6-35B-A3B (MoE: 35B total, 3B active per token)
# VRAM: ~40GB / 48GB available
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
VENV=$LLM_DIR/venv
MODEL_PATH=$LLM_DIR/models/qwen3.6-35b-a3b
PORT=8100
NODE=$(hostname)

echo "=== vLLM SERVE JOB STARTED ==="
echo "NODE=$NODE"
echo "GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo "PORT=$PORT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# --- Load CUDA ---
module load cuda/12.9.0
echo "CUDA module loaded: $(nvcc --version | grep release)"

# --- Activate venv ---
source $VENV/bin/activate
echo "Python: $(python3 --version)"
echo "venv: $VIRTUAL_ENV"

# --- Phase 1: Fix CUDA if broken ---
echo ""
echo "=== CHECKING CUDA ==="
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
echo "torch.cuda.is_available() = $CUDA_OK"

if [ "$CUDA_OK" != "True" ]; then
    echo "CUDA broken — fixing torch installation..."
    echo "Current torch: $(python3 -c 'import torch; print(torch.__version__)')"

    # Reinstall torch with CUDA 12.4 support (compatible with CUDA 12.9 runtime)
    # PyTorch cu124 wheels work with CUDA 12.4+ runtime (forward compatible)
    /shared/apps/cli-tools/bin/uv pip install --reinstall \
        torch==2.6.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
        2>&1 | tail -20

    # If that fails, try letting vLLM's own CUDA runtime handle it
    CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    echo "After fix: torch.cuda.is_available() = $CUDA_OK"

    if [ "$CUDA_OK" != "True" ]; then
        echo "FATAL: CUDA still not available after fix"
        echo "torch version: $(python3 -c 'import torch; print(torch.__version__)')"
        echo "CUDA_HOME: ${CUDA_HOME:-not set}"
        echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
        python3 -c "import torch; print(f'CUDA compiled: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')" 2>&1 || true
        nvidia-smi
        exit 1
    fi
fi

echo "CUDA OK — GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "VRAM: $(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')")"

# --- Phase 2: Verify vLLM imports ---
echo ""
echo "=== VERIFYING vLLM ==="
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "FATAL: vLLM import failed"
    exit 1
}

# --- Phase 3: Start vLLM server ---
echo ""
echo "=== STARTING vLLM SERVER ==="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor parallel: 1 (single L40S)"
echo ""

# Write state file for monitoring
mkdir -p $LLM_DIR/logs
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B",
    "started_at": "$(date -Iseconds)",
    "status": "starting",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF

echo "ACCESS via SSH tunnel:"
echo "  ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
echo ""
echo "Then use: http://localhost:$PORT/v1/chat/completions"
echo ""

# vLLM serve with optimized settings for MoE on single L40S
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --served-model-name "qwen3.6-35b-a3b" \
    --chat-template-content-format "auto" \
    2>&1 | tee $LLM_DIR/logs/vllm_serve_${SLURM_JOB_ID}.log

# Update state on exit
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B",
    "started_at": "$(date -Iseconds)",
    "status": "stopped"
}
EOF
