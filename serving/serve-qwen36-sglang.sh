#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=llm-serve-qwen36
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:2
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/serve_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/serve_%j.err

# =============================================================================
# SGLang Inference Server — Qwen3.6-35B-A3B on 2× L40S
# =============================================================================
# Engine: SGLang 0.5.9
# Port: 8100 (OpenAI-compatible API)
# Model: Qwen/Qwen3.6-35B-A3B (MoE: 35B total, 3B active per token)
# VRAM: ~20GB per GPU (tensor parallel = 2)
# Context: 32K tokens
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/qwen3.6-35b-a3b
PORT=8100
NODE=$(hostname)

module load cuda/12.9.0
source $LLM_DIR/venv/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1

echo "=== LLM SERVE JOB STARTED ==="
echo "NODE=$NODE"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "PORT=$PORT"
echo "JOB=$SLURM_JOB_ID"
echo "ENGINE=SGLang $(python3 -c 'import sglang; print(sglang.__version__)')"
echo "TORCH=$(python3 -c 'import torch; print(torch.__version__)')"
echo "=============================="
echo ""
echo "ACCESS (SSH tunnel):"
echo "  ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
echo ""

# Write state file
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B",
    "engine": "sglang",
    "tp_size": 2,
    "started_at": "$(date -Iseconds)",
    "status": "loading",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF

# Start SGLang server
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 --port $PORT \
    --dtype bfloat16 \
    --tp-size 2 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
    --trust-remote-code \
    --served-model-name qwen3.6-35b-a3b \
    --max-running-requests 8 &
SERVER_PID=$!

# Wait for ready (model load from NFS can take 5-10 min)
echo "Waiting for model load..."
READY=false
for i in $(seq 1 120); do
    sleep 5
    if curl -s http://localhost:$PORT/health 2>/dev/null | grep -q healthy; then
        echo ""
        echo "SERVER READY after $((i*5)) seconds!"
        READY=true
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "SERVER CRASHED"
        cat > $LLM_DIR/.serve-state.json << EOF2
{"job_id":"$SLURM_JOB_ID","node":"$NODE","status":"crashed"}
EOF2
        exit 1
    fi
    if [ $((i % 12)) -eq 0 ]; then
        echo "  ...still loading ($((i*5))s)"
    fi
done

if [ "$READY" != "true" ]; then
    echo "Server failed to start within 10 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Update state
cat > $LLM_DIR/.serve-state.json << EOF3
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B",
    "engine": "sglang",
    "tp_size": 2,
    "started_at": "$(date -Iseconds)",
    "status": "serving",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF3

echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""
echo "=== Server is live ==="
echo "OpenAI endpoint: http://localhost:$PORT/v1/chat/completions"
echo "Health: http://localhost:$PORT/health"
echo ""

# Quick validation
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}')
echo "Validation: $RESP" | head -1
echo ""
echo "=== SERVING — waiting for scancel or wall time ==="

# Keep alive
wait $SERVER_PID
