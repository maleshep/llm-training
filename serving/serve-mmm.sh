#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=llm-serve-mmm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/serve_mmm_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/serve_mmm_%j.err

# =============================================================================
# SGLang Inference Server — Qwen3.6-35B-A3B-MMM (Fine-tuned)
# =============================================================================
# This serves the MERGED model (SFT + GRPO adapters baked in).
# Same API as before — Pi CLI connects without config changes.
#
# Model: qwen3.6-35b-a3b-mmm (BF16 merged, ~67GB)
#        OR qwen3.6-35b-a3b-mmm-fp8 (quantized, ~18GB) if available
# Port: 8100 (same as before)
# Served name: qwen3.6-35b-a3b (same as before — transparent upgrade)
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
PORT=8100
NODE=$(hostname)

# Prefer FP8 quantized if available, fall back to BF16 merged
if [ -f "$LLM_DIR/models/qwen3.6-35b-a3b-mmm-fp8/config.json" ]; then
    MODEL=$LLM_DIR/models/qwen3.6-35b-a3b-mmm-fp8
    DTYPE_FLAG="--quantization fp8"
    MODEL_LABEL="MMM-FP8"
elif [ -f "$LLM_DIR/models/qwen3.6-35b-a3b-mmm/config.json" ]; then
    MODEL=$LLM_DIR/models/qwen3.6-35b-a3b-mmm
    DTYPE_FLAG="--dtype bfloat16"
    MODEL_LABEL="MMM-BF16"
else
    echo "ERROR: No merged MMM model found."
    echo "  Expected: $LLM_DIR/models/qwen3.6-35b-a3b-mmm/"
    echo "  Run: sbatch scripts/merge_and_export.sh"
    exit 1
fi

module load cuda/12.9.0
source $LLM_DIR/venv/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1

echo "=== LLM SERVE (MMM Fine-tuned) ==="
echo "NODE=$NODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "PORT=$PORT"
echo "JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
echo "LABEL=$MODEL_LABEL"
echo "ENGINE=SGLang $(python3 -c 'import sglang; print(sglang.__version__)')"
echo "=================================="
echo ""
echo "ACCESS:"
echo "  ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
echo "  omp --model hpc-qwen/qwen3.6-35b-a3b"
echo ""

# Write state
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B-MMM",
    "model_path": "$MODEL",
    "model_label": "$MODEL_LABEL",
    "engine": "sglang",
    "tp_size": 1,
    "fine_tuned": true,
    "adapters": ["SFT", "GRPO"],
    "started_at": "$(date -Iseconds)",
    "status": "loading",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF

# Start SGLang
# Served model name stays "qwen3.6-35b-a3b" so Pi CLI doesn't need reconfiguring
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 --port $PORT \
    $DTYPE_FLAG \
    --mem-fraction-static 0.90 \
    --context-length 65536 \
    --trust-remote-code \
    --served-model-name qwen3.6-35b-a3b \
    --max-running-requests 8 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for ready
echo "Loading model..."
READY=false
for i in $(seq 1 90); do
    sleep 5
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q 200; then
        echo ""
        echo "SERVER READY after $((i*5)) seconds!"
        READY=true
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "SERVER CRASHED"
        cat > $LLM_DIR/.serve-state.json << EOFX
{"job_id":"$SLURM_JOB_ID","status":"crashed","node":"$NODE"}
EOFX
        exit 1
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  ...loading ($((i*5))s)"
    fi
done

if [ "$READY" != "true" ]; then
    echo "Server failed to start within 7.5 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Update state
cat > $LLM_DIR/.serve-state.json << EOF2
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B-MMM",
    "model_path": "$MODEL",
    "model_label": "$MODEL_LABEL",
    "engine": "sglang",
    "tp_size": 1,
    "fine_tuned": true,
    "adapters": ["SFT", "GRPO"],
    "started_at": "$(date -Iseconds)",
    "status": "serving",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF2

echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Quick validation
echo "=== VALIDATION ==="
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [
            {"role": "system", "content": "You are an expert pharma marketing mix model optimization agent."},
            {"role": "user", "content": "The model has F2F plausibility at 70% and email attribution at 18%. Trust score is 33. What should we change?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }')
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
u = d.get('usage', {})
print(f'Tokens: {u.get(\"prompt_tokens\",0)}+{u.get(\"completion_tokens\",0)}')
print(f'Response: {msg[:300]}')
"

echo ""
echo "=========================================="
echo "=== SERVING MMM AGENT on port $PORT ==="
echo "=== Node: $NODE ==="
echo "=== Model: $MODEL_LABEL ==="
echo "=== Tunnel: ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com ==="
echo "=========================================="
echo ""

# Keep alive
wait $SERVER_PID
