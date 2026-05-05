#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=llm-serve-fp8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/llm/logs/serve_fp8_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/llm/logs/serve_fp8_%j.err

# =============================================================================
# SGLang Inference Server — Qwen3.6-35B-A3B-FP8 on 1× L40S
# =============================================================================
# Engine: SGLang 0.5.9
# Port: 8100 (OpenAI-compatible API)
# Model: Qwen/Qwen3.6-35B-A3B-FP8 (pre-quantized, ~18GB VRAM)
# Available KV cache: ~26GB (enough for 32K context)
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/tdr-mmm-hpc
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/qwen3.6-35b-a3b-fp8
PORT=8100
NODE=$(hostname)

module load cuda/12.9.0
source $LLM_DIR/venv/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1

echo "=== LLM SERVE (FP8) STARTED ==="
echo "NODE=$NODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "PORT=$PORT"
echo "JOB=$SLURM_JOB_ID"
echo "ENGINE=SGLang $(python3 -c 'import sglang; print(sglang.__version__)')"
echo "MODEL=$MODEL"
echo "=============================="
echo ""
echo "ACCESS:"
echo "  ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
echo "  Then: http://localhost:$PORT/v1/chat/completions"
echo ""

# Verify model exists
if [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Run download-fp8.sh first"
    exit 1
fi

# Write state
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B-FP8",
    "engine": "sglang",
    "tp_size": 1,
    "dtype": "fp8",
    "started_at": "$(date -Iseconds)",
    "status": "loading",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF

# Start SGLang — FP8 model, single GPU, large context
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 --port $PORT \
    --dtype auto \
    --quantization fp8 \
    --mem-fraction-static 0.90 \
    --context-length 65536 \
    --trust-remote-code \
    --served-model-name qwen3.6-35b-a3b \
    --max-running-requests 8 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for ready (FP8 model loads much faster — ~1-2 min)
echo "Loading model..."
READY=false
for i in $(seq 1 60); do
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
    echo "Server failed to start within 5 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Update state
cat > $LLM_DIR/.serve-state.json << EOF2
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "Qwen3.6-35B-A3B-FP8",
    "engine": "sglang",
    "tp_size": 1,
    "dtype": "fp8",
    "started_at": "$(date -Iseconds)",
    "status": "serving",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com"
}
EOF2

echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Quick validation + benchmark
echo "=== VALIDATION ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"What is marketing mix modelling? One sentence."}],"max_tokens":60,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "$RESP" > /tmp/validation.json
python3 << PYEOF
import json
with open('/tmp/validation.json') as f:
    d = json.load(f)
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
print(f"Latency: ${MS}ms | Tokens: {pt}+{ct} | Speed: {tps:.1f} tok/s")
print(f"Response: {msg[:400]}")
PYEOF

echo ""
echo "=========================================="
echo "=== SERVING on port $PORT ==="
echo "=== Node: $NODE ==="
echo "=== Tunnel: ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com ==="
echo "=========================================="
echo ""

# Keep alive until wall-time or scancel
wait $SERVER_PID
