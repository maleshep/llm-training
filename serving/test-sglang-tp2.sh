#!/bin/bash
# SGLang inference with tensor parallelism = 2 (2× L40S)
set -euo pipefail

module load cuda/12.9.0
source /shared/project/tdr-mmm-hpc/llm/venv/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1

MODEL=/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b
PORT=8100

echo "=== SGLang TP=2 Inference Test ==="
echo "Model: $MODEL"
echo "GPUs: 2× L40S (tensor parallel)"
echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "sglang: $(python3 -c 'import sglang; print(sglang.__version__)')"
echo ""

# Start server with TP=2
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
echo "Server PID: $SERVER_PID"

# Wait for ready
echo "Waiting for model load (2 GPUs)..."
READY=false
for i in $(seq 1 60); do
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
        exit 1
    fi
    printf "."
done
echo ""

if [ "$READY" != "true" ]; then
    echo "Server failed to start within 5 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# GPU memory
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# Test 1: Short
echo "=== TEST 1: Short ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"What is marketing mix modelling? One sentence."}],"max_tokens":60,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "$RESP" > /tmp/resp1.json
python3 << PYEOF
import json
with open('/tmp/resp1.json') as f:
    d = json.load(f)
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
print(f"Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s")
print(f"Response: {msg[:400]}")
PYEOF
echo ""

# Test 2: Medium (256 tokens)
echo "=== TEST 2: Medium (256 tok) ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Explain adstock decay in marketing mix models. Include the mathematical formula and practical implications for budget optimization."}],"max_tokens":256,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "$RESP" > /tmp/resp2.json
python3 << PYEOF2
import json
with open('/tmp/resp2.json') as f:
    d = json.load(f)
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
print(f"Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s")
print(f"Preview: {msg[:200]}...")
PYEOF2
echo ""

# Test 3: Long (512 tokens)
echo "=== TEST 3: Long (512 tok) ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Write a detailed technical analysis of channel attribution in pharmaceutical marketing mix models. Cover: prior specification, adstock transformations, saturation curves, geographic hierarchical effects, and model validation."}],"max_tokens":512,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "$RESP" > /tmp/resp3.json
python3 << PYEOF3
import json
with open('/tmp/resp3.json') as f:
    d = json.load(f)
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
print(f"Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s")
PYEOF3
echo ""

echo "=== SUMMARY ==="
echo "Engine: SGLang 0.5.9"
echo "Model: Qwen3.6-35B-A3B (bf16, TP=2)"
echo "GPUs: 2× $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Node: $(hostname)"
echo "Port: $PORT"
echo "Tunnel: ssh -L $PORT:$(hostname):$PORT -N M316235@onehpc.merckgroup.com"
echo ""
echo "=== Context for long-running serve job ==="
echo "To run as persistent serve job, use --qos=3d --time=3-00:00:00"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
