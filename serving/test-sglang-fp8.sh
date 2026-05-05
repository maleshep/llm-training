#!/bin/bash
# Test SGLang with FP8 quantization on single L40S
set -euo pipefail

module load cuda/12.9.0
source /shared/project/tdr-mmm-hpc/llm/venv/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/shared/project/tdr-mmm-hpc/llm/models/qwen3.6-35b-a3b
PORT=8100

echo "=== SGLang FP8 Inference Test ==="
echo "Model: $MODEL"
echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "sglang: $(python3 -c 'import sglang; print(sglang.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Start server with FP8 quantization
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 --port $PORT \
    --dtype float16 \
    --quantization fp8 \
    --mem-fraction-static 0.85 \
    --context-length 8192 \
    --trust-remote-code \
    --served-model-name qwen3.6-35b-a3b \
    --max-running-requests 4 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for ready
echo "Waiting for model load + FP8 quantize..."
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

# GPU memory after load
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Test 1: Short
echo "=== TEST 1: Short generation ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"What is marketing mix modelling? One sentence."}],"max_tokens":60,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
python3 -c "
import json, sys
d = json.loads('''$RESP''')
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
print(f'Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s')
print(f'Response: {msg[:400]}')
"
echo ""

# Test 2: Medium (256 tokens)
echo "=== TEST 2: Medium generation (256 tok) ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Explain adstock decay in marketing mix models. Include the mathematical formula and practical implications."}],"max_tokens":256,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
python3 -c "
import json, sys
d = json.loads('''$RESP''')
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
print(f'Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s')
"
echo ""

# Test 3: Long (512 tokens)
echo "=== TEST 3: Long generation (512 tok) ==="
START=$(date +%s%N)
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Write a detailed analysis of how to set priors for face-to-face channel effects in a pharmaceutical MMM. Consider: specialty drugs vs primary care, sales rep visit frequency, detailing effectiveness literature, and how to bound attribution to avoid implausible values."}],"max_tokens":512,"temperature":0.7}')
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
python3 -c "
import json, sys
d = json.loads('''$RESP''')
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tps = ct / ($MS / 1000) if $MS > 0 else 0
msg = d.get('choices', [{}])[0].get('message', {}).get('content', '?')
print(f'Latency: ${MS}ms | Prompt: {pt} | Completion: {ct} | Speed: {tps:.1f} tok/s')
print(f'Preview: {msg[:200]}...')
"
echo ""

echo "=== SUMMARY ==="
echo "Model: Qwen3.6-35B-A3B (FP8 quantized on-the-fly)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "Node: $(hostname)"
echo "Tunnel: ssh -L $PORT:$(hostname):$PORT -N M316235@onehpc.merckgroup.com"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
