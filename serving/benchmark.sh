#!/bin/bash
# =============================================================================
# Benchmark vLLM inference — run AFTER serve job is up
# Usage: ssh to HPC, then: bash /shared/project/tdr-mmm-hpc/llm/scripts/benchmark.sh
# =============================================================================

set -euo pipefail

PORT=8100
BASE_URL="http://localhost:$PORT"
MODEL="qwen3.6-35b-a3b"
RESULTS_DIR="/shared/project/tdr-mmm-hpc/llm/benchmarks"
mkdir -p $RESULTS_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$RESULTS_DIR/bench_${TIMESTAMP}.json"

echo "=== vLLM BENCHMARK ==="
echo "Server: $BASE_URL"
echo "Model: $MODEL"
echo ""

# Check server is up
echo "--- Health check ---"
curl -s "$BASE_URL/health" || { echo "ERROR: Server not responding on port $PORT"; exit 1; }
echo " OK"
echo ""

# --- Test 1: Time to first token (short prompt) ---
echo "--- Test 1: TTFT (short prompt, streaming) ---"
START=$(date +%s%N)
FIRST_CHUNK=$(curl -s -w "\n%{time_starttransfer}" "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is marketing mix modelling?\"}],
        \"max_tokens\": 1,
        \"stream\": false
    }" 2>/dev/null)
END=$(date +%s%N)
TTFT_MS=$(( (END - START) / 1000000 ))
echo "TTFT: ${TTFT_MS}ms"
echo ""

# --- Test 2: Throughput (medium generation) ---
echo "--- Test 2: Throughput (256 tokens) ---"
START=$(date +%s%N)
RESPONSE=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Explain the concept of adstock decay in marketing mix modelling. Include mathematical formulation and practical implications for budget optimization.\"}],
        \"max_tokens\": 256,
        \"temperature\": 0.7
    }")
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
if [ "$TOKENS" -gt 0 ]; then
    TPS=$(python3 -c "print(f'{$TOKENS / ($ELAPSED_MS / 1000):.1f}')")
    echo "Generated: $TOKENS tokens in ${ELAPSED_MS}ms = $TPS tok/s"
else
    echo "ERROR: No tokens generated"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
fi
echo ""

# --- Test 3: Long generation (1024 tokens) ---
echo "--- Test 3: Throughput (1024 tokens) ---"
START=$(date +%s%N)
RESPONSE=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed technical analysis of how Bayesian Marketing Mix Models handle channel attribution in pharmaceutical marketing. Cover: 1) Prior specification for channel effects, 2) Adstock transformations, 3) Saturation curves, 4) Geographic hierarchical effects, 5) Model diagnostics and validation.\"}],
        \"max_tokens\": 1024,
        \"temperature\": 0.7
    }")
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
PROMPT_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null || echo "0")
if [ "$TOKENS" -gt 0 ]; then
    TPS=$(python3 -c "print(f'{$TOKENS / ($ELAPSED_MS / 1000):.1f}')")
    echo "Generated: $TOKENS tokens in ${ELAPSED_MS}ms = $TPS tok/s"
    echo "Prompt tokens: $PROMPT_TOKENS"
else
    echo "ERROR: No tokens generated"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
fi
echo ""

# --- Test 4: Reasoning mode (think tokens) ---
echo "--- Test 4: Reasoning mode ---"
START=$(date +%s%N)
RESPONSE=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Think step by step: If a marketing mix model shows email channel attribution of 25% but we know email opens correlate with prescribing intent (endogeneity), what prior constraints should we apply?\"}],
        \"max_tokens\": 512,
        \"temperature\": 0.7
    }")
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
if [ "$TOKENS" -gt 0 ]; then
    TPS=$(python3 -c "print(f'{$TOKENS / ($ELAPSED_MS / 1000):.1f}')")
    echo "Generated: $TOKENS tokens in ${ELAPSED_MS}ms = $TPS tok/s"
    CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])" 2>/dev/null)
    echo "Preview: $CONTENT..."
else
    echo "ERROR: No tokens generated"
fi
echo ""

# --- Test 5: Concurrent requests (2 parallel) ---
echo "--- Test 5: Concurrency (2 parallel requests) ---"
START=$(date +%s%N)
curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain saturation curves\"}],\"max_tokens\":128}" > /tmp/bench_r1.json &
PID1=$!
curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain adstock decay\"}],\"max_tokens\":128}" > /tmp/bench_r2.json &
PID2=$!
wait $PID1 $PID2
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
T1=$(cat /tmp/bench_r1.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
T2=$(cat /tmp/bench_r2.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
TOTAL=$((T1 + T2))
if [ "$TOTAL" -gt 0 ]; then
    TPS=$(python3 -c "print(f'{$TOTAL / ($ELAPSED_MS / 1000):.1f}')")
    echo "2 requests: $TOTAL total tokens in ${ELAPSED_MS}ms = $TPS tok/s aggregate"
fi
echo ""

# --- Summary ---
echo "=== BENCHMARK COMPLETE ==="
echo "Results saved to: $OUTFILE"

# Save structured results
python3 << 'PYEOF'
import json, os
results = {
    "timestamp": os.popen("date -Iseconds").read().strip(),
    "model": "Qwen3.6-35B-A3B",
    "config": "bf16, 1xL40S, max_model_len=32768, gpu_util=0.92",
    "note": "Check log above for actual numbers"
}
with open(os.environ.get("OUTFILE", "/tmp/bench.json"), "w") as f:
    json.dump(results, f, indent=2)
PYEOF

echo ""
echo "Next steps:"
echo "  1. If tok/s < 30: try FP8 quantized model"
echo "  2. If tok/s < 50: try speculative decoding with small draft model"
echo "  3. If tok/s > 60: baseline is good, optimize TTFT"
