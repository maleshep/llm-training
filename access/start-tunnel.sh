#!/bin/bash
# SSH tunnel to HPC LLM inference server
# Run this BEFORE launching Pi (omp)
#
# Usage: bash start-tunnel.sh
#        Then in another terminal: omp
#
# The tunnel forwards localhost:8100 → HPC GPU node running SGLang
# Job 1782730 on demu4xgpu002 (may change after restart)

NODE=${1:-demu4xgpu002}
PORT=8100

echo "=== HPC LLM Tunnel ==="
echo "Forwarding localhost:$PORT → $NODE:$PORT"
echo "Press Ctrl+C to stop"
echo ""

ssh -L $PORT:${NODE}:$PORT -N M316235@onehpc.merckgroup.com
