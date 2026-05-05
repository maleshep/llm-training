#!/bin/bash
#SBATCH --account=tdr-mmm-hpc
#SBATCH --job-name=wisprflow
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/tdr-mmm-hpc/logs/wisprflow_%j.out
#SBATCH --error=/shared/project/tdr-mmm-hpc/logs/wisprflow_%j.err

# =============================================================================
# WisprFlow Unified Serve Job
# Launches ALL services on a single L40S (48 GB):
#   - SGLang (Qwen3.6-35B-A3B-FP8)  → port 8100 (~18 GB)
#   - ASR (Qwen2-Audio-7B)           → port 8200 (~14 GB)
#   - TTS (CosyVoice3-0.5B)          → port 8300 (~2 GB)
#   Total: ~34 GB / 48 GB available
# =============================================================================

set -e

PROJECT=/shared/project/tdr-mmm-hpc
WISPRFLOW=$PROJECT/model_training/wisprflow
NODE=$(hostname)
PIDS=()

echo "============================================"
echo "WisprFlow Unified Serve"
echo "Node: $NODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

# --- Load environment ---
module purge
module load onehpc/2509
module load cli-tools

# Activate Python environment
source $PROJECT/venv/bin/activate 2>/dev/null || {
    echo "Creating venv..."
    cd $PROJECT
    uv venv venv --python 3.11
    source venv/bin/activate
    uv pip install torch torchaudio transformers accelerate librosa
    uv pip install fastapi uvicorn python-multipart
    uv pip install sglang[all]
}

# --- 1. SGLang LLM (port 8100) ---
echo ""
echo ">>> Starting SGLang (Qwen3.6-35B-A3B-FP8) on port 8100..."
python -m sglang.launch_server \
    --model-path $PROJECT/models/Qwen3.6-35B-A3B-FP8 \
    --port 8100 \
    --host 0.0.0.0 \
    --tp 1 \
    --quantization fp8 \
    --context-length 32768 \
    --mem-fraction-static 0.35 \
    --disable-cuda-graph &
PIDS+=($!)
echo "SGLang PID: ${PIDS[-1]}"

# Wait for SGLang to be ready
echo "Waiting for SGLang to load model..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 1
done

# --- 2. ASR Server (port 8200) ---
echo ""
echo ">>> Starting ASR server (Qwen2-Audio-7B) on port 8200..."
cd $WISPRFLOW
python -m uvicorn asr_server:app --host 0.0.0.0 --port 8200 &
PIDS+=($!)
echo "ASR PID: ${PIDS[-1]}"

# Wait for ASR to load
echo "Waiting for ASR model to load..."
for i in $(seq 1 180); do
    if curl -s http://localhost:8200/health > /dev/null 2>&1; then
        echo "ASR ready after ${i}s"
        break
    fi
    sleep 1
done

# --- 3. TTS Server (port 8300) ---
echo ""
echo ">>> Starting TTS server (CosyVoice3) on port 8300..."
python -m uvicorn tts_server:app --host 0.0.0.0 --port 8300 &
PIDS+=($!)
echo "TTS PID: ${PIDS[-1]}"

# Wait for TTS to load
echo "Waiting for TTS model to load..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8300/health > /dev/null 2>&1; then
        echo "TTS ready after ${i}s"
        break
    fi
    sleep 1
done

# --- All services ready ---
echo ""
echo "============================================"
echo "ALL SERVICES RUNNING"
echo "============================================"
echo "LLM:  http://${NODE}:8100  (Qwen3.6-35B-A3B-FP8)"
echo "ASR:  http://${NODE}:8200  (Qwen2-Audio-7B)"
echo "TTS:  http://${NODE}:8300  (CosyVoice3-0.5B)"
echo ""
echo "TUNNEL COMMAND (run on Windows):"
echo "  ssh -L 8100:${NODE}:8100 -L 8200:${NODE}:8200 -L 8300:${NODE}:8300 -N M316235@onehpc.merckgroup.com"
echo ""
echo "VRAM usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "============================================"

# Write state file for tunnel reconnection
cat > $PROJECT/.wisprflow-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "services": {
        "llm": {"port": 8100, "model": "Qwen3.6-35B-A3B-FP8"},
        "asr": {"port": 8200, "model": "Qwen2-Audio-7B-Instruct"},
        "tts": {"port": 8300, "model": "CosyVoice3-0.5B"}
    },
    "tunnel_cmd": "ssh -L 8100:${NODE}:8100 -L 8200:${NODE}:8200 -L 8300:${NODE}:8300 -N M316235@onehpc.merckgroup.com",
    "started_at": "$(date -Iseconds)"
}
EOF

# --- Keep alive ---
# Trap SIGTERM for graceful shutdown
cleanup() {
    echo "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "All services stopped."
}
trap cleanup SIGTERM SIGINT

# Wait forever (Slurm will kill when time limit reached)
wait
