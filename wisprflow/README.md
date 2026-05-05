# WisprFlow — Self-Hosted Speech Pipeline on HPC

Press a hotkey, speak, get text at your cursor. All inference runs on HPC — no cloud APIs.

**Status: LIVE on demu4xgpu006 (Job 1811142, as of 2026-05-01)**

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Windows (local)                                                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────┐                   │
│  │  WisprFlow Client (Python, system tray)          │                   │
│  │                                                  │                   │
│  │  1. Ctrl+Shift+Space → start recording           │                   │
│  │  2. Release → stop, send audio via HTTP          │                   │
│  │  3. Animated overlay while processing            │                   │
│  │  4. Text inserted at cursor position             │                   │
│  │  5. (Optional) TTS plays back response           │                   │
│  └──────────────────────────────────────────────────┘                   │
│           │ POST audio/wav                    ▲ JSON {text: "..."}       │
│           ▼ localhost:8200                    │                          │
└───────────┼──────────────────────────────────┼──────────────────────────┘
            │          SSH TUNNEL               │
            ▼                                  │
┌───────────┼──────────────────────────────────┼──────────────────────────┐
│ HPC (L40S compute node — demu4xgpu006)       │                          │
│           │                                  │                          │
│  ┌────────▼──────────────────────────────────┐                          │
│  │  ASR Server (FastAPI, port 8200)          │                          │
│  │  Model: Qwen3-ASR-1.7B                   │                          │
│  │  VRAM: 4.1 GB                            │                          │
│  │  Latency: 189ms (warm) / 12s (cold)      │                          │
│  │  Languages: 52 (30 languages + 22        │                          │
│  │  Chinese dialects)                        │                          │
│  └───────────────────────────────────────────┘                          │
│                                                                         │
│  ┌───────────────────────────────────────────┐                          │
│  │  TTS Server (FastAPI, port 8300)          │                          │
│  │  Model: Qwen3-TTS-12Hz-1.7B-CustomVoice  │                          │
│  │  VRAM: 4.2 GB                            │                          │
│  │  Speakers: Ryan, Aiden, Vivian, Serena,   │                          │
│  │  Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee   │                          │
│  │  Languages: 10 (EN, ZH, JP, KR, DE, FR,  │                          │
│  │  RU, PT, ES, IT)                          │                          │
│  │  Features: voice clone, voice design,     │                          │
│  │  streaming, emotion control               │                          │
│  └───────────────────────────────────────────┘                          │
│                                                                         │
│  Total VRAM: 9.5 GB / 46 GB (80% headroom)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance (measured 2026-05-01)

| Metric | Value |
|--------|-------|
| ASR cold start | ~22s (model load) |
| ASR warm latency | **189ms** |
| TTS generation | ~7.5s for 12-word sentence |
| ASR VRAM | 4.1 GB |
| TTS VRAM | 4.2 GB |
| Total VRAM | 9.5 GB / 46 GB |

## Quick Start

### 1. Start tunnel

```bash
ssh -L 8200:demu4xgpu006:8200 -L 8300:demu4xgpu006:8300 -N M316235@onehpc.merckgroup.com
```

### 2. Test ASR

```bash
curl -X POST http://localhost:8200/transcribe \
  -F "audio=@recording.wav" -F "language=English"
# → {"text": "...", "processing_ms": 189.1}
```

### 3. Test TTS

```bash
curl -X POST http://localhost:8300/synthesize \
  -F "text=Hello world" -F "speaker=Ryan" -F "language=English" \
  -o output.wav
```

### 4. Run WisprFlow client

```bash
pip install pyaudio keyboard requests pyperclip pywin32
python wisprflow/client/wisprflow.pyw
# Press Ctrl+Shift+Space to record, release to transcribe
```

## Endpoints

### POST /transcribe (ASR, port 8200)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| audio | file | required | WAV/MP3/FLAC audio file |
| language | string | auto | Language hint (English, Chinese, etc.) or null for auto-detect |

Response: `{"text": "...", "language": "English", "processing_ms": 189.1}`

### POST /synthesize (TTS, port 8300)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| text | string | required | Text to speak |
| speaker | string | "Ryan" | Voice: ryan, aiden, vivian, serena, uncle_fu, dylan, eric, ono_anna, sohee |
| language | string | "English" | Target language |
| instruct | string | "" | Emotion/style instruction (e.g., "Very happy", "Speak slowly") |

Response: `audio/wav` binary stream

### POST /clone (Voice Clone, port 8300)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| reference | file | required | 3-10s WAV sample of target voice |
| ref_text | string | "" | Transcript of reference audio (improves quality) |
| text | string | required | Text to synthesize in cloned voice |
| language | string | "English" | Target language |

Response: `audio/wav` binary stream

### GET /health (both ports)

Returns model status, VRAM usage, and available speakers (TTS).

## HPC Deployment

### Paths (on cluster)

| What | Path |
|------|------|
| ASR server code | `~/model_training/wisprflow/asr_server.py` |
| TTS server code | `~/model_training/wisprflow/tts_server.py` |
| Serve script | `~/model_training/wisprflow/serve-unified.sh` |
| ASR model | `/shared/project/tdr-mmm-hpc/llm/models/qwen3-asr-1.7b/` |
| TTS model | `/shared/project/tdr-mmm-hpc/llm/models/qwen3-tts-1.7b/` |
| ASR venv | `~/venv-asr/` (qwen-asr + torch 2.6.0+cu124) |
| TTS venv | `~/venv-tts/` (qwen-tts + torch 2.6.0+cu124) |
| Job logs | `~/logs/wisprflow_*.out` |
| State file | `~/.wisprflow-state.json` |

### Resubmit after maintenance

```bash
ssh M316235@onehpc.merckgroup.com "sbatch --qos=1d --time=1-00:00:00 ~/model_training/wisprflow/serve-unified.sh"
```

### Why separate venvs?

`qwen-asr` requires `transformers==4.57.6` and `qwen-tts` requires `transformers==4.57.3`.
They cannot coexist in the same environment.
