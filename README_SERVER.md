# Kokoro TTS FastAPI Server

This adds a minimal HTTP API to synthesize speech via the Kokoro model.

## Endpoints

- GET /voices
  - Returns available voice IDs (e.g., `af_bella`).
- POST /tts
  - JSON body: `{ "voice": "af_bella", "text": "Hello", "speed": 1.0 }`
  - Responds with MP3 (`audio/mpeg`).

## Requirements

- Python packages: see `requirements.txt` (adds `fastapi`, `uvicorn[standard]`).
- FFmpeg installed and on PATH (required by `pydub` for MP3 export).
  - Windows: download from https://ffmpeg.org, add `bin/` to PATH.

## Run the API

```bash
# (Optional) Create/activate a virtualenv
# python -m venv .venv
# .\.venv\Scripts\activate

pip install -r requirements.txt

# Start server
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Example Requests

```bash
# List voices
curl http://localhost:8000/voices

# Synthesize (save to file)
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"voice":"af_bella","text":"Hello from Kokoro","speed":1.0}' \
  --output output.mp3
```

## Notes
- Text length is limited (10k chars; reduced if low RAM).
- Speed must be between 0.1 and 3.0.
- The service loads the model once at startup and serializes inference with a global lock for safety.
