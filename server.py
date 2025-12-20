import os
import io
import time
import threading
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Import Kokoro model helpers
from models import build_model, list_available_voices

# --------------------
# Constants & Defaults
# --------------------
MAX_TEXT_LENGTH = 10000
MAX_GENERATION_TIME = 300  # seconds (overall)
MIN_SEGMENT_TIME = 60      # seconds (per-segment soft guard)
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0

VOICES_DIR = Path("voices").resolve()
DEFAULT_MODEL_PATH = Path("kokoro-v1_0.pth").resolve()

# --------------------
# Utility helpers (copied/adapted from tts_demo)
# --------------------
def validate_sample_rate(rate: int) -> int:
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        # Default to 24kHz which Kokoro uses
        return 24000
    return rate

SAMPLE_RATE = validate_sample_rate(DEFAULT_SAMPLE_RATE)

# --------------------
# FastAPI app setup
# --------------------
app = FastAPI(title="Kokoro TTS API", version="1.0")

# Allow all origins/methods/headers (relaxed CORS)
# TODO : tighten for production use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Log all HTTPExceptions with request context (voice/text)
@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException):
    voice = None
    text = None
    try:
        # Attempt to read JSON body for context (safe if JSON and small)
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            if isinstance(body, dict):
                voice = body.get("voice")
                text = body.get("text")
    except Exception:
        pass

    try:
        _logger.error(
            "HTTPException status=%s detail=%s voice=%s text_len=%s path=%s",
            getattr(exc, "status_code", None),
            getattr(exc, "detail", None),
            voice,
            (len(text) if isinstance(text, str) else None),
            request.url.path,
        )
        # Optionally log the text itself (could be large)
        if isinstance(text, str):
            # Log up to 500 chars to avoid flooding logs
            snippet = text if len(text) <= 500 else text[:500] + "..."
            _logger.error("HTTPException text_snippet=%s", snippet)
    except Exception:
        pass

    # Return JSON response consistent with FastAPI defaults
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Global model and concurrency control
_model = None
_device = "cpu"
_inference_lock = threading.Lock()
_logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    voice: str = Field(..., description="Voice ID (e.g., af_bella)")
    text: str = Field(..., description="Text to synthesize")
    speed: Optional[float] = Field(DEFAULT_SPEED, ge=MIN_SPEED, le=MAX_SPEED, description="Playback speed multiplier")


def _select_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@app.on_event("startup")
def on_startup() -> None:
    global _model, _device
    _device = _select_device()
    # Build/load model once
    _model = build_model(DEFAULT_MODEL_PATH, _device)


@app.get("/voices", response_model=List[str])
def list_voices() -> List[str]:
    voices = list_available_voices()
    return voices


def _enforce_text_limit(text: str) -> None:
    # Optionally reduce limit based on memory, similar to tts_demo
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        dynamic_max = MAX_TEXT_LENGTH if available_gb >= 2.0 else min(MAX_TEXT_LENGTH, 3000)
    except Exception:
        dynamic_max = MAX_TEXT_LENGTH

    if len(text) > dynamic_max:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long ({len(text)} chars). Max allowed: {dynamic_max}"
        )


def _voice_path_for_id(voice_id: str) -> Path:
    return VOICES_DIR / f"{voice_id}.pt"


def _synthesize(text: str, voice_id: str, speed: float) -> np.ndarray:
    if _model is None:
        raise HTTPException(status_code=500, detail="TTS model not initialized")

    # Validate voice
    voice_path = _voice_path_for_id(voice_id)
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")

    # Validate speed
    if speed < MIN_SPEED or speed > MAX_SPEED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Speed must be between {MIN_SPEED} and {MAX_SPEED}"
        )

    # Enforce text limit
    _enforce_text_limit(text)

    # Synthesize via Kokoro generator
    start_time = time.time()
    segment_start = start_time
    all_audio: List[torch.Tensor] = []

    # Use a lock for safety around model execution
    with _inference_lock:
        try:
            generator = _model(text, voice=str(voice_path), speed=speed, split_pattern=r"\n+")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing generator: {type(e).__name__}: {e}")

        try:
            for gs, ps, audio in generator:
                now = time.time()
                if now - start_time > MAX_GENERATION_TIME:
                    # Stop if overall time exceeded
                    break
                if now - segment_start > MIN_SEGMENT_TIME:
                    # Soft stop if a single segment is taking too long
                    break
                segment_start = now

                if audio is not None:
                    tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                    all_audio.append(tensor)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during synthesis: {type(e).__name__}: {e}")

    if not all_audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    if len(all_audio) == 1:
        final_audio = all_audio[0]
    else:
        try:
            final_audio = torch.cat(all_audio, dim=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error concatenating audio: {e}")

    return final_audio.detach().cpu().numpy()


def _encode_mp3_from_float32_mono(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE, bitrate: str = "128k") -> bytes:
    # Ensure float32 mono in [-1, 1]
    t0 = time.time()
    try:
        _logger.debug(
            "mp3_encode:start dtype=%s len=%d sr=%d bitrate=%s min=%.6f max=%.6f",
            getattr(audio_f32, "dtype", None),
            len(audio_f32) if hasattr(audio_f32, "__len__") else -1,
            sample_rate,
            bitrate,
            float(np.min(audio_f32)) if audio_f32 is not None and audio_f32.size else float("nan"),
            float(np.max(audio_f32)) if audio_f32 is not None and audio_f32.size else float("nan"),
        )
    except Exception:
        _logger.debug("mp3_encode:start (stats unavailable)")

    if audio_f32.dtype != np.float32:
        audio_f32 = audio_f32.astype(np.float32)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)

    # Convert to int16 PCM for pydub/ffmpeg
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    try:
        _logger.debug(
            "mp3_encode:after_i16 len=%d min=%d max=%d",
            len(audio_i16) if hasattr(audio_i16, "__len__") else -1,
            int(audio_i16.min()) if audio_i16.size else 0,
            int(audio_i16.max()) if audio_i16.size else 0,
        )
    except Exception:
        pass

    # Build AudioSegment from raw bytes (no temp WAV)
    seg = AudioSegment(
        data=audio_i16.tobytes(),
        sample_width=2,  # int16
        frame_rate=sample_rate,
        channels=1,
    )

    buf = io.BytesIO()
    try:
        seg.export(buf, format="mp3", bitrate=bitrate)
    except Exception as e:
        # Typically means FFmpeg not found or encoder error
        _logger.exception("mp3_encode:export_failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"MP3 encoding failed: {type(e).__name__}: {e}. Ensure FFmpeg is installed and on PATH."
        )
    data = buf.getvalue()
    _logger.debug("mp3_encode:success bytes=%d elapsed=%.3fs", len(data), time.time() - t0)
    return data


@app.post("/tts")
def tts(req: TTSRequest):
    audio = _synthesize(req.text, req.voice, req.speed if req.speed is not None else DEFAULT_SPEED)
    mp3_bytes = _encode_mp3_from_float32_mono(audio, SAMPLE_RATE)

    headers = {
        "Content-Disposition": "inline; filename=tts.mp3"
    }
    return Response(content=mp3_bytes, media_type="audio/mpeg", headers=headers)


# Optional entrypoint for running directly
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
