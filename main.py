"""
FastAPI-based real-time translation service for livestreams.

This application is designed as an MVP for streaming Spanish sermons and
generating live English subtitles.  It uses the following components:

* FastAPI for the web server and WebSocket support.
* yt-dlp to extract audio from YouTube/Facebook livestream URLs.
* Whisper (or faster-whisper) for speech-to-text transcription.
* A translation model (e.g. MarianMT) to convert Spanish text to English.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Conditional imports
try:
    import yt_dlp
except ImportError:
    yt_dlp = None  # type: ignore

try:
    import whisper
except ImportError:
    whisper = None  # type: ignore

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    MarianMTModel = None  # type: ignore
    MarianTokenizer = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="Livestream Translation Service")

# Static files & templates
base_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info("WebSocket connected (%s active)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logging.info("WebSocket disconnected (%s active)", len(self.active_connections))

    async def send_json(self, data: dict):
        to_remove = []
        for ws in self.active_connections:
            try:
                await ws.send_text(json.dumps(data))
            except WebSocketDisconnect:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)

manager = ConnectionManager()

# Lazy-loaded models
asr_model = None
translation_model = None

def load_asr_model() -> Optional[object]:
    if whisper is None:
        logging.warning("Whisper library is not installed")
        return None
    logging.info("Loading Whisper ASR model (base)")
    return whisper.load_model("base")

def load_translation_model() -> Optional[tuple]:
    if MarianMTModel is None or MarianTokenizer is None:
        logging.warning("Transformers library is not installed")
        return None
    model_name = "Helsinki-NLP/opus-mt-es-en"
    logging.info("Loading MarianMT tokenizer and model with safe_serialization")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(
        model_name,
        safe_serialization=True
    )
    return model, tokenizer

async def extract_audio_chunks(url: str, chunk_seconds: int = 6):
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed; cannot extract audio")
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_chunks_"))
    logging.info("Storing temporary audio chunks in %s", tmp_dir)

    proc = await asyncio.create_subprocess_exec(
        "yt-dlp", "-f", "bestaudio", "-o", "-", url,
        stdout=asyncio.subprocess.PIPE
    )
    segment_cmd = [
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", str(tmp_dir / "chunk-%03d.m4a")
    ]
    ffmpeg_proc = await asyncio.create_subprocess_exec(
        *segment_cmd,
        stdin=proc.stdout,
        stdout=asyncio
