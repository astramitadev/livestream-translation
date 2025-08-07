"""
FastAPI-based real-time translation service for livestreams.

This application is designed as an MVP for streaming Spanish sermons and
generating live English subtitles.  It uses the following components:

* FastAPI for the web server and WebSocket support.
* yt-dlp to extract audio from YouTube/Facebook livestream URLs.
* Whisper (or faster-whisper) for speech-to-text transcription.
* A translation model (e.g. MarianMT) to convert Spanish text to English.

The environment used for this assistant may not have the required
dependencies, but this code provides a solid starting point.  To run
locally, install the required packages listed in the README and adjust
the configuration as needed.
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

# Conditional imports – these will fail if the dependencies are not installed.
try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None  # type: ignore

try:
    import whisper  # type: ignore
except ImportError:
    whisper = None  # type: ignore

try:
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore
except ImportError:
    MarianMTModel = None  # type: ignore
    MarianTokenizer = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="Livestream Translation Service")

# Mount static files (for JS/CSS) and set up templates
base_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))

# State for active WebSocket connections
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

# Load models lazily

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
    logging.info("Loading MarianMT es→en model")
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Use safe_serialization to bypass torch.load vulnerability check
    model = MarianMTModel.from_pretrained(
        model_name,
        safe_serialization=True
    )
    return model, tokenizer

asr_model = None  # loaded on first use
translation_model = None  # loaded on first use

async def extract_audio_chunks(url: str, chunk_seconds: int = 6):
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed; cannot extract audio")

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_chunks_"))
    logging.info("Storing temporary audio chunks in %s", tmp_dir)

    cmd = ['yt-dlp', '-f', 'bestaudio', '-o', '-', url]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)

    segment_cmd = [
        'ffmpeg', '-loglevel', 'error', '-i', 'pipe:0',
        '-f', 'segment', '-segment_time', str(chunk_seconds), '-c', 'copy',
        str(tmp_dir / 'chunk-%03d.m4a')
    ]
    ffmpeg_proc = await asyncio.create_subprocess_exec(
        *segment_cmd,
        stdin=proc.stdout,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    chunk_index = 0
    try:
        while True:
            chunk_path = tmp_dir / f'chunk-{chunk_index:03d}.m4a'
            for _ in range(10):
                if chunk_path.exists():
                    break
                await asyncio.sleep(0.1)
            if not chunk_path.exists():
                if ffmpeg_proc.returncode is not None:
                    break
                continue
            yield chunk_path
            chunk_index += 1
    finally:
        logging.info("Cleaning up extraction processes")
        if proc.returncode is None:
            proc.kill()
        if ffmpeg_proc.returncode is None:
            ffmpeg_proc.kill()


def transcribe_file(file_path: Path) -> str:
    global asr_model
    if asr_model is None:
        asr_model = load_asr_model()
        if asr_model is None:
            return "[transcription unavailable: whisper not installed]"
    logging.info("Transcribing %s", file_path)
    result = asr_model.transcribe(str(file_path), language='es')
    return result.get('text', '')


def translate_spanish_to_english(text: str) -> str:
    global translation_model
    if translation_model is None:
        m = load_translation_model()
        if m is None:
            return "[translation unavailable: transformers not installed]"
        translation_model = m
    model, tokenizer = translation_model
    logging.info("Translating %d characters", len(text))
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

async def process_stream(url: str):
    logging.info("Starting stream processing for %s", url)
    subtitle_index = 0
    async for chunk_path in extract_audio_chunks(url):
        spanish = transcribe_file(chunk_path)
        english = translate_spanish_to_english(spanish)
        payload = {
            'index': subtitle_index,
            'spanish': spanish,
            'english': english,
        }
        subtitle_index += 1
        await manager.send_json(payload)
        try:
            os.remove(chunk_path)
        except Exception:
            pass

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/pricing', response_class=HTMLResponse)
async def pricing(request: Request):
    return templates.TemplateResponse('pricing.html', {'request': request})

@app.get('/admin', response_class=HTMLResponse)
async def admin(request: Request):
    data = {'active_connections': len(manager.active_connections)}
    return templates.TemplateResponse('admin.html', {'request': request, 'data': data})

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        data = await websocket.receive_text()
        message = json.loads(data)
        url = message.get('url')
        if not url:
            await websocket.send_text(json.dumps({'error': 'Missing URL'}))
            return
        asyncio.create_task(process_stream(url))
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.exception("WebSocket error: %s", e)
        manager.disconnect(websocket)
