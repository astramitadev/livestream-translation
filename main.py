import os
import tempfile

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from faster_whisper import WhisperModel
from transformers import MarianTokenizer, MarianMTModel

import yt_dlp

# Try to add ffmpeg to path if installed via ffmpeg-binaries
try:
    import ffmpeg
    ffmpeg.add_to_path()
except Exception:
    pass

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global variables for models
whisper_model = None
tokenizer = None
translation_model = None


@app.on_event("startup")
async def startup_event():
    """Load models at application startup."""
    global whisper_model, tokenizer, translation_model
    # Load a small faster-whisper model on CPU for lower resource usage
    whisper_model = WhisperModel("base", device="cpu")
    # Load MarianMT Spanish-to-English translation model
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page with an optional translated text."""
    return templates.TemplateResponse("index.html", {"request": request, "translated_text": None, "original_text": None})


@app.post("/translate", response_class=HTMLResponse)
async def translate(request: Request, url: str = Form(...)):
    """
    Download the audio from the provided YouTube/Facebook URL, transcribe it
    using a CPU-based Whisper model, translate the Spanish text to English,
    and render the results on the home page.
    """
    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download best available audio and convert to WAV
        output_template = os.path.join(tmpdir, "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # The base filename without extension
            filename = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            audio_path = base + ".wav"

        # Perform transcription with faster-whisper
        segments, info = whisper_model.transcribe(audio_path, beam_size=5, language="es")
        spanish_text = " ".join([segment.text for segment in segments])

    # Translate the Spanish text to English
    tokens = tokenizer(spanish_text, return_tensors="pt", padding=True)
    translated_tokens = translation_model.generate(**tokens)
    english_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "translated_text": english_text,
            "original_text": spanish_text,
        },
    )
