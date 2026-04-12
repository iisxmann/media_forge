"""Audio processing API endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse

from mediaforge.core.config import get_settings

router = APIRouter()


def _save_upload(file: UploadFile) -> Path:
    settings = get_settings()
    temp_path = settings.temp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_path


@router.post("/info")
async def audio_info(file: UploadFile = File(...)):
    """Returns audio file information."""
    input_path = _save_upload(file)
    try:
        from mediaforge.audio.processor import AudioProcessor
        ap = AudioProcessor()
        info = ap.get_audio_info(input_path)
        return {
            "format": info.format, "duration": info.duration,
            "audio_codec": info.audio_codec, "channels": info.channels,
            "sample_rate": info.sample_rate, "bitrate": info.bitrate,
            "size_mb": info.size_mb,
        }
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/convert")
async def convert_audio(
    file: UploadFile = File(...),
    target_format: str = Form("mp3"),
    bitrate: str = Form("192k"),
):
    """Converts audio format."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"{input_path.stem}.{target_format}"

    try:
        from mediaforge.audio.converter import AudioConverter
        converter = AudioConverter()
        result = converter.convert(input_path, output_path, target_format, bitrate=bitrate)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/normalize")
async def normalize_audio(
    file: UploadFile = File(...),
    target_loudness: float = Form(-16.0),
):
    """Normalizes audio level."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"normalized_{input_path.name}"

    try:
        from mediaforge.audio.processor import AudioProcessor
        ap = AudioProcessor()
        result = ap.normalize(input_path, output_path, target_loudness=target_loudness)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/trim")
async def trim_audio(
    file: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(None),
    duration: float = Form(None),
):
    """Trims audio file."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"trimmed_{input_path.name}"

    try:
        from mediaforge.audio.processor import AudioProcessor
        ap = AudioProcessor()
        result = ap.trim(input_path, output_path, start_time, end_time, duration)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/waveform")
async def generate_waveform(file: UploadFile = File(...)):
    """Generates a waveform image for audio."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"waveform_{input_path.stem}.png"

    try:
        from mediaforge.audio.analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        result = analyzer.generate_waveform(input_path, output_path)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)
