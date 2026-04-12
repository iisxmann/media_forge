"""Video processing API endpoints."""

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
async def video_info(file: UploadFile = File(...)):
    """Returns video information."""
    input_path = _save_upload(file)
    try:
        from mediaforge.video.processor import VideoProcessor
        vp = VideoProcessor()
        info = vp.get_video_info(input_path)
        return {
            "format": info.format, "duration": info.duration,
            "width": info.width, "height": info.height,
            "fps": info.fps, "codec": info.codec,
            "audio_codec": info.audio_codec, "size_mb": info.size_mb,
            "bitrate": info.bitrate,
        }
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/convert")
async def convert_video(
    file: UploadFile = File(...),
    target_format: str = Form("mp4"),
    quality_preset: str = Form("medium"),
):
    """Converts video format."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"{input_path.stem}.{target_format}"

    try:
        from mediaforge.video.converter import VideoConverter
        converter = VideoConverter()
        result = converter.convert(input_path, output_path, target_format, quality_preset=quality_preset)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/compress")
async def compress_video(
    file: UploadFile = File(...),
    target_size_mb: float = Form(None),
    crf: int = Form(28),
):
    """Compresses video."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"compressed_{input_path.name}"

    try:
        from mediaforge.video.converter import VideoConverter
        converter = VideoConverter()
        result = converter.compress(input_path, output_path, target_size_mb=target_size_mb, crf=crf)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/trim")
async def trim_video(
    file: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(None),
    duration: float = Form(None),
):
    """Trims video."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"trimmed_{input_path.name}"

    try:
        from mediaforge.video.editor import VideoEditor
        editor = VideoEditor()
        result = editor.trim(input_path, output_path, start_time, end_time, duration)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/extract-audio")
async def extract_audio(
    file: UploadFile = File(...),
    audio_format: str = Form("mp3"),
    bitrate: str = Form("192k"),
):
    """Extracts audio from video."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"{input_path.stem}.{audio_format}"

    try:
        from mediaforge.video.processor import VideoProcessor
        vp = VideoProcessor()
        result = vp.extract_audio(input_path, output_path, audio_format, bitrate)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/gif")
async def create_gif(
    file: UploadFile = File(...),
    start_time: float = Form(0),
    duration: float = Form(5),
    fps: int = Form(15),
    width: int = Form(480),
):
    """Creates a GIF from video."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"{input_path.stem}.gif"

    try:
        from mediaforge.video.processor import VideoProcessor
        vp = VideoProcessor()
        result = vp.create_gif(input_path, output_path, start_time, duration, fps, width)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/quality")
async def analyze_quality(file: UploadFile = File(...)):
    """Runs video quality analysis."""
    input_path = _save_upload(file)
    try:
        from mediaforge.video.quality import VideoQualityAnalyzer
        analyzer = VideoQualityAnalyzer()
        return analyzer.analyze(input_path)
    finally:
        input_path.unlink(missing_ok=True)
