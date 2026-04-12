"""AI-powered operations API endpoints."""

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


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    model_size: str = Form("base"),
):
    """Transcribes audio/video to text (Whisper)."""
    input_path = _save_upload(file)
    try:
        from mediaforge.ai.transcription import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size=model_size)
        return transcriber.transcribe(input_path, language=language)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    format: str = Form("srt"),
    language: str = Form(None),
    model_size: str = Form("base"),
):
    """Generates subtitles from audio/video."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"{input_path.stem}.{format}"

    try:
        from mediaforge.ai.transcription import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size=model_size)
        result = transcriber.generate_subtitles(input_path, output_path, format=format, language=language)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...), blur: bool = Form(False)):
    """Face detection and optional blurring."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"faces_{input_path.name}"

    try:
        from mediaforge.ai.face_detection import FaceDetector
        detector = FaceDetector()

        if blur:
            result = detector.blur_faces(input_path, output_path)
            if result.success:
                return FileResponse(str(result.output_path), filename=output_path.name)
        else:
            result = detector.draw_faces(input_path, output_path)
            if result.success:
                return FileResponse(str(result.output_path), filename=output_path.name)

        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """Object detection (YOLO)."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"objects_{input_path.name}"

    try:
        from mediaforge.ai.object_detection import ObjectDetector
        detector = ObjectDetector()
        result = detector.detect_and_draw(input_path, output_path)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Background removal."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"nobg_{input_path.stem}.png"

    try:
        from mediaforge.ai.background_removal import BackgroundRemover
        remover = BackgroundRemover()
        result = remover.remove_background(input_path, output_path)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/ocr")
async def ocr(file: UploadFile = File(...), engine: str = Form("tesseract")):
    """Extracts text from image (OCR)."""
    input_path = _save_upload(file)
    try:
        from mediaforge.ai.ocr import OCREngine
        ocr_engine = OCREngine(engine=engine)
        return ocr_engine.extract_text(input_path)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(2)):
    """Image upscaling (super resolution)."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"upscaled_{scale}x_{input_path.name}"

    try:
        from mediaforge.ai.super_resolution import SuperResolution
        sr = SuperResolution(scale=scale)
        result = sr.upscale(input_path, output_path)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)
