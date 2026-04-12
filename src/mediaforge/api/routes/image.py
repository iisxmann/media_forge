"""Image processing API endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse

from mediaforge.core.config import get_settings
from mediaforge.image.processor import ImageProcessor
from mediaforge.image.converter import ImageConverter
from mediaforge.image.watermark import WatermarkEngine
from mediaforge.image.filters import ImageFilterEngine
from mediaforge.image.metadata import ImageMetadataManager

router = APIRouter()


def _save_upload(file: UploadFile) -> Path:
    """Saves the uploaded file to the temporary directory."""
    settings = get_settings()
    temp_path = settings.temp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_path


@router.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None),
    keep_aspect_ratio: bool = Form(True),
    quality: int = Form(95),
):
    """Resizes an image."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"resized_{input_path.name}"

    try:
        processor = ImageProcessor()
        result = processor.resize(input_path, output_path, width=width, height=height,
                                  keep_aspect_ratio=keep_aspect_ratio, quality=quality)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    left: int = Form(...),
    top: int = Form(...),
    right: int = Form(...),
    bottom: int = Form(...),
    quality: int = Form(95),
):
    """Crops an image."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"cropped_{input_path.name}"

    try:
        processor = ImageProcessor()
        result = processor.crop(input_path, output_path, left, top, right, bottom, quality)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/convert")
async def convert_image(
    file: UploadFile = File(...),
    target_format: str = Form(...),
    quality: int = Form(95),
):
    """Converts image format."""
    input_path = _save_upload(file)
    stem = input_path.stem
    output_path = get_settings().output_dir / f"{stem}.{target_format}"

    try:
        converter = ImageConverter()
        result = converter.convert(input_path, output_path, target_format, quality)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/watermark/text")
async def add_text_watermark(
    file: UploadFile = File(...),
    text: str = Form(...),
    position: str = Form("bottom-right"),
    font_size: int = Form(24),
    opacity: float = Form(0.5),
):
    """Adds a text watermark."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"watermarked_{input_path.name}"

    try:
        engine = WatermarkEngine()
        result = engine.add_text_watermark(
            input_path, output_path, text=text, position=position,
            font_size=font_size, opacity=opacity,
        )

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/filter")
async def apply_filter(
    file: UploadFile = File(...),
    filter_name: str = Form(...),
    quality: int = Form(95),
):
    """Applies an image filter."""
    input_path = _save_upload(file)
    output_path = get_settings().output_dir / f"filtered_{input_path.name}"

    try:
        engine = ImageFilterEngine()
        result = engine.apply_filter(input_path, output_path, filter_name, quality=quality)

        if result.success:
            return FileResponse(str(result.output_path), filename=output_path.name)
        raise HTTPException(500, result.message)
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/metadata")
async def read_metadata(file: UploadFile = File(...)):
    """Reads image metadata."""
    input_path = _save_upload(file)
    try:
        manager = ImageMetadataManager()
        return manager.read_metadata(input_path)
    finally:
        input_path.unlink(missing_ok=True)


@router.get("/filters")
async def list_filters():
    """Lists available filters."""
    engine = ImageFilterEngine()
    return {"filters": engine.list_filters()}


@router.get("/formats")
async def list_formats():
    """Lists supported formats."""
    converter = ImageConverter()
    return {"formats": converter.get_supported_formats()}
