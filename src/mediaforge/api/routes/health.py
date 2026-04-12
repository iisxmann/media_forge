"""Health check and system information endpoints."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from fastapi import APIRouter

from mediaforge import __version__
from mediaforge.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
    }


@router.get("/info")
async def system_info():
    """Detailed system information."""
    settings = get_settings()

    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None

    return {
        "app": settings.app_name,
        "version": __version__,
        "environment": settings.env,
        "dependencies": {
            "ffmpeg": ffmpeg_available,
            "ffprobe": ffprobe_available,
        },
        "supported_formats": {
            "image": settings.image.supported_formats,
            "video": settings.video.supported_formats,
            "audio": settings.audio.supported_formats,
        },
        "directories": {
            "output": str(settings.output_dir),
            "temp": str(settings.temp_dir),
            "cache": str(settings.cache_dir),
        },
    }
