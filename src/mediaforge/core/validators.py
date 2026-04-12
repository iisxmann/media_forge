"""Input validation functions."""

from __future__ import annotations

from pathlib import Path

from mediaforge.core.config import get_settings
from mediaforge.core.exceptions import (
    UnsupportedFormatError,
    InvalidResolutionError,
    ValidationError,
)


def validate_image_format(file_path: str | Path) -> bool:
    """Checks whether the image file uses a supported format."""
    ext = Path(file_path).suffix.lstrip(".").lower()
    settings = get_settings()
    if ext not in settings.image.supported_formats:
        raise UnsupportedFormatError(ext, settings.image.supported_formats)
    return True


def validate_video_format(file_path: str | Path) -> bool:
    """Checks whether the video file uses a supported format."""
    ext = Path(file_path).suffix.lstrip(".").lower()
    settings = get_settings()
    if ext not in settings.video.supported_formats:
        raise UnsupportedFormatError(ext, settings.video.supported_formats)
    return True


def validate_audio_format(file_path: str | Path) -> bool:
    """Checks whether the audio file uses a supported format."""
    ext = Path(file_path).suffix.lstrip(".").lower()
    settings = get_settings()
    if ext not in settings.audio.supported_formats:
        raise UnsupportedFormatError(ext, settings.audio.supported_formats)
    return True


def validate_resolution(width: int, height: int, max_resolution: int | None = None) -> bool:
    """Validates width and height."""
    if width <= 0 or height <= 0:
        raise InvalidResolutionError(f"Resolution values must be positive: {width}x{height}")

    max_res = max_resolution or get_settings().image.max_resolution
    if width > max_res or height > max_res:
        raise InvalidResolutionError(
            f"Resolution exceeds maximum limit: {width}x{height} (max: {max_res})"
        )
    return True


def validate_fps(fps: float) -> bool:
    """Validates the FPS value."""
    if fps <= 0 or fps > 240:
        raise ValidationError(f"Invalid FPS value: {fps}. Must be between 0 and 240.")
    return True


def validate_quality(quality: int) -> bool:
    """Validates the quality value (1-100)."""
    if not 1 <= quality <= 100:
        raise ValidationError(f"Quality must be between 1 and 100: {quality}")
    return True


def validate_opacity(opacity: float) -> bool:
    """Validates the opacity value (0.0-1.0)."""
    if not 0.0 <= opacity <= 1.0:
        raise ValidationError(f"Opacity must be between 0.0 and 1.0: {opacity}")
    return True


def validate_file_exists(file_path: str | Path) -> Path:
    """Checks that the file exists."""
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    return path


def validate_directory_exists(dir_path: str | Path, create: bool = False) -> Path:
    """Checks that the directory exists; optionally creates it."""
    path = Path(dir_path)
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValidationError(f"Directory not found: {path}")
    return path


def validate_aspect_ratio(ratio: str) -> tuple[int, int]:
    """Parses and validates an aspect ratio string. E.g. '16:9' -> (16, 9)."""
    try:
        parts = ratio.split(":")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except (ValueError, IndexError):
        raise ValidationError(f"Invalid aspect ratio: {ratio}. Examples: '16:9', '4:3'")


def get_format_from_path(file_path: str | Path) -> str:
    """Returns the format extension from a file path."""
    return Path(file_path).suffix.lstrip(".").lower()
