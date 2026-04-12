"""General helper functions."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from mediaforge.core.base import MediaType


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".svg", ".ico", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".3gp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".m4a", ".opus"}


def get_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """Computes file hash."""
    path = Path(file_path)
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def human_readable_size(size_bytes: int) -> str:
    """Converts byte value to human-readable format (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Converts seconds to a human-readable duration string."""
    if seconds < 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}min")
    if s > 0 or not parts:
        parts.append(f"{s}s")
    if ms > 0 and not h:
        parts.append(f"{ms}ms")

    return " ".join(parts)


def ensure_even_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    Video encoding requires even dimensions.
    Rounds odd values up to the next even number.
    """
    return (width + width % 2, height + height % 2)


def generate_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    suffix: str = "",
    extension: str | None = None,
) -> Path:
    """Builds output file path."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = extension or input_path.suffix
    if not ext.startswith("."):
        ext = f".{ext}"

    name = f"{input_path.stem}{suffix}{ext}"
    output_path = output_dir / name

    if output_path.exists():
        name = f"{input_path.stem}{suffix}_{uuid.uuid4().hex[:8]}{ext}"
        output_path = output_dir / name

    return output_path


def get_media_type(file_path: str | Path) -> MediaType | None:
    """Returns media type based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    elif ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    elif ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    return None


def is_image(file_path: str | Path) -> bool:
    return get_media_type(file_path) == MediaType.IMAGE


def is_video(file_path: str | Path) -> bool:
    return get_media_type(file_path) == MediaType.VIDEO


def is_audio(file_path: str | Path) -> bool:
    return get_media_type(file_path) == MediaType.AUDIO
