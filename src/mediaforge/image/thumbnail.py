"""
Thumbnail generator.
Produces thumbnails at various sizes and formats.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageOps

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


PRESET_SIZES = {
    "icon": (64, 64),
    "small": (128, 128),
    "medium": (256, 256),
    "large": (512, 512),
    "social_square": (1080, 1080),
    "social_story": (1080, 1920),
    "social_cover": (1200, 630),
    "youtube": (1280, 720),
    "twitter_header": (1500, 500),
    "favicon": (32, 32),
}


class ThumbnailGenerator:
    """Creates thumbnails at various sizes and formats."""

    def generate(
        self,
        input_path: str | Path,
        output_path: str | Path,
        size: tuple[int, int] = (256, 256),
        mode: str = "fit",
        quality: int = 90,
    ) -> ProcessingResult:
        """
        Generates a thumbnail.

        Args:
            size: Target size (width, height)
            mode: Resize mode
                - 'fit': Fit inside bounds while preserving aspect ratio
                - 'fill': Preserve aspect ratio, fill and crop
                - 'stretch': Resize to exact dimensions (may distort aspect ratio)
                - 'pad': Preserve aspect ratio with padding
        """
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(input_path)

            if mode == "fit":
                img.thumbnail(size, Image.LANCZOS)
            elif mode == "fill":
                img = ImageOps.fit(img, size, Image.LANCZOS)
            elif mode == "stretch":
                img = img.resize(size, Image.LANCZOS)
            elif mode == "pad":
                img = ImageOps.pad(img, size, Image.LANCZOS, color=(255, 255, 255))
            else:
                raise ProcessingError(f"Invalid mode: {mode}. Must be fit, fill, stretch, or pad")

            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Thumbnail created: {img.size[0]}x{img.size[1]} ({mode})",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Thumbnail error: {e}")

    def generate_preset(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        preset: str,
        mode: str = "fill",
        quality: int = 90,
    ) -> ProcessingResult:
        """Generates a thumbnail using a built-in preset size."""
        if preset not in PRESET_SIZES:
            available = ", ".join(PRESET_SIZES.keys())
            raise ProcessingError(f"Invalid preset: {preset}. Available: {available}")

        size = PRESET_SIZES[preset]
        output_dir = Path(output_dir)
        stem = Path(input_path).stem
        output_path = output_dir / f"{stem}_{preset}.jpg"

        return self.generate(input_path, output_path, size, mode, quality)

    def generate_multiple(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        sizes: list[tuple[int, int]] | None = None,
        presets: list[str] | None = None,
        mode: str = "fill",
        quality: int = 90,
    ) -> list[ProcessingResult]:
        """Generates thumbnails at multiple sizes."""
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        if sizes:
            for size in sizes:
                out = output_dir / f"{stem}_{size[0]}x{size[1]}.jpg"
                results.append(self.generate(input_path, out, size, mode, quality))

        if presets:
            for preset in presets:
                results.append(self.generate_preset(input_path, output_dir, preset, mode, quality))

        return results

    def generate_responsive_set(
        self, input_path: str | Path, output_dir: str | Path, quality: int = 85
    ) -> list[ProcessingResult]:
        """Creates a responsive image set for the web (widths 320, 640, 768, 1024, 1280, 1920 px)."""
        widths = [320, 640, 768, 1024, 1280, 1920]
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        img = Image.open(input_path)
        orig_w, orig_h = img.size

        for w in widths:
            if w > orig_w:
                continue
            h = int(orig_h * (w / orig_w))
            out = output_dir / f"{stem}_{w}w.jpg"
            results.append(self.generate(input_path, out, (w, h), "fit", quality))

        return results

    @staticmethod
    def get_available_presets() -> dict[str, tuple[int, int]]:
        """Returns available preset sizes."""
        return PRESET_SIZES.copy()
