"""
Image format converter.
Conversions between PNG, JPG, WebP, BMP, TIFF, GIF, ICO, and more.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from mediaforge.core.base import BaseConverter, ProcessingResult
from mediaforge.core.exceptions import ConversionError, UnsupportedFormatError
from mediaforge.core.validators import validate_quality


class ImageConverter(BaseConverter):
    """Converter between image formats."""

    SUPPORTED_FORMATS = [
        "jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif",
        "webp", "ico", "ppm", "pgm", "pbm",
    ]

    FORMAT_OPTIONS = {
        "jpg": {"format": "JPEG"},
        "jpeg": {"format": "JPEG"},
        "png": {"format": "PNG"},
        "bmp": {"format": "BMP"},
        "gif": {"format": "GIF"},
        "tiff": {"format": "TIFF"},
        "tif": {"format": "TIFF"},
        "webp": {"format": "WEBP"},
        "ico": {"format": "ICO"},
        "ppm": {"format": "PPM"},
        "pgm": {"format": "PGM"},
        "pbm": {"format": "PBM"},
    }

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_format: str,
        quality: int = 95,
        optimize: bool = True,
        **kwargs,
    ) -> ProcessingResult:
        """
        Converts the image to the target format.

        Args:
            target_format: Target format (jpg, png, webp, etc.)
            quality: Output quality (1-100, for lossy formats)
            optimize: File size optimization
        """
        start = time.time()
        target_format = target_format.lower().lstrip(".")
        if not self.is_format_supported(target_format):
            raise UnsupportedFormatError(target_format, self.SUPPORTED_FORMATS)

        validate_quality(quality)
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(input_path)

            if target_format in ("jpg", "jpeg") and img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif target_format == "png" and img.mode not in ("RGBA", "RGB", "L", "P"):
                img = img.convert("RGBA")

            save_kwargs = {"quality": quality}
            if optimize and target_format in ("jpg", "jpeg", "png"):
                save_kwargs["optimize"] = True
            if target_format == "webp":
                save_kwargs["method"] = kwargs.get("webp_method", 4)
                save_kwargs["lossless"] = kwargs.get("lossless", False)
            if target_format == "png":
                save_kwargs["compress_level"] = kwargs.get("compress_level", 6)

            pil_format = self.FORMAT_OPTIONS[target_format]["format"]
            img.save(output_path, format=pil_format, **save_kwargs)

            src_fmt = input_path.suffix.lstrip(".").upper()
            dst_fmt = target_format.upper()
            self.logger.info(f"Converted: {src_fmt} -> {dst_fmt}")

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Converted {src_fmt} -> {dst_fmt}",
                duration_seconds=time.time() - start,
                details={
                    "source_format": src_fmt,
                    "target_format": dst_fmt,
                    "quality": quality,
                    "optimize": optimize,
                },
            )
        except Exception as e:
            raise ConversionError(f"Conversion error ({target_format}): {e}")

    def batch_convert(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        target_format: str,
        quality: int = 95,
    ) -> list[ProcessingResult]:
        """Batch-converts multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for input_path in input_paths:
            input_path = Path(input_path)
            output_path = output_dir / f"{input_path.stem}.{target_format}"
            try:
                result = self.convert(input_path, output_path, target_format, quality)
                results.append(result)
            except Exception as e:
                results.append(ProcessingResult(
                    success=False, message=f"Error: {input_path.name}: {e}"
                ))

        return results

    def get_supported_formats(self) -> list[str]:
        return self.SUPPORTED_FORMATS.copy()

    def get_format_info(self, format: str) -> dict:
        """Returns information about a format."""
        format_info = {
            "jpg": {"name": "JPEG", "lossy": True, "transparency": False, "animation": False},
            "png": {"name": "PNG", "lossy": False, "transparency": True, "animation": False},
            "webp": {"name": "WebP", "lossy": True, "transparency": True, "animation": True},
            "gif": {"name": "GIF", "lossy": False, "transparency": True, "animation": True},
            "bmp": {"name": "Bitmap", "lossy": False, "transparency": False, "animation": False},
            "tiff": {"name": "TIFF", "lossy": False, "transparency": True, "animation": False},
            "ico": {"name": "Icon", "lossy": False, "transparency": True, "animation": False},
        }
        return format_info.get(format.lower(), {"name": format.upper(), "lossy": None})
