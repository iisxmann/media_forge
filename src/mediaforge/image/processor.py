"""
Image processing module.
Resize, crop, rotate, flip, and basic image manipulation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from mediaforge.core.base import BaseProcessor, MediaInfo, ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.validators import validate_image_format, validate_resolution, validate_quality


class ImageProcessor(BaseProcessor):
    """
    Comprehensive image processing class.
    Resize, crop, rotate, flip, brightness/contrast adjustment, etc.
    """

    def process(self, input_path: str | Path, output_path: str | Path, **kwargs) -> ProcessingResult:
        """Processes the image according to the given parameters."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)
        validate_image_format(input_path)

        try:
            img = Image.open(input_path)
            original_info = self._build_media_info(input_path, img)

            for operation, value in kwargs.items():
                img = self._apply_operation(img, operation, value)

            quality = kwargs.get("quality", 95)
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message="Image processed successfully",
                duration_seconds=time.time() - start,
                input_info=original_info,
                output_info=self._build_media_info(output_path, Image.open(output_path)),
            )
        except Exception as e:
            raise ProcessingError(f"Error while processing image: {e}")

    def resize(
        self,
        input_path: str | Path,
        output_path: str | Path,
        width: int | None = None,
        height: int | None = None,
        keep_aspect_ratio: bool = True,
        resample: str = "lanczos",
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Resizes the image.

        Args:
            width: Target width (px)
            height: Target height (px)
            keep_aspect_ratio: Preserve aspect ratio
            resample: Resampling method (lanczos, bilinear, bicubic, nearest)
            quality: Output quality (1-100)
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        resample_methods = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }

        try:
            img = Image.open(input_path)
            orig_w, orig_h = img.size

            if width and height and not keep_aspect_ratio:
                new_size = (width, height)
            elif width and height and keep_aspect_ratio:
                ratio = min(width / orig_w, height / orig_h)
                new_size = (int(orig_w * ratio), int(orig_h * ratio))
            elif width:
                ratio = width / orig_w
                new_size = (width, int(orig_h * ratio))
            elif height:
                ratio = height / orig_h
                new_size = (int(orig_w * ratio), height)
            else:
                raise ProcessingError("At least width or height must be specified")

            validate_resolution(*new_size)
            img = img.resize(new_size, resample_methods.get(resample, Image.LANCZOS))
            img.save(output_path, quality=quality)

            self.logger.info(f"Resize: {orig_w}x{orig_h} -> {new_size[0]}x{new_size[1]}")
            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Resized: {new_size[0]}x{new_size[1]}",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Resize error: {e}")

    def crop(
        self,
        input_path: str | Path,
        output_path: str | Path,
        left: int,
        top: int,
        right: int,
        bottom: int,
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Crops the image.

        Args:
            left, top, right, bottom: Crop coordinates (px)
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            w, h = img.size

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            if left >= right or top >= bottom:
                raise ProcessingError(f"Invalid crop coordinates: ({left},{top})-({right},{bottom})")

            img = img.crop((left, top, right, bottom))
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Cropped: ({left},{top})-({right},{bottom}) -> {img.size[0]}x{img.size[1]}",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Crop error: {e}")

    def auto_crop(
        self,
        input_path: str | Path,
        output_path: str | Path,
        aspect_ratio: str = "16:9",
        focus: str = "center",
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Automatically crops to the given aspect ratio.

        Args:
            aspect_ratio: Target ratio (e.g. '16:9', '4:3', '1:1')
            focus: Focus point (center, top, bottom, left, right)
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            w, h = img.size
            ar_parts = aspect_ratio.split(":")
            target_ratio = int(ar_parts[0]) / int(ar_parts[1])
            current_ratio = w / h

            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                new_h = h
            else:
                new_w = w
                new_h = int(w / target_ratio)

            offsets = {
                "center": ((w - new_w) // 2, (h - new_h) // 2),
                "top": ((w - new_w) // 2, 0),
                "bottom": ((w - new_w) // 2, h - new_h),
                "left": (0, (h - new_h) // 2),
                "right": (w - new_w, (h - new_h) // 2),
            }
            x, y = offsets.get(focus, offsets["center"])
            img = img.crop((x, y, x + new_w, y + new_h))
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Auto-cropped ({aspect_ratio}): {new_w}x{new_h}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Auto crop error: {e}")

    def rotate(
        self,
        input_path: str | Path,
        output_path: str | Path,
        angle: float,
        expand: bool = True,
        fill_color: tuple[int, int, int] = (0, 0, 0),
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Rotates the image.

        Args:
            angle: Rotation angle in degrees (counter-clockwise)
            expand: Include overflow after rotation
            fill_color: RGB color to fill empty areas
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            img = img.rotate(angle, expand=expand, fillcolor=fill_color, resample=Image.BICUBIC)
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Rotated {angle}°",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Rotate error: {e}")

    def flip(
        self,
        input_path: str | Path,
        output_path: str | Path,
        direction: str = "horizontal",
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Flips the image.

        Args:
            direction: Flip direction ('horizontal' or 'vertical')
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            if direction == "horizontal":
                img = ImageOps.mirror(img)
            elif direction == "vertical":
                img = ImageOps.flip(img)
            else:
                raise ProcessingError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Flipped {direction}",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Flip error: {e}")

    def adjust_brightness(
        self, input_path: str | Path, output_path: str | Path, factor: float = 1.5, quality: int = 95
    ) -> ProcessingResult:
        """Adjusts brightness. factor=1.0 is original, >1.0 brighter, <1.0 darker."""
        return self._enhance(input_path, output_path, "brightness", factor, quality)

    def adjust_contrast(
        self, input_path: str | Path, output_path: str | Path, factor: float = 1.5, quality: int = 95
    ) -> ProcessingResult:
        """Adjusts contrast. factor=1.0 is original."""
        return self._enhance(input_path, output_path, "contrast", factor, quality)

    def adjust_saturation(
        self, input_path: str | Path, output_path: str | Path, factor: float = 1.5, quality: int = 95
    ) -> ProcessingResult:
        """Adjusts saturation. factor=1.0 is original, 0.0 grayscale."""
        return self._enhance(input_path, output_path, "saturation", factor, quality)

    def adjust_sharpness(
        self, input_path: str | Path, output_path: str | Path, factor: float = 2.0, quality: int = 95
    ) -> ProcessingResult:
        """Adjusts sharpness. factor=1.0 is original."""
        return self._enhance(input_path, output_path, "sharpness", factor, quality)

    def grayscale(
        self, input_path: str | Path, output_path: str | Path, quality: int = 95
    ) -> ProcessingResult:
        """Converts the image to grayscale."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path).convert("L")
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Converted to grayscale", duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Grayscale error: {e}")

    def invert(
        self, input_path: str | Path, output_path: str | Path, quality: int = 95
    ) -> ProcessingResult:
        """Inverts colors (negative)."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            if img.mode == "RGBA":
                r, g, b, a = img.split()
                rgb = Image.merge("RGB", (r, g, b))
                rgb = ImageOps.invert(rgb)
                img = Image.merge("RGBA", (*rgb.split(), a))
            else:
                img = ImageOps.invert(img.convert("RGB"))
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Colors inverted", duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Invert error: {e}")

    def equalize_histogram(
        self, input_path: str | Path, output_path: str | Path, quality: int = 95
    ) -> ProcessingResult:
        """Applies histogram equalization for contrast improvement."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            img = ImageOps.equalize(img)
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Histogram equalized", duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Histogram error: {e}")

    def auto_enhance(
        self, input_path: str | Path, output_path: str | Path, quality: int = 95
    ) -> ProcessingResult:
        """Auto enhancement: autocontrast + sharpness + color."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Sharpness(img).enhance(1.3)
            img = ImageEnhance.Color(img).enhance(1.1)
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Auto enhancement applied", duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Auto enhance error: {e}")

    def pad(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_width: int,
        target_height: int,
        color: tuple[int, int, int] = (0, 0, 0),
        quality: int = 95,
    ) -> ProcessingResult:
        """Pads the image to the target size."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path)
            img = ImageOps.pad(img, (target_width, target_height), color=color)
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Padding applied: {target_width}x{target_height}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Pad error: {e}")

    def _enhance(
        self, input_path: str | Path, output_path: str | Path,
        enhancement_type: str, factor: float, quality: int
    ) -> ProcessingResult:
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        enhancers = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
            "sharpness": ImageEnhance.Sharpness,
        }

        try:
            img = Image.open(input_path)
            enhancer = enhancers[enhancement_type](img)
            img = enhancer.enhance(factor)
            img.save(output_path, quality=quality)
            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"{enhancement_type} adjusted (factor={factor})",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"{enhancement_type} error: {e}")

    def _apply_operation(self, img: Image.Image, operation: str, value: Any) -> Image.Image:
        operations = {
            "resize": lambda: img.resize(value, Image.LANCZOS) if isinstance(value, tuple) else img,
            "rotate": lambda: img.rotate(value, expand=True, resample=Image.BICUBIC),
            "flip_h": lambda: ImageOps.mirror(img) if value else img,
            "flip_v": lambda: ImageOps.flip(img) if value else img,
            "grayscale": lambda: img.convert("L") if value else img,
            "brightness": lambda: ImageEnhance.Brightness(img).enhance(value),
            "contrast": lambda: ImageEnhance.Contrast(img).enhance(value),
            "saturation": lambda: ImageEnhance.Color(img).enhance(value),
            "sharpness": lambda: ImageEnhance.Sharpness(img).enhance(value),
        }
        if operation in operations:
            return operations[operation]()
        return img

    def _build_media_info(self, path: Path, img: Image.Image) -> MediaInfo:
        return MediaInfo(
            path=path,
            format=path.suffix.lstrip(".").lower(),
            size_bytes=path.stat().st_size,
            width=img.size[0],
            height=img.size[1],
            metadata=dict(img.info) if hasattr(img, "info") else {},
        )
