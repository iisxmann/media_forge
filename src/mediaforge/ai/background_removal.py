"""
Background removal module.
AI-based background removal and replacement.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import AIModelError, ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class BackgroundRemover:
    """AI-based background removal class."""

    def __init__(self, method: str = "rembg"):
        """
        Args:
            method: Background removal method
                - 'rembg': rembg library (U2Net-based, best quality)
                - 'grabcut': OpenCV GrabCut (no model required)
        """
        self.method = method

    def remove_background(
        self,
        input_path: str | Path,
        output_path: str | Path,
        alpha_matting: bool = False,
    ) -> ProcessingResult:
        """
        Removes the background and saves as a transparent PNG.

        Args:
            alpha_matting: Edge refinement (better hair/fur detail)
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.method == "rembg":
                return self._remove_rembg(input_path, output_path, alpha_matting, start)
            elif self.method == "grabcut":
                return self._remove_grabcut(input_path, output_path, start)
            else:
                raise ProcessingError(f"Unknown method: {self.method}")
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Background removal error: {e}")

    def replace_background(
        self,
        input_path: str | Path,
        background_path: str | Path,
        output_path: str | Path,
    ) -> ProcessingResult:
        """Replaces the background with another image."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            temp_path = output_path.parent / f"_temp_nobg_{output_path.stem}.png"
            self.remove_background(input_path, temp_path)

            fg = Image.open(temp_path).convert("RGBA")
            bg = Image.open(background_path).convert("RGBA")

            bg = bg.resize(fg.size, Image.LANCZOS)
            bg.paste(fg, (0, 0), fg)

            result = bg.convert("RGB")
            result.save(output_path, quality=95)

            if temp_path.exists():
                temp_path.unlink()

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Background replaced",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Background replace error: {e}")

    def replace_with_color(
        self,
        input_path: str | Path,
        output_path: str | Path,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> ProcessingResult:
        """Replaces the background with a solid color."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            temp_path = output_path.parent / f"_temp_nobg_{output_path.stem}.png"
            self.remove_background(input_path, temp_path)

            fg = Image.open(temp_path).convert("RGBA")
            bg = Image.new("RGBA", fg.size, (*color, 255))
            bg.paste(fg, (0, 0), fg)

            result = bg.convert("RGB")
            result.save(output_path, quality=95)

            if temp_path.exists():
                temp_path.unlink()

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Background replaced with color: RGB{color}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Color replace error: {e}")

    def blur_background(
        self,
        input_path: str | Path,
        output_path: str | Path,
        blur_radius: int = 25,
    ) -> ProcessingResult:
        """Blurs the background (portrait-mode effect)."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            temp_path = output_path.parent / f"_temp_nobg_{output_path.stem}.png"
            self.remove_background(input_path, temp_path)

            fg = Image.open(temp_path).convert("RGBA")
            original = Image.open(input_path).convert("RGBA")

            from PIL import ImageFilter
            blurred_bg = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_bg.paste(fg, (0, 0), fg)

            result = blurred_bg.convert("RGB")
            result.save(output_path, quality=95)

            if temp_path.exists():
                temp_path.unlink()

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Background blurred (portrait mode)",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Background blur error: {e}")

    def _remove_rembg(
        self, input_path, output_path, alpha_matting, start
    ) -> ProcessingResult:
        try:
            from rembg import remove

            with open(input_path, "rb") as f:
                input_data = f.read()

            output_data = remove(
                input_data,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=240 if alpha_matting else 0,
                alpha_matting_background_threshold=10 if alpha_matting else 0,
            )

            with open(output_path, "wb") as f:
                f.write(output_data)

            return ProcessingResult(
                success=True, output_path=Path(output_path),
                message="Background removed (rembg)",
                duration_seconds=time.time() - start,
            )
        except ImportError:
            raise AIModelError("rembg package required: pip install rembg")

    def _remove_grabcut(self, input_path, output_path, start) -> ProcessingResult:
        img = cv2.imread(str(input_path))
        if img is None:
            raise ProcessingError(f"Could not read image: {input_path}")

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        margin = int(min(w, h) * 0.05)
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask2 * 255

        cv2.imwrite(str(output_path), result)

        return ProcessingResult(
            success=True, output_path=Path(output_path),
            message="Background removed (GrabCut)",
            duration_seconds=time.time() - start,
        )
