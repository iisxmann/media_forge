"""
Watermark engine.
Text and image watermarks, tiled watermark, opacity control.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mediaforge.core.base import BaseProcessor, Position, ProcessingResult
from mediaforge.core.exceptions import WatermarkError


class WatermarkEngine(BaseProcessor):
    """Engine for text- and image-based watermarks."""

    def process(self, input_path: str | Path, output_path: str | Path, **kwargs) -> ProcessingResult:
        if "text" in kwargs:
            return self.add_text_watermark(input_path, output_path, **kwargs)
        elif "watermark_path" in kwargs:
            return self.add_image_watermark(input_path, output_path, **kwargs)
        raise WatermarkError("Either 'text' or 'watermark_path' parameter is required")

    def add_text_watermark(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text: str,
        position: str = "bottom-right",
        font_size: int = 24,
        font_path: str | None = None,
        color: tuple[int, int, int, int] = (255, 255, 255, 128),
        opacity: float = 0.5,
        margin: int = 10,
        angle: float = 0,
        quality: int = 95,
        **kwargs,
    ) -> ProcessingResult:
        """
        Adds a text watermark to the image.

        Args:
            text: Watermark text
            position: Position (top-left, top-center, top-right, center, bottom-left, bottom-center, bottom-right)
            font_size: Font size
            font_path: Path to a custom font file
            color: RGBA color (default semi-transparent white)
            opacity: Opacity (0.0-1.0)
            margin: Distance from edges (px)
            angle: Text angle in degrees
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.truetype("arial.ttf", font_size)
            except (IOError, OSError):
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            x, y = self._calculate_position(img.size, (text_w, text_h), position, margin)

            alpha = int(255 * opacity)
            text_color = (*color[:3], alpha)
            draw.text((x, y), text, font=font, fill=text_color)

            if angle != 0:
                overlay = overlay.rotate(angle, expand=False, center=(x + text_w // 2, y + text_h // 2))

            result_img = Image.alpha_composite(img, overlay)
            result_img = result_img.convert("RGB")
            result_img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Text watermark added: '{text}'",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise WatermarkError(f"Text watermark error: {e}")

    def add_image_watermark(
        self,
        input_path: str | Path,
        output_path: str | Path,
        watermark_path: str | Path,
        position: str = "bottom-right",
        opacity: float = 0.5,
        scale: float = 0.2,
        margin: int = 10,
        quality: int = 95,
        **kwargs,
    ) -> ProcessingResult:
        """
        Adds another image as a watermark.

        Args:
            watermark_path: Watermark image file
            position: Position
            opacity: Opacity (0.0-1.0)
            scale: Watermark size relative to the main image
            margin: Distance from edges (px)
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)
        wm_path = self.validate_input(watermark_path)

        try:
            img = Image.open(input_path).convert("RGBA")
            wm = Image.open(wm_path).convert("RGBA")

            wm_w = int(img.size[0] * scale)
            wm_h = int(wm.size[1] * (wm_w / wm.size[0]))
            wm = wm.resize((wm_w, wm_h), Image.LANCZOS)

            if opacity < 1.0:
                alpha = wm.split()[3]
                alpha = alpha.point(lambda p: int(p * opacity))
                wm.putalpha(alpha)

            x, y = self._calculate_position(img.size, wm.size, position, margin)
            img.paste(wm, (x, y), wm)

            result_img = img.convert("RGB")
            result_img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Image watermark added",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise WatermarkError(f"Image watermark error: {e}")

    def add_tiled_watermark(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text: str,
        font_size: int = 20,
        opacity: float = 0.15,
        angle: float = -30,
        spacing: int = 100,
        color: tuple[int, int, int] = (128, 128, 128),
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Adds a repeating (tiled) watermark across the image.
        Suitable for copyright protection.

        Args:
            text: Watermark text
            spacing: Spacing between watermarks (px)
            angle: Overall rotation angle
        """
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        try:
            img = Image.open(input_path).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (IOError, OSError):
                font = ImageFont.load_default()

            alpha = int(255 * opacity)
            text_color = (*color, alpha)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            step_x = text_w + spacing
            step_y = text_h + spacing

            for y in range(-img.size[1], img.size[1] * 2, step_y):
                for x in range(-img.size[0], img.size[0] * 2, step_x):
                    draw.text((x, y), text, font=font, fill=text_color)

            overlay = overlay.rotate(angle, expand=False)
            overlay = overlay.crop((
                (overlay.size[0] - img.size[0]) // 2,
                (overlay.size[1] - img.size[1]) // 2,
                (overlay.size[0] + img.size[0]) // 2,
                (overlay.size[1] + img.size[1]) // 2,
            ))
            if overlay.size != img.size:
                overlay = overlay.resize(img.size)

            result_img = Image.alpha_composite(img, overlay).convert("RGB")
            result_img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Tiled watermark added: '{text}'",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise WatermarkError(f"Tiled watermark error: {e}")

    @staticmethod
    def _calculate_position(
        img_size: tuple[int, int],
        wm_size: tuple[int, int],
        position: str,
        margin: int,
    ) -> tuple[int, int]:
        """Computes watermark position."""
        img_w, img_h = img_size
        wm_w, wm_h = wm_size

        positions = {
            "top-left": (margin, margin),
            "top-center": ((img_w - wm_w) // 2, margin),
            "top-right": (img_w - wm_w - margin, margin),
            "center-left": (margin, (img_h - wm_h) // 2),
            "center": ((img_w - wm_w) // 2, (img_h - wm_h) // 2),
            "center-right": (img_w - wm_w - margin, (img_h - wm_h) // 2),
            "bottom-left": (margin, img_h - wm_h - margin),
            "bottom-center": ((img_w - wm_w) // 2, img_h - wm_h - margin),
            "bottom-right": (img_w - wm_w - margin, img_h - wm_h - margin),
        }
        return positions.get(position, positions["bottom-right"])
