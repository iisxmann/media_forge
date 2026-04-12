"""
Collage builder.
Grid collage, strip collage, photo mosaic.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from PIL import Image, ImageDraw

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class CollageBuilder:
    """Photo collage builder."""

    def create_grid(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
        columns: int = 3,
        cell_size: tuple[int, int] = (400, 400),
        spacing: int = 5,
        background_color: tuple[int, int, int] = (255, 255, 255),
        border_radius: int = 0,
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Creates a grid collage.

        Args:
            image_paths: List of image file paths
            columns: Number of columns
            cell_size: Size of each cell (width, height)
            spacing: Gap between cells (px)
            background_color: Background color (RGB)
            border_radius: Corner radius (px)
        """
        start = time.time()
        if not image_paths:
            raise ProcessingError("At least 1 image is required")

        try:
            rows = math.ceil(len(image_paths) / columns)
            canvas_w = columns * cell_size[0] + (columns + 1) * spacing
            canvas_h = rows * cell_size[1] + (rows + 1) * spacing

            canvas = Image.new("RGB", (canvas_w, canvas_h), background_color)

            for idx, img_path in enumerate(image_paths):
                row = idx // columns
                col = idx % columns

                x = spacing + col * (cell_size[0] + spacing)
                y = spacing + row * (cell_size[1] + spacing)

                img = Image.open(img_path)
                img = self._fit_image(img, cell_size)

                if border_radius > 0:
                    img = self._round_corners(img, border_radius)

                canvas.paste(img, (x, y))

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Grid collage created: {len(image_paths)} images ({columns}x{rows})",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Grid collage error: {e}")

    def create_horizontal_strip(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
        height: int = 400,
        spacing: int = 5,
        background_color: tuple[int, int, int] = (255, 255, 255),
        quality: int = 95,
    ) -> ProcessingResult:
        """Creates a horizontal strip collage. Images are placed side by side."""
        start = time.time()
        if not image_paths:
            raise ProcessingError("At least 1 image is required")

        try:
            images = []
            total_width = spacing

            for img_path in image_paths:
                img = Image.open(img_path)
                ratio = height / img.size[1]
                new_w = int(img.size[0] * ratio)
                img = img.resize((new_w, height), Image.LANCZOS)
                images.append(img)
                total_width += new_w + spacing

            canvas = Image.new("RGB", (total_width, height + 2 * spacing), background_color)
            x = spacing
            for img in images:
                canvas.paste(img, (x, spacing))
                x += img.size[0] + spacing

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Horizontal collage created: {len(images)} images",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Horizontal collage error: {e}")

    def create_vertical_strip(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
        width: int = 600,
        spacing: int = 5,
        background_color: tuple[int, int, int] = (255, 255, 255),
        quality: int = 95,
    ) -> ProcessingResult:
        """Creates a vertical strip collage. Images are stacked top to bottom."""
        start = time.time()
        if not image_paths:
            raise ProcessingError("At least 1 image is required")

        try:
            images = []
            total_height = spacing

            for img_path in image_paths:
                img = Image.open(img_path)
                ratio = width / img.size[0]
                new_h = int(img.size[1] * ratio)
                img = img.resize((width, new_h), Image.LANCZOS)
                images.append(img)
                total_height += new_h + spacing

            canvas = Image.new("RGB", (width + 2 * spacing, total_height), background_color)
            y = spacing
            for img in images:
                canvas.paste(img, (spacing, y))
                y += img.size[1] + spacing

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Vertical collage created: {len(images)} images",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Vertical collage error: {e}")

    def create_photo_mosaic(
        self,
        main_image_path: str | Path,
        tile_paths: list[str | Path],
        output_path: str | Path,
        tile_size: int = 20,
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Creates a photo mosaic. Rebuilds the main image from small photos.

        Args:
            main_image_path: Main source image
            tile_paths: Images to use as mosaic tiles
            tile_size: Tile size in pixels
        """
        start = time.time()
        import numpy as np

        try:
            main = Image.open(main_image_path).convert("RGB")
            main_w, main_h = main.size

            cols = main_w // tile_size
            rows = main_h // tile_size
            main = main.resize((cols * tile_size, rows * tile_size))

            tiles = []
            for tp in tile_paths:
                try:
                    t = Image.open(tp).convert("RGB").resize((tile_size, tile_size), Image.LANCZOS)
                    avg = np.array(t).mean(axis=(0, 1))
                    tiles.append((t, avg))
                except Exception:
                    continue

            if not tiles:
                raise ProcessingError("No usable mosaic tiles found")

            tile_avgs = np.array([t[1] for t in tiles])
            mosaic = Image.new("RGB", main.size)

            for row in range(rows):
                for col in range(cols):
                    x, y = col * tile_size, row * tile_size
                    region = main.crop((x, y, x + tile_size, y + tile_size))
                    region_avg = np.array(region).mean(axis=(0, 1))

                    distances = np.sqrt(((tile_avgs - region_avg) ** 2).sum(axis=1))
                    best_idx = distances.argmin()
                    mosaic.paste(tiles[best_idx][0], (x, y))

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mosaic.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Photo mosaic created: {cols}x{rows} tiles",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Mosaic error: {e}")

    @staticmethod
    def _fit_image(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """Fits the image to the target size (crop + resize)."""
        from PIL import ImageOps
        return ImageOps.fit(img, target_size, Image.LANCZOS)

    @staticmethod
    def _round_corners(img: Image.Image, radius: int) -> Image.Image:
        """Rounds image corners."""
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, *img.size], radius=radius, fill=255)
        img.putalpha(mask)
        return img
