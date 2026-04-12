"""
Image filters.
Blur, sharpen, emboss, edge detection, sepia, vintage, and more.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from mediaforge.core.base import BaseFilter, ProcessingResult
from mediaforge.core.exceptions import FilterError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class BlurFilter(BaseFilter):
    """Gaussian blur filter."""

    def __init__(self):
        super().__init__("blur", "Gaussian blur")

    def apply(self, data: Image.Image, radius: int = 5, **kwargs) -> Image.Image:
        return data.filter(ImageFilter.GaussianBlur(radius=radius))


class BoxBlurFilter(BaseFilter):
    """Box blur filter."""

    def __init__(self):
        super().__init__("box_blur", "Box blur")

    def apply(self, data: Image.Image, radius: int = 5, **kwargs) -> Image.Image:
        return data.filter(ImageFilter.BoxBlur(radius=radius))


class SharpenFilter(BaseFilter):
    """Sharpening filter."""

    def __init__(self):
        super().__init__("sharpen", "Sharpen")

    def apply(self, data: Image.Image, **kwargs) -> Image.Image:
        return data.filter(ImageFilter.SHARPEN)


class UnsharpMaskFilter(BaseFilter):
    """Unsharp mask filter for professional sharpening."""

    def __init__(self):
        super().__init__("unsharp_mask", "Unsharp mask sharpening")

    def apply(
        self, data: Image.Image, radius: int = 2, percent: int = 150, threshold: int = 3, **kwargs
    ) -> Image.Image:
        return data.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


class EmbossFilter(BaseFilter):
    """Emboss effect."""

    def __init__(self):
        super().__init__("emboss", "Emboss effect")

    def apply(self, data: Image.Image, **kwargs) -> Image.Image:
        return data.filter(ImageFilter.EMBOSS)


class EdgeDetectionFilter(BaseFilter):
    """Edge detection filter."""

    def __init__(self):
        super().__init__("edge_detect", "Edge detection")

    def apply(self, data: Image.Image, enhanced: bool = False, **kwargs) -> Image.Image:
        if enhanced:
            return data.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.EDGE_ENHANCE_MORE)
        return data.filter(ImageFilter.FIND_EDGES)


class ContourFilter(BaseFilter):
    """Contour extraction filter."""

    def __init__(self):
        super().__init__("contour", "Contour extraction")

    def apply(self, data: Image.Image, **kwargs) -> Image.Image:
        return data.filter(ImageFilter.CONTOUR)


class SepiaFilter(BaseFilter):
    """Sepia (vintage photo) effect."""

    def __init__(self):
        super().__init__("sepia", "Sepia toning")

    def apply(self, data: Image.Image, intensity: float = 1.0, **kwargs) -> Image.Image:
        img = data.convert("RGB")
        arr = np.array(img, dtype=np.float64)

        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])

        sepia = arr @ sepia_matrix.T
        sepia = np.clip(sepia, 0, 255)

        if intensity < 1.0:
            sepia = arr * (1 - intensity) + sepia * intensity
            sepia = np.clip(sepia, 0, 255)

        return Image.fromarray(sepia.astype(np.uint8))


class VintageFilter(BaseFilter):
    """Vintage (retro) effect."""

    def __init__(self):
        super().__init__("vintage", "Vintage effect")

    def apply(self, data: Image.Image, **kwargs) -> Image.Image:
        img = data.convert("RGB")
        arr = np.array(img, dtype=np.float64)

        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1 + 20, 0, 255)  # Boost red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.9 + 10, 0, 255)  # Reduce green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.8, 0, 255)       # Reduce blue

        img = Image.fromarray(arr.astype(np.uint8))
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(0.85)
        return img


class VignetteFilter(BaseFilter):
    """Vignette (corner darkening) effect."""

    def __init__(self):
        super().__init__("vignette", "Vignette effect")

    def apply(self, data: Image.Image, intensity: float = 0.5, **kwargs) -> Image.Image:
        img = data.convert("RGB")
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape[:2]

        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_dist

        vignette = 1 - dist * intensity
        vignette = np.clip(vignette, 0, 1)
        vignette = np.stack([vignette] * 3, axis=-1)

        result = arr * vignette
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class PosterizeFilter(BaseFilter):
    """Posterize effect. Reduces the number of colors."""

    def __init__(self):
        super().__init__("posterize", "Posterize effect")

    def apply(self, data: Image.Image, bits: int = 4, **kwargs) -> Image.Image:
        img = data.convert("RGB")
        return ImageOps.posterize(img, bits)


class SolarizeFilter(BaseFilter):
    """Solarize effect. Inverts pixels above a threshold."""

    def __init__(self):
        super().__init__("solarize", "Solarize effect")

    def apply(self, data: Image.Image, threshold: int = 128, **kwargs) -> Image.Image:
        return ImageOps.solarize(data.convert("RGB"), threshold=threshold)


class PixelateFilter(BaseFilter):
    """Pixelation effect. Creates a mosaic look."""

    def __init__(self):
        super().__init__("pixelate", "Pixelation")

    def apply(self, data: Image.Image, pixel_size: int = 10, **kwargs) -> Image.Image:
        w, h = data.size
        small = data.resize((w // pixel_size, h // pixel_size), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)


class NoiseFilter(BaseFilter):
    """Random noise filter."""

    def __init__(self):
        super().__init__("noise", "Add noise")

    def apply(self, data: Image.Image, intensity: float = 25.0, **kwargs) -> Image.Image:
        arr = np.array(data, dtype=np.float64)
        noise = np.random.normal(0, intensity, arr.shape)
        result = np.clip(arr + noise, 0, 255)
        return Image.fromarray(result.astype(np.uint8))


class ImageFilterEngine:
    """Engine that manages all filters."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._filters: dict[str, BaseFilter] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            BlurFilter(), BoxBlurFilter(), SharpenFilter(), UnsharpMaskFilter(),
            EmbossFilter(), EdgeDetectionFilter(), ContourFilter(),
            SepiaFilter(), VintageFilter(), VignetteFilter(),
            PosterizeFilter(), SolarizeFilter(), PixelateFilter(), NoiseFilter(),
        ]
        for f in defaults:
            self._filters[f.name] = f

    def register_filter(self, filter: BaseFilter) -> None:
        """Registers a custom filter."""
        self._filters[filter.name] = filter

    def get_filter(self, name: str) -> BaseFilter:
        if name not in self._filters:
            raise FilterError(
                f"Filter not found: {name}. "
                f"Available filters: {', '.join(self._filters.keys())}"
            )
        return self._filters[name]

    def list_filters(self) -> list[dict[str, str]]:
        """Returns a list of available filters."""
        return [{"name": f.name, "description": f.description} for f in self._filters.values()]

    def apply_filter(
        self,
        input_path: str | Path,
        output_path: str | Path,
        filter_name: str,
        quality: int = 95,
        **kwargs,
    ) -> ProcessingResult:
        """Applies a single filter."""
        start = time.time()
        try:
            img = Image.open(input_path)
            filter_obj = self.get_filter(filter_name)
            img = filter_obj.apply(img, **kwargs)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Filter applied: {filter_name}",
                duration_seconds=time.time() - start,
            )
        except FilterError:
            raise
        except Exception as e:
            raise FilterError(f"Filter application error ({filter_name}): {e}")

    def apply_chain(
        self,
        input_path: str | Path,
        output_path: str | Path,
        filters: list[dict[str, Any]],
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Applies a chain of filters.

        Args:
            filters: [{"name": "blur", "params": {"radius": 5}}, {"name": "sepia"}]
        """
        start = time.time()
        try:
            img = Image.open(input_path)
            applied = []

            for filter_config in filters:
                name = filter_config["name"]
                params = filter_config.get("params", {})
                filter_obj = self.get_filter(name)
                img = filter_obj.apply(img, **params)
                applied.append(name)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, quality=quality)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Filter chain applied: {' -> '.join(applied)}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise FilterError(f"Filter chain error: {e}")
