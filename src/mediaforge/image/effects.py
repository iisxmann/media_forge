"""
Advanced image effects.
HDR, panorama, histogram analysis, color palette extraction, depth maps, etc.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class ImageEffects:
    """Advanced image effects."""

    def histogram_analysis(self, file_path: str | Path) -> dict[str, Any]:
        """Runs histogram analysis on the image. Returns per-channel statistics."""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            channels = {"blue": 0, "green": 1, "red": 2}
            result = {}

            for name, idx in channels.items():
                channel = img[:, :, idx].flatten()
                result[name] = {
                    "mean": float(np.mean(channel)),
                    "std": float(np.std(channel)),
                    "min": int(np.min(channel)),
                    "max": int(np.max(channel)),
                    "median": float(np.median(channel)),
                }

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result["luminance"] = {
                "mean": float(np.mean(gray)),
                "std": float(np.std(gray)),
                "is_dark": float(np.mean(gray)) < 85,
                "is_bright": float(np.mean(gray)) > 170,
                "is_low_contrast": float(np.std(gray)) < 40,
            }

            return result
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Histogram analysis error: {e}")

    def save_histogram_image(
        self, file_path: str | Path, output_path: str | Path, size: tuple[int, int] = (512, 400)
    ) -> ProcessingResult:
        """Saves the histogram plot as an image."""
        start = time.time()
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            hist_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, size[1] - 20, cv2.NORM_MINMAX)
                for j in range(1, 256):
                    x1 = int((j - 1) * size[0] / 256)
                    x2 = int(j * size[0] / 256)
                    y1 = size[1] - int(hist[j - 1])
                    y2 = size[1] - int(hist[j])
                    cv2.line(hist_img, (x1, y1), (x2, y2), color, 1)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), hist_img)

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Histogram image saved",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Histogram save error: {e}")

    def extract_color_palette(
        self, file_path: str | Path, num_colors: int = 8, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Extracts dominant colors from the image (K-means clustering).

        Args:
            num_colors: Number of colors to extract
            output_path: Optional path to save a palette preview image
        """
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            pixels = img.reshape(-1, 3).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
            _, labels, centers = cv2.kmeans(
                pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )

            centers = centers.astype(np.uint8)
            label_counts = np.bincount(labels.flatten())
            total_pixels = len(labels)

            colors = []
            for i in np.argsort(-label_counts):
                b, g, r = centers[i]
                colors.append({
                    "rgb": (int(r), int(g), int(b)),
                    "hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
                    "percentage": round(label_counts[i] / total_pixels * 100, 2),
                })

            if output_path:
                self._save_palette_image(colors, output_path)

            return {"colors": colors, "total_colors": num_colors}
        except Exception as e:
            raise ProcessingError(f"Color palette extraction error: {e}")

    def apply_color_map(
        self,
        input_path: str | Path,
        output_path: str | Path,
        colormap: str = "jet",
    ) -> ProcessingResult:
        """
        Applies a color map. For heat maps, depth visualization, etc.

        Args:
            colormap: OpenCV colormap name (jet, hot, cool, rainbow, ocean, bone, etc.)
        """
        start = time.time()
        colormaps = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "ocean": cv2.COLORMAP_OCEAN,
            "bone": cv2.COLORMAP_BONE,
            "spring": cv2.COLORMAP_SPRING,
            "summer": cv2.COLORMAP_SUMMER,
            "autumn": cv2.COLORMAP_AUTUMN,
            "winter": cv2.COLORMAP_WINTER,
            "twilight": cv2.COLORMAP_TWILIGHT,
            "turbo": cv2.COLORMAP_TURBO,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }

        try:
            img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ProcessingError(f"Could not read image: {input_path}")

            cmap = colormaps.get(colormap, cv2.COLORMAP_JET)
            colored = cv2.applyColorMap(img, cmap)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), colored)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Color map applied: {colormap}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Color map error: {e}")

    def create_hdr(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
        gamma: float = 1.0,
    ) -> ProcessingResult:
        """
        Creates HDR from multiple exposure-bracketed images.
        At least 3 different exposures are required.
        """
        start = time.time()
        try:
            images = [cv2.imread(str(p)) for p in image_paths]
            images = [img for img in images if img is not None]

            if len(images) < 2:
                raise ProcessingError("At least 2 images are required for HDR")

            exposure_times = np.array([1 / 30, 1 / 15, 1 / 8, 1 / 4, 1 / 2, 1], dtype=np.float32)
            exposure_times = exposure_times[:len(images)]

            merge = cv2.createMergeMertens()
            hdr = merge.process(images)
            hdr = np.clip(hdr * 255, 0, 255).astype(np.uint8)

            if gamma != 1.0:
                lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
                hdr = cv2.LUT(hdr, lookup)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), hdr)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"HDR created ({len(images)} images)",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"HDR creation error: {e}")

    def stitch_panorama(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
    ) -> ProcessingResult:
        """Stitches multiple images into a panorama."""
        start = time.time()
        try:
            images = [cv2.imread(str(p)) for p in image_paths]
            images = [img for img in images if img is not None]

            if len(images) < 2:
                raise ProcessingError("At least 2 images are required for a panorama")

            stitcher = cv2.Stitcher_create()
            status, panorama = stitcher.stitch(images)

            if status != cv2.Stitcher_OK:
                error_messages = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "More images required",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters could not be adjusted",
                }
                raise ProcessingError(f"Panorama error: {error_messages.get(status, 'Unknown error')}")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), panorama)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Panorama created ({len(images)} images)",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Panorama error: {e}")

    def detect_blur(self, file_path: str | Path) -> dict[str, Any]:
        """Analyzes blur level using Laplacian variance."""
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

            return {
                "laplacian_variance": round(laplacian_var, 2),
                "is_blurry": laplacian_var < 100,
                "quality": "sharp" if laplacian_var > 500 else "normal" if laplacian_var > 100 else "blurry",
            }
        except Exception as e:
            raise ProcessingError(f"Blur detection error: {e}")

    @staticmethod
    def _save_palette_image(
        colors: list[dict], output_path: str | Path, swatch_size: int = 80
    ) -> None:
        width = swatch_size * len(colors)
        height = swatch_size
        palette = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(palette)

        for i, color in enumerate(colors):
            x = i * swatch_size
            draw.rectangle([x, 0, x + swatch_size, height], fill=color["rgb"])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        palette.save(output_path)
