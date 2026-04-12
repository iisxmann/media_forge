"""
Super-resolution (image upscaling) module.
AI-based image quality improvement and enlargement.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import AIModelError, ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class SuperResolution:
    """
    AI-based super-resolution class.
    Uses OpenCV DNN Super Resolution or Real-ESRGAN-style models.
    """

    def __init__(self, method: str = "opencv", scale: int = 4):
        """
        Args:
            method: Upscaling method ('opencv', 'lanczos', 'cubic')
            scale: Scale factor (2, 3, 4, 8)
        """
        self.method = method
        self.scale = scale

    def upscale(
        self,
        input_path: str | Path,
        output_path: str | Path,
        scale: int | None = None,
    ) -> ProcessingResult:
        """
        Upscales the image and improves quality.

        Args:
            scale: Scale factor (uses default if omitted)
        """
        start = time.time()
        scale = scale or self.scale
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = cv2.imread(str(input_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {input_path}")

            h, w = img.shape[:2]
            new_w, new_h = w * scale, h * scale

            if self.method == "opencv":
                result = self._upscale_opencv_sr(img, scale)
            elif self.method == "lanczos":
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            elif self.method == "cubic":
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            result = self._post_process(result)

            cv2.imwrite(str(output_path), result)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Upscaled: {w}x{h} -> {result.shape[1]}x{result.shape[0]} ({scale}x)",
                duration_seconds=time.time() - start,
                details={
                    "original_size": f"{w}x{h}",
                    "new_size": f"{result.shape[1]}x{result.shape[0]}",
                    "scale": scale,
                    "method": self.method,
                },
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Super-resolution error: {e}")

    def upscale_with_enhancement(
        self,
        input_path: str | Path,
        output_path: str | Path,
        scale: int | None = None,
        denoise: bool = True,
        sharpen: bool = True,
    ) -> ProcessingResult:
        """
        Upscaling + denoising + sharpening.
        Combined pipeline for best results.
        """
        start = time.time()
        scale = scale or self.scale

        try:
            img = cv2.imread(str(input_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {input_path}")

            h, w = img.shape[:2]
            new_w, new_h = w * scale, h * scale

            if denoise:
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            if sharpen:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                result = cv2.filter2D(result, -1, kernel)

            result = self._post_process(result)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Enhanced upscale: {w}x{h} -> {new_w}x{new_h}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Enhanced upscale error: {e}")

    def batch_upscale(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        scale: int | None = None,
    ) -> list[ProcessingResult]:
        """Batch-upscales multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scale = scale or self.scale

        results = []
        for path in input_paths:
            path = Path(path)
            output_path = output_dir / f"{path.stem}_{scale}x{path.suffix}"
            try:
                result = self.upscale(path, output_path, scale)
                results.append(result)
            except Exception as e:
                results.append(ProcessingResult(
                    success=False, message=f"Error: {path.name}: {e}"
                ))

        return results

    def _upscale_opencv_sr(self, img: np.ndarray, scale: int) -> np.ndarray:
        """Upscaling using OpenCV DNN Super Resolution."""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            model_map = {
                2: ("EDSR", "EDSR_x2.pb"),
                3: ("EDSR", "EDSR_x3.pb"),
                4: ("EDSR", "EDSR_x4.pb"),
            }

            if scale in model_map:
                model_name, model_file = model_map[scale]
                model_path = Path(f"./models/super_resolution/{model_file}")

                if model_path.exists():
                    sr.readModel(str(model_path))
                    sr.setModel(model_name.lower(), scale)
                    return sr.upsample(img)

            logger.warning("SR model not found, using Lanczos interpolation")
            h, w = img.shape[:2]
            return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        except AttributeError:
            h, w = img.shape[:2]
            return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _post_process(img: np.ndarray) -> np.ndarray:
        """Fine-tuning after upscale: contrast and color correction."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
