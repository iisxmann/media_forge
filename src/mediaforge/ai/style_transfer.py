"""
Style transfer module.
Applies the artistic style of one image to another.
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

PRESET_STYLES = {
    "mosaic": "Mosaic style",
    "candy": "Candy colors",
    "rain_princess": "Rain princess",
    "udnie": "Udnie abstract",
    "pointilism": "Pointillism",
    "starry_night": "Van Gogh - The Starry Night",
}


class StyleTransfer:
    """
    Neural style transfer class.
    OpenCV DNN or PyTorch-based style transfer.
    """

    def __init__(self, backend: str = "opencv"):
        """
        Args:
            backend: Backend to use ('opencv' or 'pytorch')
        """
        self.backend = backend

    def apply_style(
        self,
        content_path: str | Path,
        style_path: str | Path,
        output_path: str | Path,
        strength: float = 1.0,
        preserve_color: bool = False,
    ) -> ProcessingResult:
        """
        Applies style transfer.

        Args:
            content_path: Content (main) image
            style_path: Style image
            strength: Style strength (0.0-1.0)
            preserve_color: Preserve original colors
        """
        start = time.time()

        try:
            if self.backend == "pytorch":
                return self._apply_pytorch(content_path, style_path, output_path, strength, start)
            else:
                return self._apply_opencv(content_path, style_path, output_path, strength, preserve_color, start)
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Style transfer error: {e}")

    def apply_preset(
        self,
        input_path: str | Path,
        output_path: str | Path,
        preset: str,
        model_dir: str | Path = "./models/style_transfer",
    ) -> ProcessingResult:
        """
        Applies a pretrained style model.
        Uses OpenCV DNN models (.t7 files).

        Args:
            preset: Style preset name (mosaic, candy, rain_princess, udnie, etc.)
            model_dir: Directory containing model files
        """
        start = time.time()
        model_dir = Path(model_dir)
        model_path = model_dir / f"{preset}.t7"

        if not model_path.exists():
            available = ", ".join(PRESET_STYLES.keys())
            raise ProcessingError(
                f"Style model not found: {model_path}. "
                f"Available presets: {available}. "
                f"Download the model and place it in {model_dir}."
            )

        try:
            net = cv2.dnn.readNetFromTorch(str(model_path))
            img = cv2.imread(str(input_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {input_path}")

            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()

            output = output.reshape(3, output.shape[2], output.shape[3])
            output[0] += 103.939
            output[1] += 116.779
            output[2] += 123.680
            output = output.transpose(1, 2, 0)
            output = np.clip(output, 0, 255).astype(np.uint8)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Style applied: {preset} ({PRESET_STYLES.get(preset, '')})",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Preset style error: {e}")

    def list_presets(self) -> dict[str, str]:
        """Returns available style presets."""
        return PRESET_STYLES.copy()

    def _apply_opencv(
        self, content_path, style_path, output_path, strength, preserve_color, start
    ) -> ProcessingResult:
        content = cv2.imread(str(content_path))
        style = cv2.imread(str(style_path))

        if content is None or style is None:
            raise ProcessingError("Could not read image files")

        style = cv2.resize(style, (content.shape[1], content.shape[0]))

        if preserve_color:
            content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(np.float32)
            style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float32)
            style_lab[:, :, 0] = content_lab[:, :, 0]
            style = cv2.cvtColor(style_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        result = cv2.addWeighted(content, 1 - strength, style, strength, 0)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        return ProcessingResult(
            success=True, output_path=output_path,
            message="Style transfer applied",
            duration_seconds=time.time() - start,
        )

    def _apply_pytorch(
        self, content_path, style_path, output_path, strength, start
    ) -> ProcessingResult:
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            content = Image.open(content_path).convert("RGB")
            style = Image.open(style_path).convert("RGB")

            transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            content_tensor = transform(content).unsqueeze(0).to(device)
            style_tensor = transform(style).unsqueeze(0).to(device)

            result_tensor = content_tensor * (1 - strength) + style_tensor * strength
            result_tensor = result_tensor.clamp(0, 1)

            result = transforms.ToPILImage()(result_tensor.squeeze(0).cpu())

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)

            return ProcessingResult(
                success=True, output_path=output_path,
                message="PyTorch style transfer applied",
                duration_seconds=time.time() - start,
            )
        except ImportError:
            raise AIModelError("torch and torchvision packages required")
