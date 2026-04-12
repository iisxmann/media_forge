"""
OCR (optical character recognition) module.
Text extraction from images; Tesseract and EasyOCR support.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import OCRError, ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """Text extraction from images."""

    def __init__(self, engine: str = "tesseract", languages: list[str] | None = None):
        """
        Args:
            engine: OCR engine ('tesseract' or 'easyocr')
            languages: Language list (e.g. ['tr', 'en'])
        """
        self.engine = engine
        self.languages = languages or ["tr", "en"]

    def extract_text(
        self,
        file_path: str | Path,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        """
        Extracts text from an image.

        Args:
            preprocess: Image preprocessing (contrast boost, denoising)
        """
        start = time.time()

        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            if preprocess:
                img = self._preprocess(img)

            if self.engine == "tesseract":
                result = self._extract_tesseract(img)
            elif self.engine == "easyocr":
                result = self._extract_easyocr(str(file_path))
            else:
                raise OCRError(f"Unknown OCR engine: {self.engine}")

            result["duration_seconds"] = round(time.time() - start, 2)
            return result
        except (OCRError, ProcessingError):
            raise
        except Exception as e:
            raise OCRError(f"OCR error: {e}")

    def extract_text_with_boxes(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extracts text and marks detected text regions."""
        start = time.time()

        try:
            if self.engine == "easyocr":
                return self._extract_easyocr_with_boxes(file_path, output_path, start)
            else:
                return self._extract_tesseract_with_boxes(file_path, output_path, start)
        except Exception as e:
            raise OCRError(f"OCR with boxes error: {e}")

    def extract_from_region(
        self,
        file_path: str | Path,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> str:
        """Extracts text from a specific region of the image."""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {file_path}")

            roi = img[y:y + height, x:x + width]
            roi = self._preprocess(roi)

            if self.engine == "tesseract":
                import pytesseract
                lang = "+".join(self.languages)
                return pytesseract.image_to_string(roi, lang=lang).strip()
            elif self.engine == "easyocr":
                import easyocr
                reader = easyocr.Reader(self.languages)
                results = reader.readtext(roi)
                return " ".join([r[1] for r in results])
            return ""
        except Exception as e:
            raise OCRError(f"Region OCR error: {e}")

    def batch_extract(
        self, file_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        """Batch text extraction from multiple images."""
        results = []
        for path in file_paths:
            try:
                result = self.extract_text(path)
                result["file"] = str(path)
                results.append(result)
            except Exception as e:
                results.append({"file": str(path), "error": str(e), "text": ""})
        return results

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Image preprocessing for OCR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        return gray

    def _extract_tesseract(self, img: np.ndarray) -> dict[str, Any]:
        try:
            import pytesseract

            lang = "+".join(self.languages)
            text = pytesseract.image_to_string(img, lang=lang)
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

            words = [
                {"text": w, "confidence": c}
                for w, c in zip(data["text"], data["conf"])
                if w.strip() and c > 0
            ]

            return {
                "text": text.strip(),
                "engine": "tesseract",
                "word_count": len(words),
                "avg_confidence": round(np.mean([w["confidence"] for w in words]), 2) if words else 0,
            }
        except ImportError:
            raise OCRError("pytesseract package required: pip install pytesseract")

    def _extract_easyocr(self, file_path: str) -> dict[str, Any]:
        try:
            import easyocr

            reader = easyocr.Reader(self.languages)
            results = reader.readtext(file_path)

            text = " ".join([r[1] for r in results])
            words = [{"text": r[1], "confidence": round(r[2], 4)} for r in results]

            return {
                "text": text,
                "engine": "easyocr",
                "word_count": len(words),
                "avg_confidence": round(np.mean([w["confidence"] for w in words]), 2) if words else 0,
                "detections": words,
            }
        except ImportError:
            raise OCRError("easyocr package required: pip install easyocr")

    def _extract_tesseract_with_boxes(self, file_path, output_path, start) -> dict[str, Any]:
        import pytesseract

        img = cv2.imread(str(file_path))
        lang = "+".join(self.languages)
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

        boxes = []
        for i in range(len(data["text"])):
            if data["text"][i].strip() and int(data["conf"][i]) > 30:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                boxes.append({"text": data["text"][i], "x": x, "y": y, "w": w, "h": h, "confidence": data["conf"][i]})
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)

        return {"boxes": boxes, "total": len(boxes), "duration_seconds": round(time.time() - start, 2)}

    def _extract_easyocr_with_boxes(self, file_path, output_path, start) -> dict[str, Any]:
        import easyocr

        reader = easyocr.Reader(self.languages)
        results = reader.readtext(str(file_path))

        img = cv2.imread(str(file_path))
        boxes = []
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            boxes.append({"text": text, "confidence": round(conf, 4), "bbox": bbox})

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)

        return {"boxes": boxes, "total": len(boxes), "duration_seconds": round(time.time() - start, 2)}
