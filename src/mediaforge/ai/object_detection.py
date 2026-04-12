"""
Object detection module.
YOLOv8-based object detection, classification, counting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import AIModelError, ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": self.bbox[0], "y1": self.bbox[1], "x2": self.bbox[2], "y2": self.bbox[3]},
        }


class ObjectDetector:
    """YOLOv8-based object detection class."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ):
        """
        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum confidence threshold
            device: Compute device (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self._model = YOLO(f"{self.model_name}.pt")
            logger.info("YOLO model loaded")
        except ImportError:
            raise AIModelError("ultralytics package required: pip install ultralytics")
        except Exception as e:
            raise AIModelError(f"YOLO model load error: {e}")

    def detect(
        self, file_path: str | Path, classes: list[str] | None = None
    ) -> list[Detection]:
        """
        Detects objects in an image.

        Args:
            classes: Class names to filter (None=all)
        """
        self._load_model()

        try:
            results = self._model(str(file_path), conf=self.confidence_threshold, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                    if classes and cls_name not in classes:
                        continue

                    detections.append(Detection(
                        class_id=cls_id, class_name=cls_name,
                        confidence=conf, bbox=(x1, y1, x2, y2),
                    ))

            return detections
        except Exception as e:
            raise ProcessingError(f"Object detection error: {e}")

    def detect_and_draw(
        self,
        input_path: str | Path,
        output_path: str | Path,
        classes: list[str] | None = None,
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> ProcessingResult:
        """Detects objects and draws boxes on the image."""
        start = time.time()
        self._load_model()

        try:
            results = self._model(str(input_path), conf=self.confidence_threshold, verbose=False)
            annotated = results[0].plot()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

            detections = self.detect(input_path, classes)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"{len(detections)} object(s) detected",
                duration_seconds=time.time() - start,
                details={"detections": [d.to_dict() for d in detections]},
            )
        except Exception as e:
            raise ProcessingError(f"Detect and draw error: {e}")

    def count_objects(
        self, file_path: str | Path, classes: list[str] | None = None
    ) -> dict[str, int]:
        """Counts objects in the image by class."""
        detections = self.detect(file_path, classes)
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def detect_in_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        classes: list[str] | None = None,
        process_every_n: int = 1,
    ) -> ProcessingResult:
        """
        Detects objects in a video and writes an annotated video.

        Args:
            process_every_n: Process every Nth frame (for performance)
        """
        start = time.time()
        self._load_model()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_count = 0
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % process_every_n == 0:
                    results = self._model(frame, conf=self.confidence_threshold, verbose=False)
                    last_annotated = results[0].plot()
                    writer.write(last_annotated)
                elif last_annotated is not None:
                    writer.write(frame)
                else:
                    writer.write(frame)

                frame_count += 1

            cap.release()
            writer.release()

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Video object detection complete ({frame_count} frames)",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            cap.release()
            writer.release()
            raise ProcessingError(f"Video object detection error: {e}")

    def get_available_classes(self) -> list[str]:
        """Returns object class names the model can detect."""
        self._load_model()
        return list(self._model.names.values())
