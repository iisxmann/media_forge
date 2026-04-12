"""
Face detection and recognition module.
Haar Cascade, DNN-based face detection, face blurring, face cropping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import AIModelError, ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceRegion:
    """Represents a detected face region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "confidence": round(self.confidence, 4),
        }


class FaceDetector:
    """Face detection and processing class."""

    def __init__(self, method: str = "haar", confidence_threshold: float = 0.5):
        """
        Args:
            method: Detection method ('haar', 'dnn')
            confidence_threshold: Minimum confidence threshold
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        if self.method == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)
            if self._detector.empty():
                raise AIModelError("Haar cascade file could not be loaded")
        elif self.method == "dnn":
            try:
                model_path = cv2.data.haarcascades + "../res10_300x300_ssd_iter_140000.caffemodel"
                config_path = cv2.data.haarcascades + "../deploy.prototxt"
                self._detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
            except Exception:
                logger.warning("DNN model not found, falling back to Haar cascade")
                self.method = "haar"
                self._init_detector()

    def detect_faces(self, file_path: str | Path) -> list[FaceRegion]:
        """Detects faces in an image."""
        img = cv2.imread(str(file_path))
        if img is None:
            raise ProcessingError(f"Could not read image: {file_path}")

        if self.method == "haar":
            return self._detect_haar(img)
        elif self.method == "dnn":
            return self._detect_dnn(img)
        return []

    def draw_faces(
        self,
        input_path: str | Path,
        output_path: str | Path,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
    ) -> ProcessingResult:
        """Draws rectangles around detected faces."""
        start = time.time()
        img = cv2.imread(str(input_path))
        faces = self.detect_faces(input_path)

        for face in faces:
            cv2.rectangle(
                img,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color, thickness,
            )
            if show_confidence and face.confidence > 0:
                label = f"{face.confidence:.1%}"
                cv2.putText(
                    img, label,
                    (face.x, face.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"{len(faces)} face(s) detected and marked",
            duration_seconds=time.time() - start,
            details={"faces": [f.to_dict() for f in faces]},
        )

    def blur_faces(
        self,
        input_path: str | Path,
        output_path: str | Path,
        blur_strength: int = 51,
    ) -> ProcessingResult:
        """Blurs detected faces. For privacy protection."""
        start = time.time()
        img = cv2.imread(str(input_path))
        faces = self.detect_faces(input_path)

        for face in faces:
            roi = img[face.y:face.y + face.height, face.x:face.x + face.width]
            blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            img[face.y:face.y + face.height, face.x:face.x + face.width] = blurred

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"{len(faces)} face(s) blurred",
            duration_seconds=time.time() - start,
        )

    def crop_faces(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        padding: float = 0.2,
    ) -> list[ProcessingResult]:
        """Crops each detected face to a separate file."""
        img = cv2.imread(str(input_path))
        faces = self.detect_faces(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        results = []
        h, w = img.shape[:2]

        for i, face in enumerate(faces):
            pad_x = int(face.width * padding)
            pad_y = int(face.height * padding)

            x1 = max(0, face.x - pad_x)
            y1 = max(0, face.y - pad_y)
            x2 = min(w, face.x + face.width + pad_x)
            y2 = min(h, face.y + face.height + pad_y)

            cropped = img[y1:y2, x1:x2]
            out_path = output_dir / f"{stem}_face_{i}.jpg"
            cv2.imwrite(str(out_path), cropped)

            results.append(ProcessingResult(
                success=True, output_path=out_path,
                message=f"Face {i} cropped",
            ))

        return results

    def count_faces(self, file_path: str | Path) -> int:
        """Returns the number of faces in the image."""
        return len(self.detect_faces(file_path))

    def blur_faces_in_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        blur_strength: int = 51,
    ) -> ProcessingResult:
        """Blurs all faces in a video."""
        start = time.time()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.method == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                for (x, y, fw, fh) in rects:
                    roi = frame[y:y+fh, x:x+fw]
                    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                    frame[y:y+fh, x:x+fw] = blurred

            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Video face blur complete ({frame_count} frames)",
            duration_seconds=time.time() - start,
        )

    def _detect_haar(self, img: np.ndarray) -> list[FaceRegion]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [FaceRegion(x=int(x), y=int(y), width=int(w), height=int(h)) for (x, y, w, h) in rects]

    def _detect_dnn(self, img: np.ndarray) -> list[FaceRegion]:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
        self._detector.setInput(blob)
        detections = self._detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            faces.append(FaceRegion(
                x=max(0, x1), y=max(0, y1),
                width=x2 - x1, height=y2 - y1,
                confidence=float(confidence),
            ))

        return faces
