"""
Webcam capture module.
Take photos from the webcam, record video, create timelapses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import StreamingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class WebcamCapture:
    """Webcam capture and recording class."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

    def capture_photo(
        self,
        output_path: str | Path,
        width: int | None = None,
        height: int | None = None,
    ) -> ProcessingResult:
        """Captures a single photo from the webcam."""
        start = time.time()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise StreamingError(f"Could not open webcam (index: {self.camera_index})")

        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise StreamingError("Could not read frame from webcam")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Photo captured: {frame.shape[1]}x{frame.shape[0]}",
            duration_seconds=time.time() - start,
        )

    def record_video(
        self,
        output_path: str | Path,
        duration: float = 10.0,
        fps: int = 30,
        display: bool = False,
    ) -> ProcessingResult:
        """
        Records video from the webcam.

        Args:
            duration: Recording duration (seconds)
            fps: Frame rate
            display: Show preview while recording
        """
        start = time.time()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise StreamingError(f"Could not open webcam (index: {self.camera_index})")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        try:
            while time.time() - start < duration:
                ret, frame = cap.read()
                if not ret:
                    break

                writer.write(frame)
                frame_count += 1

                if display:
                    cv2.imshow("Recording...", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Video recorded: {frame_count} frames, {time.time() - start:.1f}s",
            duration_seconds=time.time() - start,
        )

    def create_timelapse(
        self,
        output_path: str | Path,
        total_duration: float = 60.0,
        capture_interval: float = 2.0,
        output_fps: int = 30,
    ) -> ProcessingResult:
        """
        Creates a timelapse from the webcam.

        Args:
            total_duration: Total capture time (seconds)
            capture_interval: Delay between frames (seconds)
            output_fps: Output video FPS
        """
        start = time.time()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise StreamingError(f"Could not open webcam (index: {self.camera_index})")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

        frame_count = 0
        try:
            while time.time() - start < total_duration:
                ret, frame = cap.read()
                if ret:
                    writer.write(frame)
                    frame_count += 1
                time.sleep(capture_interval)
        finally:
            cap.release()
            writer.release()

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Timelapse created: {frame_count} frames",
            duration_seconds=time.time() - start,
        )

    def list_cameras(self) -> list[dict[str, Any]]:
        """Lists available cameras."""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    "index": i,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                })
                cap.release()
        return cameras
