"""
Real-time video processing.
Apply filters and effects to a live video stream.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from mediaforge.core.exceptions import StreamingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class RealtimeProcessor:
    """
    Real-time video processing class.
    Reads frame by frame from a video source (file, webcam, RTSP) and processes it.
    """

    def __init__(self):
        self._filters: list[Callable[[np.ndarray], np.ndarray]] = []
        self._running = False

    def add_filter(self, filter_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Adds a filter to the processing pipeline."""
        self._filters.append(filter_func)

    def clear_filters(self) -> None:
        """Clears all filters."""
        self._filters.clear()

    def process_stream(
        self,
        source: str | int,
        output_path: str | Path | None = None,
        display: bool = True,
        fps: int | None = None,
        max_duration: float | None = None,
        on_frame: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Processes a video source in real time.

        Args:
            source: Video source (file path, webcam index, RTSP URL)
            output_path: Output file (None = do not save)
            display: Show on screen
            fps: Output FPS (None = source FPS)
            max_duration: Maximum processing time (seconds)
            on_frame: Callback for each frame
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise StreamingError(f"Could not open video source: {source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_fps = fps or int(src_fps)

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))

        self._running = True
        frame_count = 0
        start_time = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                for filter_func in self._filters:
                    frame = filter_func(frame)

                if on_frame:
                    on_frame(frame, frame_count)

                if writer:
                    writer.write(frame)

                if display:
                    cv2.imshow("MediaForge - Realtime", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

                if max_duration and (time.time() - start_time) >= max_duration:
                    break

        finally:
            self._running = False
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        return {
            "frames_processed": frame_count,
            "duration_seconds": round(elapsed, 2),
            "avg_fps": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
            "output": str(output_path) if output_path else None,
        }

    def stop(self) -> None:
        """Stops processing."""
        self._running = False

    @staticmethod
    def grayscale_filter(frame: np.ndarray) -> np.ndarray:
        """Grayscale filter."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_filter(frame: np.ndarray) -> np.ndarray:
        """Edge detection filter."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur_filter(radius: int = 15) -> Callable:
        """Creates a blur filter."""
        def _filter(frame: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(frame, (radius, radius), 0)
        return _filter

    @staticmethod
    def mirror_filter(frame: np.ndarray) -> np.ndarray:
        """Mirror effect."""
        return cv2.flip(frame, 1)

    @staticmethod
    def cartoon_filter(frame: np.ndarray) -> np.ndarray:
        """Cartoon effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    @staticmethod
    def thermal_filter(frame: np.ndarray) -> np.ndarray:
        """Thermal camera effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def negative_filter(frame: np.ndarray) -> np.ndarray:
        """Negative effect."""
        return cv2.bitwise_not(frame)
