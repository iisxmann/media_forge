"""
Screen recording module.
Take screenshots and record screen video.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import StreamingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class ScreenRecorder:
    """Screen recording and screenshot class."""

    def capture_screenshot(
        self,
        output_path: str | Path,
        region: tuple[int, int, int, int] | None = None,
    ) -> ProcessingResult:
        """
        Captures a screenshot.

        Args:
            region: Capture region (x, y, width, height). None = full screen.
        """
        start = time.time()
        try:
            import mss

            with mss.mss() as sct:
                if region:
                    monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                else:
                    monitor = sct.monitors[1]

                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), img)

                return ProcessingResult(
                    success=True, output_path=output_path,
                    message=f"Screenshot captured: {img.shape[1]}x{img.shape[0]}",
                    duration_seconds=time.time() - start,
                )
        except ImportError:
            raise StreamingError("mss package required: pip install mss")
        except Exception as e:
            raise StreamingError(f"Screenshot error: {e}")

    def record_screen(
        self,
        output_path: str | Path,
        duration: float = 10.0,
        fps: int = 30,
        region: tuple[int, int, int, int] | None = None,
    ) -> ProcessingResult:
        """
        Records screen video.

        Args:
            duration: Recording duration (seconds)
            fps: Frame rate
            region: Capture region (None = full screen)
        """
        start = time.time()
        try:
            import mss

            with mss.mss() as sct:
                if region:
                    monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                else:
                    monitor = sct.monitors[1]

                width = monitor["width"]
                height = monitor["height"]

                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                frame_interval = 1.0 / fps
                frame_count = 0

                while time.time() - start < duration:
                    frame_start = time.time()
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    writer.write(frame)
                    frame_count += 1

                    elapsed = time.time() - frame_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)

                writer.release()

                return ProcessingResult(
                    success=True, output_path=output_path,
                    message=f"Screen recording complete: {frame_count} frames, {time.time() - start:.1f}s",
                    duration_seconds=time.time() - start,
                )
        except ImportError:
            raise StreamingError("mss package required: pip install mss")
        except Exception as e:
            raise StreamingError(f"Screen recording error: {e}")
