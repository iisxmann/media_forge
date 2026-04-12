"""
Video thumbnail extractor.
Extract frames at a given time, sprite sheets, animated thumbnails.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import cv2

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class VideoThumbnailExtractor:
    """Video thumbnail and frame extraction class."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def extract_frame(
        self,
        video_path: str | Path,
        output_path: str | Path,
        timestamp: float = 0,
        quality: int = 95,
    ) -> ProcessingResult:
        """Extracts a single frame at the given timestamp."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y", "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", str(max(1, min(31, 31 - int(quality * 0.3)))),
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Frame extraction error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Frame extracted @ {timestamp}s",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Frame extract error: {e}")

    def extract_frames_at_intervals(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        interval: float = 5.0,
        max_frames: int = 50,
    ) -> list[ProcessingResult]:
        """Extracts frames at regular intervals."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            results = []
            t = 0.0
            idx = 0

            while t < duration and idx < max_frames:
                output_path = output_dir / f"{stem}_frame_{idx:03d}_{t:.1f}s.jpg"
                result = self.extract_frame(video_path, output_path, timestamp=t)
                results.append(result)
                t += interval
                idx += 1

            return results
        except Exception as e:
            raise ProcessingError(f"Interval frame extract error: {e}")

    def create_sprite_sheet(
        self,
        video_path: str | Path,
        output_path: str | Path,
        columns: int = 5,
        frame_count: int = 20,
        frame_width: int = 320,
    ) -> ProcessingResult:
        """
        Creates a video sprite sheet.
        Used for player previews (YouTube-style).

        Args:
            columns: Number of columns in the sprite sheet
            frame_count: Total number of frames
            frame_width: Width of each frame in pixels
        """
        start = time.time()
        from PIL import Image

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_height = int(frame_width * orig_h / orig_w)

            interval = max(1, total_frames // frame_count)
            rows = math.ceil(frame_count / columns)

            sheet = Image.new("RGB", (columns * frame_width, rows * frame_height), (0, 0, 0))
            extracted = 0

            for i in range(frame_count):
                frame_idx = i * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame).resize((frame_width, frame_height), Image.LANCZOS)

                col = extracted % columns
                row = extracted // columns
                sheet.paste(pil_frame, (col * frame_width, row * frame_height))
                extracted += 1

            cap.release()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output_path, quality=90)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Sprite sheet created: {extracted} frames ({columns}x{rows})",
                duration_seconds=time.time() - start,
                details={"frame_width": frame_width, "frame_height": frame_height, "columns": columns, "rows": rows},
            )
        except Exception as e:
            raise ProcessingError(f"Sprite sheet error: {e}")

    def create_animated_thumbnail(
        self,
        video_path: str | Path,
        output_path: str | Path,
        start_time: float = 0,
        duration: float = 3,
        fps: int = 10,
        width: int = 320,
    ) -> ProcessingResult:
        """Creates an animated GIF/WebP thumbnail."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()

        if ext == ".gif":
            from mediaforge.video.processor import VideoProcessor
            vp = VideoProcessor(ffmpeg_path=self.ffmpeg)
            return vp.create_gif(video_path, output_path, start_time, duration, fps, width)
        elif ext == ".webp":
            cmd = [
                self.ffmpeg, "-y",
                "-ss", str(start_time), "-t", str(duration),
                "-i", str(video_path),
                "-vf", f"fps={fps},scale={width}:-1",
                "-loop", "0", "-quality", "70",
                str(output_path),
            ]
            try:
                process = subprocess.run(cmd, capture_output=True, text=True)
                if process.returncode != 0:
                    raise ProcessingError(f"WebP thumbnail error: {process.stderr[:500]}")
                return ProcessingResult(
                    success=True, output_path=output_path,
                    message="Animated WebP thumbnail created",
                    duration_seconds=time.time() - start,
                )
            except Exception as e:
                raise ProcessingError(f"Animated thumbnail error: {e}")
        else:
            raise ProcessingError(f"Unsupported format: {ext}. Use .gif or .webp")

    def extract_best_thumbnail(
        self,
        video_path: str | Path,
        output_path: str | Path,
        sample_count: int = 10,
    ) -> ProcessingResult:
        """
        Picks the best frame for a thumbnail.
        Prefers sharp, high-contrast frames.
        """
        start = time.time()
        import numpy as np

        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total_frames // sample_count)

            best_frame = None
            best_score = -1

            for i in range(sample_count):
                frame_idx = i * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                contrast = gray.std()
                score = sharpness * 0.7 + contrast * 0.3

                if score > best_score:
                    best_score = score
                    best_frame = frame

            cap.release()

            if best_frame is None:
                raise ProcessingError("Could not extract a frame from the video")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), best_frame)

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Best thumbnail selected (score: {best_score:.1f})",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Best thumbnail error: {e}")
