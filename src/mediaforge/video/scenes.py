"""
Scene detection module.
Detects scene changes within a video.
"""

from __future__ import annotations

import subprocess
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Scene:
    """Represents one scene."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    start_frame: int
    end_frame: int
    score: float       # scene change score

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SceneDetector:
    """Video scene detection class."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def detect_scenes(
        self,
        video_path: str | Path,
        threshold: float = 30.0,
        min_scene_duration: float = 1.0,
    ) -> list[Scene]:
        """
        Detects scene changes.

        Args:
            threshold: Detection threshold (lower = more sensitive, more scenes; higher = fewer scenes)
            min_scene_duration: Minimum scene duration in seconds
        """
        path = Path(video_path)
        if not path.exists():
            raise ProcessingError(f"Video not found: {path}")

        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            min_frames = int(min_scene_duration * fps)

            scenes = []
            prev_frame = None
            scene_start_frame = 0
            scene_start_time = 0.0
            frame_idx = 0
            scene_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    score = float(np.mean(diff))

                    if score > threshold and (frame_idx - scene_start_frame) >= min_frames:
                        current_time = frame_idx / fps
                        scenes.append(Scene(
                            index=scene_idx,
                            start_time=round(scene_start_time, 3),
                            end_time=round(current_time, 3),
                            start_frame=scene_start_frame,
                            end_frame=frame_idx,
                            score=round(score, 2),
                        ))
                        scene_idx += 1
                        scene_start_frame = frame_idx
                        scene_start_time = current_time

                prev_frame = gray
                frame_idx += 1

            if scene_start_frame < frame_idx:
                scenes.append(Scene(
                    index=scene_idx,
                    start_time=round(scene_start_time, 3),
                    end_time=round(frame_idx / fps, 3),
                    start_frame=scene_start_frame,
                    end_frame=frame_idx,
                    score=0.0,
                ))

            cap.release()
            logger.info(f"Detected {len(scenes)} scenes: {path.name}")
            return scenes
        except Exception as e:
            raise ProcessingError(f"Scene detection error: {e}")

    def split_by_scenes(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        threshold: float = 30.0,
        min_scene_duration: float = 1.0,
    ) -> list[ProcessingResult]:
        """Splits the video by detected scenes and saves separate files."""
        scenes = self.detect_scenes(video_path, threshold, min_scene_duration)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem

        results = []
        for scene in scenes:
            output_path = output_dir / f"{stem}_scene_{scene.index:03d}.mp4"
            cmd = [
                self.ffmpeg, "-y", "-i", str(video_path),
                "-ss", str(scene.start_time),
                "-to", str(scene.end_time),
                "-c", "copy",
                str(output_path),
            ]

            try:
                process = subprocess.run(cmd, capture_output=True, text=True)
                if process.returncode == 0:
                    results.append(ProcessingResult(
                        success=True, output_path=output_path,
                        message=f"Scene {scene.index}: {scene.start_time}s-{scene.end_time}s",
                    ))
                else:
                    results.append(ProcessingResult(
                        success=False,
                        message=f"Scene {scene.index} could not be saved",
                    ))
            except Exception as e:
                results.append(ProcessingResult(success=False, message=str(e)))

        return results

    def extract_keyframes(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        max_frames: int = 20,
    ) -> list[ProcessingResult]:
        """Extracts keyframes (I-frames) from the video."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            interval = max(1, total_frames // max_frames)
            results = []
            frame_idx = 0
            saved = 0

            while saved < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                output_path = output_dir / f"{stem}_kf_{saved:03d}_{timestamp:.1f}s.jpg"
                cv2.imwrite(str(output_path), frame)

                results.append(ProcessingResult(
                    success=True, output_path=output_path,
                    message=f"Keyframe {saved} @ {timestamp:.1f}s",
                ))

                saved += 1
                frame_idx += interval

            cap.release()
            return results
        except Exception as e:
            raise ProcessingError(f"Keyframe extraction error: {e}")

    def get_scene_summary(self, scenes: list[Scene]) -> dict:
        """Scene analysis summary."""
        if not scenes:
            return {"total_scenes": 0}

        durations = [s.duration for s in scenes]
        return {
            "total_scenes": len(scenes),
            "total_duration": round(sum(durations), 2),
            "avg_scene_duration": round(sum(durations) / len(durations), 2),
            "min_scene_duration": round(min(durations), 2),
            "max_scene_duration": round(max(durations), 2),
            "short_scenes": sum(1 for d in durations if d < 2),
            "long_scenes": sum(1 for d in durations if d > 10),
        }
