"""
Audio analysis module.
Waveform, spectrogram, loudness analysis, BPM detection.
"""

from __future__ import annotations

import subprocess
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class AudioAnalyzer:
    """Audio analysis class."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def analyze_loudness(self, file_path: str | Path) -> dict[str, Any]:
        """Runs EBU R128 loudness analysis."""
        try:
            cmd = [
                self.ffmpeg, "-i", str(file_path),
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            json_start = result.stderr.rfind("{")
            json_end = result.stderr.rfind("}") + 1
            if json_start >= 0:
                loudness_data = json.loads(result.stderr[json_start:json_end])
                return {
                    "integrated_loudness_lufs": float(loudness_data.get("input_i", 0)),
                    "true_peak_dbtp": float(loudness_data.get("input_tp", 0)),
                    "loudness_range_lu": float(loudness_data.get("input_lra", 0)),
                    "threshold_lufs": float(loudness_data.get("input_thresh", 0)),
                }
            return {}
        except Exception as e:
            raise ProcessingError(f"Audio analysis error: {e}")

    def detect_silence(
        self,
        file_path: str | Path,
        threshold: float = -50,
        min_duration: float = 1.0,
    ) -> list[dict[str, float]]:
        """
        Detects silent segments.

        Args:
            threshold: Silence threshold (dB)
            min_duration: Minimum silence duration (seconds)
        """
        try:
            cmd = [
                self.ffmpeg, "-i", str(file_path),
                "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            silences = []
            current = {}
            for line in result.stderr.split("\n"):
                if "silence_start" in line:
                    parts = line.split("silence_start: ")
                    if len(parts) > 1:
                        current["start"] = float(parts[1].strip())
                elif "silence_end" in line and current:
                    parts = line.split("silence_end: ")
                    if len(parts) > 1:
                        end_parts = parts[1].strip().split("|")
                        current["end"] = float(end_parts[0].strip())
                        if len(end_parts) > 1 and "silence_duration" in end_parts[1]:
                            dur = end_parts[1].split(":")[1].strip()
                            current["duration"] = float(dur)
                        silences.append(current)
                        current = {}

            return silences
        except Exception as e:
            raise ProcessingError(f"Silence detection error: {e}")

    def generate_waveform(
        self,
        file_path: str | Path,
        output_path: str | Path,
        width: int = 1920,
        height: int = 200,
        color: str = "0x00ff00",
        bg_color: str = "0x000000",
    ) -> ProcessingResult:
        """
        Generates a waveform image.

        Args:
            width, height: Image dimensions
            color: Waveform color (hex)
            bg_color: Background color (hex)
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y", "-i", str(file_path),
            "-filter_complex",
            f"showwavespic=s={width}x{height}:colors={color}",
            "-frames:v", "1",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Waveform error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Waveform generated",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Waveform error: {e}")

    def generate_spectrogram(
        self,
        file_path: str | Path,
        output_path: str | Path,
        width: int = 1920,
        height: int = 512,
    ) -> ProcessingResult:
        """Generates a spectrogram image."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y", "-i", str(file_path),
            "-lavfi", f"showspectrumpic=s={width}x{height}:mode=combined:color=intensity",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Spectrogram error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Spectrogram generated",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Spectrogram error: {e}")

    def detect_bpm(self, file_path: str | Path) -> dict[str, Any]:
        """
        Detects BPM (beats per minute).
        Requires the librosa library.
        """
        try:
            import librosa

            y, sr = librosa.load(str(file_path))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            return {
                "bpm": round(float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo), 1),
                "sample_rate": sr,
                "duration": round(len(y) / sr, 2),
            }
        except ImportError:
            raise ProcessingError("librosa is required: pip install librosa")
        except Exception as e:
            raise ProcessingError(f"BPM detection error: {e}")
