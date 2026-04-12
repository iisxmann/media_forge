"""
Audio mixer.
Concatenate multiple files, overlay, crossfade.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class AudioMixer:
    """Mixes and concatenates audio files."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def concatenate(
        self,
        input_paths: list[str | Path],
        output_path: str | Path,
    ) -> ProcessingResult:
        """Concatenates audio files sequentially."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_file = output_path.parent / "_audio_concat_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in input_paths:
                f.write(f"file '{Path(p).absolute()}'\n")

        cmd = [
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            list_file.unlink(missing_ok=True)

            if process.returncode != 0:
                raise ProcessingError(f"Concatenation error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Merged {len(input_paths)} audio files",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            list_file.unlink(missing_ok=True)
            raise ProcessingError(f"Concatenate error: {e}")

    def mix(
        self,
        input_paths: list[str | Path],
        output_path: str | Path,
        volumes: list[float] | None = None,
        duration_mode: str = "first",
    ) -> ProcessingResult:
        """
        Mixes multiple audio files together.

        Args:
            volumes: Per-file volume (0.0-2.0)
            duration_mode: Duration mode ('first', 'longest', 'shortest')
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y"]
        for p in input_paths:
            cmd.extend(["-i", str(p)])

        n = len(input_paths)
        volume_parts = []

        if volumes:
            for i, vol in enumerate(volumes):
                volume_parts.append(f"[{i}:a]volume={vol}[a{i}]")
            mix_inputs = "".join(f"[a{i}]" for i in range(n))
            filter_str = ";".join(volume_parts) + f";{mix_inputs}amix=inputs={n}:duration={duration_mode}:dropout_transition=2"
        else:
            filter_str = f"amix=inputs={n}:duration={duration_mode}:dropout_transition=2"

        cmd.extend(["-filter_complex", filter_str, str(output_path)])

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Mix error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Mixed {n} audio files",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Mix error: {e}")

    def crossfade(
        self,
        input1: str | Path,
        input2: str | Path,
        output_path: str | Path,
        fade_duration: float = 3.0,
    ) -> ProcessingResult:
        """Applies a crossfade between two audio files."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y",
            "-i", str(input1), "-i", str(input2),
            "-filter_complex",
            f"acrossfade=d={fade_duration}:c1=tri:c2=tri",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Crossfade error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Crossfade applied ({fade_duration}s)",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Crossfade error: {e}")

    def overlay_at_position(
        self,
        base_path: str | Path,
        overlay_path: str | Path,
        output_path: str | Path,
        position: float = 0,
        overlay_volume: float = 1.0,
    ) -> ProcessingResult:
        """
        Places one audio clip onto another at a given time.

        Args:
            position: Start time in seconds
            overlay_volume: Volume of the overlay track
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        delay_ms = int(position * 1000)

        cmd = [
            self.ffmpeg, "-y",
            "-i", str(base_path), "-i", str(overlay_path),
            "-filter_complex",
            f"[1:a]volume={overlay_volume},adelay={delay_ms}|{delay_ms}[ov];"
            f"[0:a][ov]amix=inputs=2:duration=first:dropout_transition=2",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Overlay error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Audio overlay added @ {position}s",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Audio overlay error: {e}")

    def create_silence(
        self,
        output_path: str | Path,
        duration: float = 5.0,
        sample_rate: int = 44100,
    ) -> ProcessingResult:
        """Creates a silent audio file of the given duration."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
            "-t", str(duration),
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Silence generation error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Created {duration}s of silence",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Silence create error: {e}")
