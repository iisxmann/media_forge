"""
Audio processing module.
Basic audio operations: info retrieval, trim, normalize, volume adjustment.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mediaforge.core.base import BaseProcessor, MediaInfo, ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor(BaseProcessor):
    """Audio processing class. FFmpeg-based."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        super().__init__()
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def process(self, input_path: str | Path, output_path: str | Path, **kwargs) -> ProcessingResult:
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path)]

        if "volume" in kwargs:
            cmd.extend(["-af", f"volume={kwargs['volume']}"])
        if "sample_rate" in kwargs:
            cmd.extend(["-ar", str(kwargs["sample_rate"])])
        if "channels" in kwargs:
            cmd.extend(["-ac", str(kwargs["channels"])])
        if "bitrate" in kwargs:
            cmd.extend(["-b:a", kwargs["bitrate"]])

        cmd.append(str(output_path))
        return self._run_ffmpeg(cmd)

    def get_audio_info(self, file_path: str | Path) -> MediaInfo:
        """Returns audio file metadata."""
        path = Path(file_path)
        if not path.exists():
            raise ProcessingError(f"File not found: {path}")

        try:
            cmd = [
                self.ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe = json.loads(result.stdout)

            audio_stream = next((s for s in probe.get("streams", []) if s["codec_type"] == "audio"), None)
            fmt = probe.get("format", {})

            info = MediaInfo(
                path=path,
                format=path.suffix.lstrip(".").lower(),
                size_bytes=int(fmt.get("size", path.stat().st_size)),
                duration=float(fmt.get("duration", 0)),
                bitrate=int(fmt.get("bit_rate", 0)),
            )

            if audio_stream:
                info.audio_codec = audio_stream.get("codec_name", "")
                info.channels = int(audio_stream.get("channels", 0))
                info.sample_rate = int(audio_stream.get("sample_rate", 0))

            return info
        except Exception as e:
            raise ProcessingError(f"Audio info error: {e}")

    def trim(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float,
        end_time: float | None = None,
        duration: float | None = None,
    ) -> ProcessingResult:
        """Trims an audio file."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-ss", str(start_time)]
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])
        elif duration is not None:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-c", "copy", str(output_path)])

        return self._run_ffmpeg(cmd)

    def normalize(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_loudness: float = -16.0,
        target_peak: float = -1.5,
    ) -> ProcessingResult:
        """
        Normalizes loudness (EBU R128).

        Args:
            target_loudness: Target loudness (LUFS)
            target_peak: Target true peak (dBTP)
        """
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-af", f"loudnorm=I={target_loudness}:TP={target_peak}:LRA=11",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    def change_volume(
        self,
        input_path: str | Path,
        output_path: str | Path,
        factor: float = 1.5,
    ) -> ProcessingResult:
        """Changes volume. factor=1.0 original, 2.0=double, 0.5=half."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-af", f"volume={factor}",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    def change_speed(
        self,
        input_path: str | Path,
        output_path: str | Path,
        speed: float = 1.5,
        preserve_pitch: bool = True,
    ) -> ProcessingResult:
        """
        Changes playback speed.

        Args:
            speed: Speed multiplier (0.5=slower, 2.0=faster)
            preserve_pitch: Preserve pitch (via rubberband)
        """
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        if preserve_pitch:
            af = f"rubberband=tempo={speed}"
        else:
            af = f"atempo={speed}"
            if speed > 2.0:
                tempos = []
                s = speed
                while s > 2.0:
                    tempos.append("atempo=2.0")
                    s /= 2.0
                tempos.append(f"atempo={s}")
                af = ",".join(tempos)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-af", af, str(output_path)]
        return self._run_ffmpeg(cmd)

    def change_pitch(
        self,
        input_path: str | Path,
        output_path: str | Path,
        semitones: float = 2.0,
    ) -> ProcessingResult:
        """Changes pitch (in semitones)."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        factor = 2 ** (semitones / 12)
        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-af", f"rubberband=pitch={factor}",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    def to_mono(self, input_path: str | Path, output_path: str | Path) -> ProcessingResult:
        """Converts stereo audio to mono."""
        return self.process(input_path, output_path, channels=1)

    def to_stereo(self, input_path: str | Path, output_path: str | Path) -> ProcessingResult:
        """Converts mono audio to stereo."""
        return self.process(input_path, output_path, channels=2)

    def split_channels(
        self, input_path: str | Path, output_dir: str | Path
    ) -> list[ProcessingResult]:
        """Splits a stereo file into left and right channel files."""
        input_path = self.validate_input(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        ext = input_path.suffix

        results = []
        for channel, name in [(0, "left"), (1, "right")]:
            out = output_dir / f"{stem}_{name}{ext}"
            cmd = [
                self.ffmpeg, "-y", "-i", str(input_path),
                "-af", f"pan=mono|c0=c{channel}",
                str(out),
            ]
            results.append(self._run_ffmpeg(cmd))
            results[-1].output_path = out

        return results

    def _run_ffmpeg(self, cmd: list[str]) -> ProcessingResult:
        start = time.time()
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"FFmpeg error: {process.stderr[:500]}")
            return ProcessingResult(
                success=True, message="Audio processed",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"FFmpeg execution error: {e}")
