"""
Audio effects.
Fade, echo, reverb, equalizer, noise reduction, compressor.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class AudioEffects:
    """Audio effects using FFmpeg audio filters."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def fade_in(
        self, input_path: str | Path, output_path: str | Path, duration: float = 2.0
    ) -> ProcessingResult:
        """Applies a fade-in effect."""
        return self._apply(input_path, output_path, f"afade=t=in:st=0:d={duration}")

    def fade_out(
        self, input_path: str | Path, output_path: str | Path, duration: float = 2.0
    ) -> ProcessingResult:
        """Applies a fade-out effect from the end of the file."""
        from mediaforge.audio.processor import AudioProcessor
        ap = AudioProcessor(ffmpeg_path=self.ffmpeg)
        info = ap.get_audio_info(input_path)
        start = info.duration - duration
        return self._apply(input_path, output_path, f"afade=t=out:st={start}:d={duration}")

    def echo(
        self,
        input_path: str | Path,
        output_path: str | Path,
        delay: float = 0.5,
        decay: float = 0.3,
    ) -> ProcessingResult:
        """Echo effect. delay=delay in seconds, decay=decay (0-1)."""
        delay_ms = int(delay * 1000)
        return self._apply(
            input_path, output_path,
            f"aecho=0.8:0.9:{delay_ms}:{decay}"
        )

    def reverb(
        self, input_path: str | Path, output_path: str | Path, room_size: float = 0.5
    ) -> ProcessingResult:
        """Reverb effect."""
        delays = "|".join([str(int(50 * (i + 1) * room_size)) for i in range(4)])
        decays = "|".join([str(round(0.5 * (0.7 ** i), 2)) for i in range(4)])
        return self._apply(
            input_path, output_path,
            f"aecho=0.8:0.88:{delays}:{decays}"
        )

    def bass_boost(
        self, input_path: str | Path, output_path: str | Path, gain: float = 10.0
    ) -> ProcessingResult:
        """Boosts bass frequencies."""
        return self._apply(input_path, output_path, f"bass=g={gain}:f=100:w=0.5")

    def treble_boost(
        self, input_path: str | Path, output_path: str | Path, gain: float = 5.0
    ) -> ProcessingResult:
        """Boosts treble frequencies."""
        return self._apply(input_path, output_path, f"treble=g={gain}:f=3000:w=0.5")

    def equalizer(
        self,
        input_path: str | Path,
        output_path: str | Path,
        bands: list[dict] | None = None,
    ) -> ProcessingResult:
        """
        Multi-band equalizer.

        Args:
            bands: [{"frequency": 100, "gain": 5, "width": 1.0}, ...]
        """
        if not bands:
            bands = [
                {"frequency": 60, "gain": 3, "width": 1.0},
                {"frequency": 250, "gain": 0, "width": 1.0},
                {"frequency": 1000, "gain": 0, "width": 1.0},
                {"frequency": 4000, "gain": 2, "width": 1.0},
                {"frequency": 12000, "gain": 1, "width": 1.0},
            ]

        eq_parts = [
            f"equalizer=f={b['frequency']}:t=h:w={b.get('width', 1.0)}:g={b['gain']}"
            for b in bands
        ]
        return self._apply(input_path, output_path, ",".join(eq_parts))

    def noise_reduction(
        self, input_path: str | Path, output_path: str | Path, strength: float = 0.5
    ) -> ProcessingResult:
        """Noise reduction using FFmpeg's anlmdn filter."""
        return self._apply(
            input_path, output_path,
            f"anlmdn=s={strength}:p=0.01:r=0.002:o=o"
        )

    def compressor(
        self,
        input_path: str | Path,
        output_path: str | Path,
        threshold: float = -20,
        ratio: float = 4,
        attack: float = 20,
        release: float = 250,
    ) -> ProcessingResult:
        """
        Dynamic range compressor.

        Args:
            threshold: Threshold (dB)
            ratio: Compression ratio
            attack: Attack time (ms)
            release: Release time (ms)
        """
        return self._apply(
            input_path, output_path,
            f"acompressor=threshold={threshold}dB:ratio={ratio}:attack={attack}:release={release}"
        )

    def limiter(
        self, input_path: str | Path, output_path: str | Path, limit: float = -1.0
    ) -> ProcessingResult:
        """Limits peak level (in dB)."""
        return self._apply(input_path, output_path, f"alimiter=limit={limit}dB")

    def silence_remove(
        self, input_path: str | Path, output_path: str | Path, threshold: float = -50
    ) -> ProcessingResult:
        """Removes silent segments."""
        return self._apply(
            input_path, output_path,
            f"silenceremove=stop_periods=-1:stop_duration=1:stop_threshold={threshold}dB"
        )

    def reverse(self, input_path: str | Path, output_path: str | Path) -> ProcessingResult:
        """Reverses audio."""
        return self._apply(input_path, output_path, "areverse")

    def _apply(
        self, input_path: str | Path, output_path: str | Path, audio_filter: str
    ) -> ProcessingResult:
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-af", audio_filter,
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Audio effect error: {process.stderr[:500]}")
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Audio effect applied",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Audio effect error: {e}")
