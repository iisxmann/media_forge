"""
Audio format converter.
Conversions among MP3, WAV, OGG, FLAC, AAC, M4A, etc.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mediaforge.core.base import BaseConverter, ProcessingResult
from mediaforge.core.exceptions import ConversionError, UnsupportedFormatError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)

AUDIO_CODECS = {
    "mp3": {"codec": "libmp3lame", "default_bitrate": "192k"},
    "wav": {"codec": "pcm_s16le", "default_bitrate": None},
    "ogg": {"codec": "libvorbis", "default_bitrate": "192k"},
    "flac": {"codec": "flac", "default_bitrate": None},
    "aac": {"codec": "aac", "default_bitrate": "192k"},
    "m4a": {"codec": "aac", "default_bitrate": "192k"},
    "wma": {"codec": "wmav2", "default_bitrate": "192k"},
    "opus": {"codec": "libopus", "default_bitrate": "128k"},
}


class AudioConverter(BaseConverter):
    """Converts between audio formats."""

    SUPPORTED_FORMATS = list(AUDIO_CODECS.keys())

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        super().__init__()
        self.ffmpeg = ffmpeg_path

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_format: str,
        bitrate: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Converts an audio file to the target format.

        Args:
            target_format: Target format (mp3, wav, flac, etc.)
            bitrate: Bitrate (e.g. '192k', '320k')
            sample_rate: Sample rate in Hz (e.g. 44100, 48000)
            channels: Channel count (1=mono, 2=stereo)
        """
        start = time.time()
        target_format = target_format.lower().lstrip(".")

        if target_format not in AUDIO_CODECS:
            raise UnsupportedFormatError(target_format, self.SUPPORTED_FORMATS)

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        codec_config = AUDIO_CODECS[target_format]

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-acodec", codec_config["codec"],
        ]

        effective_bitrate = bitrate or codec_config["default_bitrate"]
        if effective_bitrate:
            cmd.extend(["-b:a", effective_bitrate])
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        if channels:
            cmd.extend(["-ac", str(channels)])

        cmd.append(str(output_path))

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ConversionError(f"Audio conversion error: {process.stderr[:500]}")

            src_fmt = input_path.suffix.lstrip(".").upper()
            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Converted {src_fmt} -> {target_format.upper()}",
                duration_seconds=time.time() - start,
                details={
                    "codec": codec_config["codec"],
                    "bitrate": effective_bitrate,
                    "sample_rate": sample_rate,
                    "channels": channels,
                },
            )
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(f"Audio conversion error: {e}")

    def get_supported_formats(self) -> list[str]:
        return self.SUPPORTED_FORMATS.copy()
