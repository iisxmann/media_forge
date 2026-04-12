"""
Video format converter.
Conversion between MP4, AVI, MKV, MOV, WebM, and other formats.
Codec selection, bitrate control, two-pass encoding.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mediaforge.core.base import BaseConverter, ProcessingResult
from mediaforge.core.exceptions import ConversionError, UnsupportedFormatError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


CODEC_PRESETS = {
    "mp4": {"video": "libx264", "audio": "aac", "ext": "mp4"},
    "mp4_h265": {"video": "libx265", "audio": "aac", "ext": "mp4"},
    "webm": {"video": "libvpx-vp9", "audio": "libopus", "ext": "webm"},
    "avi": {"video": "mpeg4", "audio": "mp3", "ext": "avi"},
    "mkv": {"video": "libx264", "audio": "aac", "ext": "mkv"},
    "mov": {"video": "libx264", "audio": "aac", "ext": "mov"},
    "flv": {"video": "flv1", "audio": "mp3", "ext": "flv"},
    "wmv": {"video": "wmv2", "audio": "wmav2", "ext": "wmv"},
    "mpeg": {"video": "mpeg2video", "audio": "mp2", "ext": "mpeg"},
    "3gp": {"video": "h263", "audio": "aac", "ext": "3gp"},
}

QUALITY_PRESETS = {
    "ultra": {"crf": 15, "bitrate": "10000k", "preset": "slow"},
    "high": {"crf": 18, "bitrate": "5000k", "preset": "slow"},
    "medium": {"crf": 23, "bitrate": "2500k", "preset": "medium"},
    "low": {"crf": 28, "bitrate": "1000k", "preset": "fast"},
    "web": {"crf": 25, "bitrate": "1500k", "preset": "fast"},
    "mobile": {"crf": 30, "bitrate": "800k", "preset": "veryfast"},
}


class VideoConverter(BaseConverter):
    """Video format and codec converter."""

    SUPPORTED_FORMATS = list(CODEC_PRESETS.keys())

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        super().__init__()
        self.ffmpeg = ffmpeg_path

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_format: str,
        quality_preset: str = "medium",
        video_codec: str | None = None,
        audio_codec: str | None = None,
        bitrate: str | None = None,
        crf: int | None = None,
        two_pass: bool = False,
        **kwargs,
    ) -> ProcessingResult:
        """
        Converts video to the target format.

        Args:
            target_format: Target format (mp4, webm, avi, etc.)
            quality_preset: Quality preset (ultra, high, medium, low, web, mobile)
            video_codec: Override video codec (otherwise chosen automatically)
            audio_codec: Override audio codec
            bitrate: Override bitrate (e.g. '5000k')
            crf: Constant Rate Factor (lower = higher quality)
            two_pass: Two-pass encoding (better quality/size tradeoff)
        """
        start = time.time()
        target_format = target_format.lower().lstrip(".")

        if target_format not in CODEC_PRESETS and target_format not in ("mp4_h265",):
            fmt_key = target_format
            if fmt_key not in CODEC_PRESETS:
                raise UnsupportedFormatError(target_format, list(CODEC_PRESETS.keys()))
        else:
            fmt_key = target_format

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        preset_config = CODEC_PRESETS.get(fmt_key, CODEC_PRESETS["mp4"])
        quality = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["medium"])

        v_codec = video_codec or preset_config["video"]
        a_codec = audio_codec or preset_config["audio"]
        v_bitrate = bitrate or quality["bitrate"]
        v_crf = crf if crf is not None else quality["crf"]

        try:
            if two_pass and v_codec in ("libx264", "libx265", "libvpx-vp9"):
                return self._two_pass_encode(
                    input_path, output_path, v_codec, a_codec, v_bitrate, quality["preset"], start
                )

            cmd = [
                self.ffmpeg, "-y", "-i", str(input_path),
                "-c:v", v_codec,
                "-crf", str(v_crf),
                "-preset", quality["preset"],
                "-c:a", a_codec,
                "-b:a", "128k",
                str(output_path),
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ConversionError(f"Conversion error: {process.stderr[:500]}")

            src_fmt = input_path.suffix.lstrip(".").upper()
            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Converted {src_fmt} -> {target_format.upper()}",
                duration_seconds=time.time() - start,
                details={
                    "video_codec": v_codec,
                    "audio_codec": a_codec,
                    "crf": v_crf,
                    "quality_preset": quality_preset,
                },
            )
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(f"Video conversion error: {e}")

    def compress(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_size_mb: float | None = None,
        crf: int = 28,
    ) -> ProcessingResult:
        """
        Compresses the video.

        Args:
            target_size_mb: Target file size in MB. If None, uses CRF-based compression.
            crf: CRF value (higher = smaller file, lower quality)
        """
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if target_size_mb:
                from mediaforge.video.processor import VideoProcessor
                vp = VideoProcessor(ffmpeg_path=self.ffmpeg)
                info = vp.get_video_info(input_path)
                target_bits = target_size_mb * 8 * 1024 * 1024
                audio_bits = 128000 * info.duration
                video_bitrate = int((target_bits - audio_bits) / info.duration)
                video_bitrate = max(video_bitrate, 100000)

                cmd = [
                    self.ffmpeg, "-y", "-i", str(input_path),
                    "-c:v", "libx264", "-b:v", f"{video_bitrate}",
                    "-c:a", "aac", "-b:a", "128k",
                    str(output_path),
                ]
            else:
                cmd = [
                    self.ffmpeg, "-y", "-i", str(input_path),
                    "-c:v", "libx264", "-crf", str(crf),
                    "-preset", "medium", "-c:a", "aac", "-b:a", "128k",
                    str(output_path),
                ]

            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ConversionError(f"Compression error: {process.stderr[:500]}")

            original_size = input_path.stat().st_size / (1024 * 1024)
            new_size = output_path.stat().st_size / (1024 * 1024)
            ratio = (1 - new_size / original_size) * 100

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message=f"Compressed: {original_size:.1f}MB -> {new_size:.1f}MB ({ratio:.1f}% reduction)",
                duration_seconds=time.time() - start,
                details={"original_mb": round(original_size, 2), "compressed_mb": round(new_size, 2), "ratio": round(ratio, 1)},
            )
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(f"Compression error: {e}")

    def get_supported_formats(self) -> list[str]:
        return self.SUPPORTED_FORMATS.copy()

    def _two_pass_encode(
        self, input_path: Path, output_path: Path,
        v_codec: str, a_codec: str, bitrate: str, preset: str, start: float
    ) -> ProcessingResult:
        pass1 = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-c:v", v_codec, "-b:v", bitrate, "-preset", preset,
            "-pass", "1", "-an", "-f", "null",
            "NUL" if __import__("os").name == "nt" else "/dev/null",
        ]
        subprocess.run(pass1, capture_output=True)

        pass2 = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-c:v", v_codec, "-b:v", bitrate, "-preset", preset,
            "-pass", "2", "-c:a", a_codec, "-b:a", "128k",
            str(output_path),
        ]
        process = subprocess.run(pass2, capture_output=True, text=True)

        for f in Path(".").glob("ffmpeg2pass-*"):
            f.unlink(missing_ok=True)

        if process.returncode != 0:
            raise ConversionError(f"Two-pass encoding error: {process.stderr[:500]}")

        return ProcessingResult(
            success=True, output_path=output_path,
            message="Two-pass encoding completed",
            duration_seconds=time.time() - start,
        )
