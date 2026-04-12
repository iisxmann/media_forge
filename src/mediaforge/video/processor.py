"""
Video processing module.
Basic video operations: info retrieval, resolution changes, FPS adjustment,
rotation, cropping, speed changes, audio extraction/merging.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mediaforge.core.base import BaseProcessor, MediaInfo, ProcessingResult
from mediaforge.core.exceptions import ProcessingError, DependencyError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor(BaseProcessor):
    """Comprehensive video processing class. FFmpeg-based."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        super().__init__()
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        try:
            subprocess.run([self.ffmpeg, "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise DependencyError(
                "FFmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html"
            )

    def process(self, input_path: str | Path, output_path: str | Path, **kwargs) -> ProcessingResult:
        """General video processing. Applies the appropriate operation based on kwargs."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path)]

        if "resolution" in kwargs:
            w, h = kwargs["resolution"]
            cmd.extend(["-vf", f"scale={w}:{h}"])
        if "fps" in kwargs:
            cmd.extend(["-r", str(kwargs["fps"])])
        if "bitrate" in kwargs:
            cmd.extend(["-b:v", kwargs["bitrate"]])
        if "codec" in kwargs:
            cmd.extend(["-c:v", kwargs["codec"]])

        cmd.append(str(output_path))
        return self._run_ffmpeg(cmd, input_path, output_path)

    def get_video_info(self, file_path: str | Path) -> MediaInfo:
        """Returns detailed video information via FFprobe."""
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

            video_stream = next((s for s in probe.get("streams", []) if s["codec_type"] == "video"), None)
            audio_stream = next((s for s in probe.get("streams", []) if s["codec_type"] == "audio"), None)
            fmt = probe.get("format", {})

            info = MediaInfo(
                path=path,
                format=path.suffix.lstrip(".").lower(),
                size_bytes=int(fmt.get("size", path.stat().st_size)),
                duration=float(fmt.get("duration", 0)),
                bitrate=int(fmt.get("bit_rate", 0)),
            )

            if video_stream:
                info.width = int(video_stream.get("width", 0))
                info.height = int(video_stream.get("height", 0))
                info.codec = video_stream.get("codec_name", "")
                fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
                info.fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 0

            if audio_stream:
                info.audio_codec = audio_stream.get("codec_name", "")
                info.channels = int(audio_stream.get("channels", 0))
                info.sample_rate = int(audio_stream.get("sample_rate", 0))

            info.metadata = {
                "format_name": fmt.get("format_name", ""),
                "format_long_name": fmt.get("format_long_name", ""),
                "nb_streams": int(fmt.get("nb_streams", 0)),
                "tags": fmt.get("tags", {}),
            }

            return info
        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"Could not retrieve video information: {e.stderr}")
        except Exception as e:
            raise ProcessingError(f"Video information error: {e}")

    def change_resolution(
        self,
        input_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        maintain_aspect: bool = True,
    ) -> ProcessingResult:
        """Changes video resolution."""
        start = time.time()
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        if maintain_aspect:
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = f"scale={width}:{height}"

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", scale_filter,
            "-c:a", "copy",
            str(output_path),
        ]
        result = self._run_ffmpeg(cmd, input_path, output_path)
        result.duration_seconds = time.time() - start
        return result

    def change_fps(
        self, input_path: str | Path, output_path: str | Path, fps: int
    ) -> ProcessingResult:
        """Changes video FPS."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-r", str(fps), "-c:a", "copy",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd, input_path, output_path)

    def change_speed(
        self,
        input_path: str | Path,
        output_path: str | Path,
        speed: float,
        adjust_audio: bool = True,
    ) -> ProcessingResult:
        """
        Changes video playback speed.

        Args:
            speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
            adjust_audio: Whether to speed up or slow down audio as well
        """
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        video_filter = f"setpts={1/speed}*PTS"
        if adjust_audio:
            audio_filter = f"atempo={speed}"
            if speed > 2.0:
                audio_filter = ",".join([f"atempo={min(s, 2.0)}" for s in self._split_tempo(speed)])
            cmd = [
                self.ffmpeg, "-y", "-i", str(input_path),
                "-filter:v", video_filter, "-filter:a", audio_filter,
                str(output_path),
            ]
        else:
            cmd = [
                self.ffmpeg, "-y", "-i", str(input_path),
                "-filter:v", video_filter, "-an",
                str(output_path),
            ]

        return self._run_ffmpeg(cmd, input_path, output_path)

    def rotate(
        self, input_path: str | Path, output_path: str | Path, angle: int
    ) -> ProcessingResult:
        """
        Rotates the video.

        Args:
            angle: Rotation angle (90, 180, 270)
        """
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        transpose_map = {
            90: "transpose=1",
            180: "transpose=1,transpose=1",
            270: "transpose=2",
        }

        if angle not in transpose_map:
            raise ProcessingError(f"Supported angles: 90, 180, 270. Given: {angle}")

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", transpose_map[angle], "-c:a", "copy",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd, input_path, output_path)

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        audio_format: str = "mp3",
        bitrate: str = "192k",
    ) -> ProcessingResult:
        """Extracts an audio file from the video."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        codec_map = {
            "mp3": "libmp3lame",
            "aac": "aac",
            "wav": "pcm_s16le",
            "flac": "flac",
            "ogg": "libvorbis",
        }

        codec = codec_map.get(audio_format, "libmp3lame")
        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vn", "-acodec", codec, "-ab", bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd, input_path, output_path)

    def add_audio(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        replace: bool = True,
    ) -> ProcessingResult:
        """
        Adds an audio track to the video.

        Args:
            replace: If True, replaces existing audio; if False, mixes with it
        """
        video_path = self.validate_input(video_path)
        audio_path = self.validate_input(audio_path)
        output_path = self.prepare_output(output_path)

        if replace:
            cmd = [
                self.ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                str(output_path),
            ]
        else:
            cmd = [
                self.ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
                "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=2",
                "-c:v", "copy",
                str(output_path),
            ]
        return self._run_ffmpeg(cmd, video_path, output_path)

    def remove_audio(
        self, input_path: str | Path, output_path: str | Path
    ) -> ProcessingResult:
        """Removes audio from the video."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-c:v", "copy", "-an",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd, input_path, output_path)

    def create_gif(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float = 0,
        duration: float = 5,
        fps: int = 15,
        width: int = 480,
    ) -> ProcessingResult:
        """Creates a GIF from a segment of the video."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        palette_path = output_path.parent / "_palette.png"
        palette_cmd = [
            self.ffmpeg, "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", str(input_path),
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
            str(palette_path),
        ]
        subprocess.run(palette_cmd, capture_output=True, check=True)

        gif_cmd = [
            self.ffmpeg, "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", str(input_path), "-i", str(palette_path),
            "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse",
            str(output_path),
        ]
        result = self._run_ffmpeg(gif_cmd, input_path, output_path)

        if palette_path.exists():
            palette_path.unlink()

        return result

    def stabilize(
        self, input_path: str | Path, output_path: str | Path, strength: str = "medium"
    ) -> ProcessingResult:
        """
        Applies video stabilization.

        Args:
            strength: Stabilization strength (low, medium, high)
        """
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        shakiness_map = {"low": 4, "medium": 7, "high": 10}
        shakiness = shakiness_map.get(strength, 7)

        transform_path = output_path.parent / "_transforms.trf"

        detect_cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", f"vidstabdetect=shakiness={shakiness}:result={transform_path}",
            "-f", "null", "-",
        ]
        subprocess.run(detect_cmd, capture_output=True)

        stabilize_cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", f"vidstabtransform=input={transform_path}:smoothing=30,unsharp",
            str(output_path),
        ]
        result = self._run_ffmpeg(stabilize_cmd, input_path, output_path)

        if transform_path.exists():
            transform_path.unlink()

        return result

    def reverse(
        self, input_path: str | Path, output_path: str | Path
    ) -> ProcessingResult:
        """Reverses the video."""
        input_path = self.validate_input(input_path)
        output_path = self.prepare_output(output_path)

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", "reverse", "-af", "areverse",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd, input_path, output_path)

    def _run_ffmpeg(
        self, cmd: list[str], input_path: Path, output_path: Path
    ) -> ProcessingResult:
        start = time.time()
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"FFmpeg error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True,
                output_path=output_path,
                message="Video processed successfully",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"FFmpeg execution error: {e}")

    @staticmethod
    def _split_tempo(speed: float) -> list[float]:
        """The atempo filter works in the 0.5–2.0 range; splits larger values."""
        tempos = []
        while speed > 2.0:
            tempos.append(2.0)
            speed /= 2.0
        tempos.append(speed)
        return tempos
