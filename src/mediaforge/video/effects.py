"""
Video effects.
Color correction, filters, transitions, border framing.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class VideoEffects:
    """Video effects class. FFmpeg filter-based."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def apply_filter(
        self,
        input_path: str | Path,
        output_path: str | Path,
        filter_name: str,
        **kwargs,
    ) -> ProcessingResult:
        """
        Applies an FFmpeg video filter.

        Supported filters:
            grayscale, sepia, negative, blur, sharpen, edge_detect,
            vignette, denoise, mirror, vintage, high_contrast,
            brightness, saturation, gamma
        """
        start = time.time()
        filters = self._get_filter(filter_name, **kwargs)

        if not filters:
            raise ProcessingError(
                f"Unknown filter: {filter_name}. "
                f"Supported: {', '.join(self.list_filters())}"
            )

        return self._apply(input_path, output_path, filters, start)

    def adjust_colors(
        self,
        input_path: str | Path,
        output_path: str | Path,
        brightness: float = 0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
        hue: float = 0,
    ) -> ProcessingResult:
        """
        Color adjustments.

        Args:
            brightness: Brightness (-1.0 to 1.0)
            contrast: Contrast (0.0 to 3.0)
            saturation: Saturation (0.0 to 3.0)
            gamma: Gamma (0.1 to 10.0)
            hue: Hue shift in degrees (0–360)
        """
        start = time.time()
        filter_parts = []

        if brightness != 0 or contrast != 1.0:
            filter_parts.append(f"eq=brightness={brightness}:contrast={contrast}:gamma={gamma}")
        elif gamma != 1.0:
            filter_parts.append(f"eq=gamma={gamma}")

        if saturation != 1.0 or hue != 0:
            filter_parts.append(f"hue=s={saturation}:h={hue}")

        if not filter_parts:
            filter_parts.append("null")

        return self._apply(input_path, output_path, ",".join(filter_parts), start)

    def add_border(
        self,
        input_path: str | Path,
        output_path: str | Path,
        width: int = 10,
        color: str = "black",
    ) -> ProcessingResult:
        """Adds a border around the video."""
        start = time.time()
        filter_str = f"pad=iw+{width*2}:ih+{width*2}:{width}:{width}:color={color}"
        return self._apply(input_path, output_path, filter_str, start)

    def picture_fade(
        self,
        input_path: str | Path,
        output_path: str | Path,
        fade_in_duration: float = 1.0,
        fade_out_duration: float = 1.0,
    ) -> ProcessingResult:
        """Applies fade in/out."""
        start = time.time()
        from mediaforge.video.processor import VideoProcessor
        vp = VideoProcessor(ffmpeg_path=self.ffmpeg)
        info = vp.get_video_info(input_path)
        duration = info.duration

        filters = []
        if fade_in_duration > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in_duration}")
        if fade_out_duration > 0:
            fade_start = duration - fade_out_duration
            filters.append(f"fade=t=out:st={fade_start}:d={fade_out_duration}")

        audio_filters = []
        if fade_in_duration > 0:
            audio_filters.append(f"afade=t=in:st=0:d={fade_in_duration}")
        if fade_out_duration > 0:
            fade_start = duration - fade_out_duration
            audio_filters.append(f"afade=t=out:st={fade_start}:d={fade_out_duration}")

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path)]
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])
        cmd.append(str(output_path))

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Fade error: {process.stderr[:500]}")
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Fade effect applied",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Fade error: {e}")

    def apply_lut(
        self,
        input_path: str | Path,
        output_path: str | Path,
        lut_path: str | Path,
    ) -> ProcessingResult:
        """
        Applies a LUT (look-up table).
        Uses .cube or .3dl files for professional color grading.
        """
        start = time.time()
        lut_ext = Path(lut_path).suffix.lower()
        if lut_ext == ".cube":
            filter_str = f"lut3d=file='{lut_path}'"
        elif lut_ext == ".3dl":
            filter_str = f"lut3d=file='{lut_path}'"
        else:
            raise ProcessingError(f"Unsupported LUT format: {lut_ext}. Use .cube or .3dl")

        return self._apply(input_path, output_path, filter_str, start)

    def slow_motion(
        self,
        input_path: str | Path,
        output_path: str | Path,
        factor: float = 2.0,
        interpolation: bool = True,
    ) -> ProcessingResult:
        """
        Slow-motion effect.

        Args:
            factor: Slowdown multiplier (2.0 = half speed)
            interpolation: Frame interpolation (smoother result)
        """
        start = time.time()
        if interpolation:
            filter_str = f"setpts={factor}*PTS,minterpolate=fps=60:mi_mode=mci"
        else:
            filter_str = f"setpts={factor}*PTS"

        return self._apply(input_path, output_path, filter_str, start, audio_filter=f"atempo={1/factor}" if factor <= 2.0 else None)

    def timelapse(
        self,
        input_path: str | Path,
        output_path: str | Path,
        speed_factor: int = 10,
    ) -> ProcessingResult:
        """Timelapse effect. Speeds up the video to create a time-lapse."""
        start = time.time()
        filter_str = f"setpts=PTS/{speed_factor}"
        return self._apply(input_path, output_path, filter_str, start, no_audio=True)

    def list_filters(self) -> list[str]:
        """Returns available filter names."""
        return [
            "grayscale", "sepia", "negative", "blur", "sharpen",
            "edge_detect", "vignette", "denoise", "mirror", "vintage",
            "high_contrast", "brightness", "saturation", "gamma",
        ]

    def _get_filter(self, name: str, **kwargs) -> str | None:
        filters = {
            "grayscale": "hue=s=0",
            "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
            "negative": "negate",
            "blur": f"boxblur={kwargs.get('radius', 5)}",
            "sharpen": "unsharp=5:5:1.5:5:5:0.0",
            "edge_detect": "edgedetect=low=0.1:high=0.4",
            "vignette": f"vignette=angle={kwargs.get('angle', 'PI/4')}",
            "denoise": f"nlmeans=s={kwargs.get('strength', 6)}:p=7:r=15",
            "mirror": "hflip",
            "vintage": "curves=vintage",
            "high_contrast": "eq=contrast=1.5:brightness=0.05",
            "brightness": f"eq=brightness={kwargs.get('value', 0.1)}",
            "saturation": f"hue=s={kwargs.get('value', 1.5)}",
            "gamma": f"eq=gamma={kwargs.get('value', 1.5)}",
        }
        return filters.get(name)

    def _apply(
        self, input_path: str | Path, output_path: str | Path,
        video_filter: str, start: float,
        audio_filter: str | None = None, no_audio: bool = False,
    ) -> ProcessingResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-vf", video_filter]

        if no_audio:
            cmd.append("-an")
        elif audio_filter:
            cmd.extend(["-af", audio_filter])
        else:
            cmd.extend(["-c:a", "copy"])

        cmd.append(str(output_path))

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Effect error: {process.stderr[:500]}")
            return ProcessingResult(
                success=True, output_path=output_path,
                message="Video effect applied",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Video effect error: {e}")
