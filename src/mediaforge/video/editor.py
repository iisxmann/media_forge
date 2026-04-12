"""
Video editor.
Trimming, concatenation, overlays, text, transition-style operations.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class VideoEditor:
    """Video editing operations. FFmpeg-based."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def trim(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float,
        end_time: float | None = None,
        duration: float | None = None,
    ) -> ProcessingResult:
        """
        Trims the video.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            duration: Duration in seconds (alternative to end_time)
        """
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-ss", str(start_time)]

        if end_time is not None:
            cmd.extend(["-to", str(end_time)])
        elif duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-c", "copy", str(output_path)])

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Trim error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Video trimmed: {start_time}s -> {end_time or (start_time + (duration or 0))}s",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Trim error: {e}")

    def concat(
        self,
        input_paths: list[str | Path],
        output_path: str | Path,
        method: str = "demuxer",
    ) -> ProcessingResult:
        """
        Concatenates multiple videos.

        Args:
            method: Concatenation method
                - 'demuxer': Fast, for same-codec videos (no re-encode)
                - 'filter': For different codecs/resolutions (re-encode)
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if method == "demuxer":
                return self._concat_demuxer(input_paths, output_path, start)
            elif method == "filter":
                return self._concat_filter(input_paths, output_path, start)
            else:
                raise ProcessingError(f"Invalid method: {method}. Use 'demuxer' or 'filter'")
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Concatenation error: {e}")

    def add_text_overlay(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text: str,
        position: str = "center",
        font_size: int = 48,
        font_color: str = "white",
        bg_color: str | None = None,
        start_time: float = 0,
        end_time: float | None = None,
    ) -> ProcessingResult:
        """
        Adds text on top of the video.

        Args:
            text: Text to display
            position: Position (center, top, bottom, top-left, top-right, bottom-left, bottom-right)
            font_size: Font size
            font_color: Text color
            bg_color: Background color (None = transparent)
            start_time: When the text appears
            end_time: When the text disappears (None = until end of video)
        """
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pos_map = {
            "center": "x=(w-text_w)/2:y=(h-text_h)/2",
            "top": "x=(w-text_w)/2:y=20",
            "bottom": "x=(w-text_w)/2:y=h-text_h-20",
            "top-left": "x=20:y=20",
            "top-right": "x=w-text_w-20:y=20",
            "bottom-left": "x=20:y=h-text_h-20",
            "bottom-right": "x=w-text_w-20:y=h-text_h-20",
        }

        pos = pos_map.get(position, pos_map["center"])
        escaped_text = text.replace("'", "'\\''").replace(":", "\\:")

        drawtext = f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:{pos}"

        if bg_color:
            drawtext += f":box=1:boxcolor={bg_color}@0.7:boxborderw=10"
        if start_time > 0 or end_time:
            enable = f"enable='between(t,{start_time},{end_time or 9999})'"
            drawtext += f":{enable}"

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-vf", drawtext, "-c:a", "copy",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Text overlay error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Text added: '{text}'",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Text overlay error: {e}")

    def add_image_overlay(
        self,
        input_path: str | Path,
        output_path: str | Path,
        overlay_path: str | Path,
        x: int = 10,
        y: int = 10,
        opacity: float = 1.0,
        start_time: float = 0,
        end_time: float | None = None,
    ) -> ProcessingResult:
        """Adds an image overlay on the video (logo, watermark, etc.)."""
        start = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        filter_str = f"overlay={x}:{y}"
        if opacity < 1.0:
            filter_str = f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={x}:{y}"
        if start_time > 0 or end_time:
            enable = f":enable='between(t,{start_time},{end_time or 9999})'"
            filter_str += enable

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path), "-i", str(overlay_path),
            "-filter_complex", filter_str, "-c:a", "copy",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Overlay error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Image overlay added",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Image overlay error: {e}")

    def picture_in_picture(
        self,
        main_path: str | Path,
        pip_path: str | Path,
        output_path: str | Path,
        pip_width: int = 320,
        position: str = "bottom-right",
        margin: int = 20,
    ) -> ProcessingResult:
        """
        Picture-in-picture effect. The smaller video plays in a corner of the main video.

        Args:
            pip_width: Width of the smaller video in pixels
            position: PiP position
            margin: Distance from edges in pixels
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pos_map = {
            "top-left": f"overlay={margin}:{margin}",
            "top-right": f"overlay=main_w-overlay_w-{margin}:{margin}",
            "bottom-left": f"overlay={margin}:main_h-overlay_h-{margin}",
            "bottom-right": f"overlay=main_w-overlay_w-{margin}:main_h-overlay_h-{margin}",
        }

        overlay = pos_map.get(position, pos_map["bottom-right"])
        filter_str = f"[1:v]scale={pip_width}:-1[pip];[0:v][pip]{overlay}"

        cmd = [
            self.ffmpeg, "-y", "-i", str(main_path), "-i", str(pip_path),
            "-filter_complex", filter_str,
            "-c:a", "copy", "-shortest",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"PiP error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message=f"Picture-in-picture added ({position})",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"PiP error: {e}")

    def split(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        segment_duration: float = 60,
    ) -> list[ProcessingResult]:
        """Splits the video into equal-length segments."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        cmd = [
            self.ffmpeg, "-y", "-i", str(input_path),
            "-c", "copy", "-map", "0",
            "-segment_time", str(segment_duration),
            "-f", "segment", "-reset_timestamps", "1",
            str(output_dir / f"{stem}_%03d.mp4"),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Split error: {process.stderr[:500]}")

            segments = sorted(output_dir.glob(f"{stem}_*.mp4"))
            return [
                ProcessingResult(success=True, output_path=s, message=f"Segment: {s.name}")
                for s in segments
            ]
        except Exception as e:
            raise ProcessingError(f"Split error: {e}")

    def _concat_demuxer(
        self, input_paths: list[str | Path], output_path: Path, start: float
    ) -> ProcessingResult:
        list_file = output_path.parent / "_concat_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in input_paths:
                f.write(f"file '{Path(p).absolute()}'\n")

        cmd = [
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy",
            str(output_path),
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        list_file.unlink(missing_ok=True)

        if process.returncode != 0:
            raise ProcessingError(f"Concatenation error: {process.stderr[:500]}")

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Concatenated {len(input_paths)} videos",
            duration_seconds=time.time() - start,
        )

    def _concat_filter(
        self, input_paths: list[str | Path], output_path: Path, start: float
    ) -> ProcessingResult:
        cmd = [self.ffmpeg, "-y"]
        for p in input_paths:
            cmd.extend(["-i", str(p)])

        n = len(input_paths)
        filter_parts = [f"[{i}:v][{i}:a]" for i in range(n)]
        filter_str = "".join(filter_parts) + f"concat=n={n}:v=1:a=1[outv][outa]"

        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[outv]", "-map", "[outa]",
            str(output_path),
        ])

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise ProcessingError(f"Concatenation error: {process.stderr[:500]}")

        return ProcessingResult(
            success=True, output_path=output_path,
            message=f"Concatenated {n} videos (re-encoded)",
            duration_seconds=time.time() - start,
        )
