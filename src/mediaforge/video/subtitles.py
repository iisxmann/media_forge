"""
Subtitle manager.
Read/write SRT/ASS/VTT, burn-in or embed subtitles, synchronization.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SubtitleEntry:
    """A single subtitle cue."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    style: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_time(self, seconds: float) -> str:
        """Converts seconds to SRT time format (HH:MM:SS,mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def to_vtt_time(self, seconds: float) -> str:
        """Converts seconds to WebVTT time format (HH:MM:SS.mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class SubtitleManager:
    """Subtitle management class."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    def parse_srt(self, file_path: str | Path) -> list[SubtitleEntry]:
        """Parses an SRT file."""
        path = Path(file_path)
        if not path.exists():
            raise ProcessingError(f"Subtitle file not found: {path}")

        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        entries = []
        blocks = re.split(r"\n\n+", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0].strip())
                time_match = re.match(
                    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                    lines[1].strip(),
                )
                if not time_match:
                    continue

                start = self._parse_srt_time(time_match.group(1))
                end = self._parse_srt_time(time_match.group(2))
                text = "\n".join(lines[2:])

                entries.append(SubtitleEntry(index=index, start_time=start, end_time=end, text=text))
            except (ValueError, IndexError):
                continue

        return entries

    def write_srt(self, entries: list[SubtitleEntry], output_path: str | Path) -> Path:
        """Writes a list of SubtitleEntry objects to an SRT file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for i, entry in enumerate(entries, 1):
            lines.append(str(i))
            lines.append(f"{entry.to_srt_time(entry.start_time)} --> {entry.to_srt_time(entry.end_time)}")
            lines.append(entry.text)
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def write_vtt(self, entries: list[SubtitleEntry], output_path: str | Path) -> Path:
        """Writes a list of SubtitleEntry objects to a WebVTT file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["WEBVTT", ""]
        for entry in entries:
            lines.append(f"{entry.to_vtt_time(entry.start_time)} --> {entry.to_vtt_time(entry.end_time)}")
            lines.append(entry.text)
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def convert_subtitle(
        self, input_path: str | Path, output_path: str | Path, target_format: str
    ) -> Path:
        """Converts subtitle format (SRT <-> VTT)."""
        entries = self.parse_srt(input_path)

        if target_format.lower() == "vtt":
            return self.write_vtt(entries, output_path)
        elif target_format.lower() == "srt":
            return self.write_srt(entries, output_path)
        else:
            raise ProcessingError(f"Unsupported subtitle format: {target_format}")

    def burn_subtitles(
        self,
        video_path: str | Path,
        subtitle_path: str | Path,
        output_path: str | Path,
        font_size: int = 24,
        font_color: str = "&HFFFFFF",
        outline_color: str = "&H000000",
        margin_v: int = 30,
    ) -> ProcessingResult:
        """
        Burns subtitles into the video (hardcoded). Cannot be turned off in players.

        Args:
            font_size: Font size
            font_color: Text color (ASS format)
            margin_v: Distance from bottom edge in pixels
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

        style = f"FontSize={font_size},PrimaryColour={font_color},OutlineColour={outline_color},MarginV={margin_v}"
        filter_str = f"subtitles='{sub_path_escaped}':force_style='{style}'"

        cmd = [
            self.ffmpeg, "-y", "-i", str(video_path),
            "-vf", filter_str, "-c:a", "copy",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Subtitle burn-in error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Subtitles burned into video (hardcoded)",
                duration_seconds=time.time() - start,
            )
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Burn subtitles error: {e}")

    def embed_subtitles(
        self,
        video_path: str | Path,
        subtitle_path: str | Path,
        output_path: str | Path,
        language: str = "tur",
        title: str = "Turkish",
    ) -> ProcessingResult:
        """
        Embeds subtitles in the container (soft subtitles). Can be toggled in players.

        Args:
            language: ISO 639-2 language code
            title: Subtitle track title
        """
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y",
            "-i", str(video_path), "-i", str(subtitle_path),
            "-c", "copy", "-c:s", "mov_text",
            "-metadata:s:s:0", f"language={language}",
            "-metadata:s:s:0", f"title={title}",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Subtitle embedding error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Subtitles embedded in video (soft)",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Embed subtitles error: {e}")

    def extract_subtitles(
        self,
        video_path: str | Path,
        output_path: str | Path,
        stream_index: int = 0,
    ) -> ProcessingResult:
        """Extracts embedded subtitles from the video."""
        start = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y", "-i", str(video_path),
            "-map", f"0:s:{stream_index}",
            str(output_path),
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ProcessingError(f"Subtitle extraction error: {process.stderr[:500]}")

            return ProcessingResult(
                success=True, output_path=output_path,
                message="Subtitles extracted",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            raise ProcessingError(f"Extract subtitles error: {e}")

    def shift_timing(
        self, entries: list[SubtitleEntry], offset_seconds: float
    ) -> list[SubtitleEntry]:
        """Shifts all subtitle timings (positive = later, negative = earlier)."""
        shifted = []
        for entry in entries:
            new_start = max(0, entry.start_time + offset_seconds)
            new_end = max(0, entry.end_time + offset_seconds)
            if new_end > 0:
                shifted.append(SubtitleEntry(
                    index=entry.index, start_time=new_start,
                    end_time=new_end, text=entry.text, style=entry.style,
                ))
        return shifted

    def scale_timing(
        self, entries: list[SubtitleEntry], factor: float
    ) -> list[SubtitleEntry]:
        """Scales subtitle timing (use after video speed changes)."""
        return [
            SubtitleEntry(
                index=e.index,
                start_time=e.start_time * factor,
                end_time=e.end_time * factor,
                text=e.text, style=e.style,
            )
            for e in entries
        ]

    def merge_subtitles(
        self, *subtitle_lists: list[SubtitleEntry]
    ) -> list[SubtitleEntry]:
        """Merges multiple subtitle lists and sorts by start time."""
        merged = []
        for sub_list in subtitle_lists:
            merged.extend(sub_list)
        merged.sort(key=lambda e: e.start_time)
        for i, entry in enumerate(merged, 1):
            entry.index = i
        return merged

    @staticmethod
    def _parse_srt_time(time_str: str) -> float:
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
