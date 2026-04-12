"""
Video quality analysis.
VMAF, PSNR, SSIM metrics, bitrate analysis, codec comparison.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import ProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class VideoQualityAnalyzer:
    """Video quality analysis class."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def analyze(self, file_path: str | Path) -> dict[str, Any]:
        """Performs comprehensive video quality analysis."""
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

            analysis = {
                "file": {
                    "name": path.name,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "format": fmt.get("format_long_name", ""),
                    "duration_seconds": round(float(fmt.get("duration", 0)), 2),
                    "overall_bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000, 1),
                },
            }

            if video_stream:
                fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 0

                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))

                analysis["video"] = {
                    "codec": video_stream.get("codec_name", ""),
                    "codec_long": video_stream.get("codec_long_name", ""),
                    "profile": video_stream.get("profile", ""),
                    "width": width,
                    "height": height,
                    "resolution_label": self._get_resolution_label(width, height),
                    "fps": round(fps, 2),
                    "bitrate_kbps": round(int(video_stream.get("bit_rate", 0)) / 1000, 1) if video_stream.get("bit_rate") else None,
                    "pixel_format": video_stream.get("pix_fmt", ""),
                    "color_space": video_stream.get("color_space", ""),
                    "total_frames": int(video_stream.get("nb_frames", 0)) if video_stream.get("nb_frames", "N/A") != "N/A" else None,
                }

                analysis["quality_assessment"] = self._assess_quality(analysis["video"], analysis["file"])

            if audio_stream:
                analysis["audio"] = {
                    "codec": audio_stream.get("codec_name", ""),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "bitrate_kbps": round(int(audio_stream.get("bit_rate", 0)) / 1000, 1) if audio_stream.get("bit_rate") else None,
                }

            return analysis
        except Exception as e:
            raise ProcessingError(f"Quality analysis error: {e}")

    def compare_quality(
        self,
        reference_path: str | Path,
        distorted_path: str | Path,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compares quality between two videos.
        Uses PSNR and SSIM metrics.

        Args:
            reference_path: Reference (original) video
            distorted_path: Processed video to compare
            metrics: List of metrics to compute (['psnr', 'ssim'])
        """
        metrics = metrics or ["psnr", "ssim"]
        results = {}

        if "psnr" in metrics:
            results["psnr"] = self._calculate_psnr(reference_path, distorted_path)
        if "ssim" in metrics:
            results["ssim"] = self._calculate_ssim(reference_path, distorted_path)

        return {
            "reference": str(reference_path),
            "distorted": str(distorted_path),
            "metrics": results,
        }

    def generate_report(
        self, file_path: str | Path, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Generates a detailed quality report."""
        analysis = self.analyze(file_path)

        report = {
            **analysis,
            "recommendations": self._generate_recommendations(analysis),
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _calculate_psnr(self, ref: str | Path, dist: str | Path) -> dict[str, float] | None:
        try:
            cmd = [
                self.ffmpeg, "-i", str(ref), "-i", str(dist),
                "-lavfi", "psnr=stats_file=-", "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in result.stderr.split("\n"):
                if "average:" in line.lower():
                    parts = line.split()
                    for p in parts:
                        if p.startswith("average:"):
                            return {"average": float(p.split(":")[1])}
            return None
        except Exception:
            return None

    def _calculate_ssim(self, ref: str | Path, dist: str | Path) -> dict[str, float] | None:
        try:
            cmd = [
                self.ffmpeg, "-i", str(ref), "-i", str(dist),
                "-lavfi", "ssim=stats_file=-", "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in result.stderr.split("\n"):
                if "all:" in line.lower():
                    parts = line.split()
                    for p in parts:
                        if p.startswith("All:"):
                            return {"all": float(p.split(":")[1])}
            return None
        except Exception:
            return None

    @staticmethod
    def _get_resolution_label(width: int, height: int) -> str:
        labels = {
            (7680, 4320): "8K UHD",
            (3840, 2160): "4K UHD",
            (2560, 1440): "2K QHD",
            (1920, 1080): "Full HD",
            (1280, 720): "HD",
            (854, 480): "SD",
            (640, 360): "360p",
        }
        for (w, h), label in labels.items():
            if width >= w and height >= h:
                return label
        return f"{width}x{height}"

    @staticmethod
    def _assess_quality(video: dict, file: dict) -> dict[str, Any]:
        score = 0
        notes = []

        if video.get("width", 0) >= 1920:
            score += 25
            notes.append("Full HD or higher resolution")
        elif video.get("width", 0) >= 1280:
            score += 15
            notes.append("HD resolution")
        else:
            score += 5
            notes.append("Low resolution")

        fps = video.get("fps", 0)
        if fps >= 60:
            score += 25
            notes.append("High frame rate (60+ FPS)")
        elif fps >= 30:
            score += 20
            notes.append("Standard frame rate (30 FPS)")
        elif fps >= 24:
            score += 15
            notes.append("Cinematic frame rate (24 FPS)")
        else:
            score += 5
            notes.append("Low frame rate")

        codec = video.get("codec", "").lower()
        if codec in ("h265", "hevc"):
            score += 25
            notes.append("Modern codec (H.265/HEVC)")
        elif codec in ("h264", "avc"):
            score += 20
            notes.append("Standard codec (H.264/AVC)")
        elif codec in ("vp9",):
            score += 20
            notes.append("Web-friendly codec (VP9)")
        elif codec in ("av1",):
            score += 25
            notes.append("Cutting-edge codec (AV1)")
        else:
            score += 10
            notes.append(f"Codec: {codec}")

        bitrate = file.get("overall_bitrate_kbps", 0)
        if bitrate > 10000:
            score += 25
            notes.append("High bitrate")
        elif bitrate > 5000:
            score += 20
            notes.append("Good bitrate")
        elif bitrate > 2000:
            score += 15
            notes.append("Moderate bitrate")
        else:
            score += 5
            notes.append("Low bitrate")

        return {
            "score": min(score, 100),
            "grade": "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D",
            "notes": notes,
        }

    @staticmethod
    def _generate_recommendations(analysis: dict) -> list[str]:
        recs = []
        video = analysis.get("video", {})
        quality = analysis.get("quality_assessment", {})

        if video.get("codec", "").lower() not in ("h265", "hevc", "av1"):
            recs.append("Consider converting to H.265 or AV1 to reduce file size")
        if video.get("width", 0) > 1920 and analysis.get("file", {}).get("overall_bitrate_kbps", 0) < 5000:
            recs.append("Bitrate should be increased for high resolution")
        if video.get("fps", 0) > 30 and analysis.get("file", {}).get("overall_bitrate_kbps", 0) < 3000:
            recs.append("Bitrate is insufficient for high FPS")

        audio = analysis.get("audio", {})
        if audio and audio.get("bitrate_kbps", 0) and audio.get("bitrate_kbps", 0) < 96:
            recs.append("Audio bitrate is low; at least 128 kbps is recommended")

        if not recs:
            recs.append("Video quality looks adequate; no specific improvement is recommended")

        return recs
