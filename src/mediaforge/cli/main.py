"""
Media Forge CLI.
Access all media processing features from the command line.

Usage:
    python -m mediaforge --help
    python -m mediaforge image --help
    python -m mediaforge video trim input.mp4 output.mp4 --start 10 --end 30
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _print_result(result):
    if result.success:
        console.print(f"[bold green]OK:[/] {result.message}")
        if result.output_path:
            console.print(f"  Output: {result.output_path}")
        if result.duration_seconds:
            console.print(f"  Duration: {result.duration_seconds:.2f}s")
    else:
        console.print(f"[bold red]Error:[/] {result.message}")
        sys.exit(1)


def _print_json(data):
    console.print_json(json.dumps(data, default=str, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════
#  ROOT
# ═══════════════════════════════════════════════════════════════════════

@click.group()
@click.version_option(version="1.0.0", prog_name="Media Forge")
def cli():
    """Media Forge — Comprehensive Media Processing Tool"""
    pass


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def image():
    """Image processing commands"""
    pass


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--width", "-w", type=int, help="Target width")
@click.option("--height", "-h", type=int, help="Target height")
@click.option("--quality", "-q", type=int, default=95, help="Quality (1-100)")
def resize(input_path, output_path, width, height, quality):
    """Resize an image."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().resize(input_path, output_path, width=width, height=height, quality=quality))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--left", type=int, required=True)
@click.option("--top", type=int, required=True)
@click.option("--right", type=int, required=True)
@click.option("--bottom", type=int, required=True)
def crop(input_path, output_path, left, top, right, bottom):
    """Crop an image by coordinates."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().crop(input_path, output_path, left=left, top=top, right=right, bottom=bottom))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--angle", "-a", type=float, required=True, help="Rotation angle in degrees")
@click.option("--expand", is_flag=True, default=True, help="Expand canvas to fit rotated image")
def rotate(input_path, output_path, angle, expand):
    """Rotate an image."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().rotate(input_path, output_path, angle=angle, expand=expand))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--direction", "-d", type=click.Choice(["horizontal", "vertical"]), default="horizontal")
def flip(input_path, output_path, direction):
    """Flip an image horizontally or vertically."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().flip(input_path, output_path, direction=direction))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.argument("target_format")
@click.option("--quality", "-q", type=int, default=95)
def convert(input_path, output_path, target_format, quality):
    """Convert image format (jpg, png, webp, bmp, etc.)."""
    from mediaforge.image.converter import ImageConverter
    _print_result(ImageConverter().convert(input_path, output_path, target_format, quality))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--text", "-t", required=True, help="Watermark text")
@click.option("--position", "-p", default="bottom-right",
              type=click.Choice(["top-left", "top-right", "bottom-left", "bottom-right", "center"]))
@click.option("--opacity", "-o", type=float, default=0.5)
@click.option("--font-size", "-s", type=int, default=24)
def watermark(input_path, output_path, text, position, opacity, font_size):
    """Add a text watermark to an image."""
    from mediaforge.image.watermark import WatermarkEngine
    _print_result(WatermarkEngine().add_text_watermark(
        input_path, output_path, text=text, position=position,
        opacity=opacity, font_size=font_size,
    ))


@image.command(name="filter")
@click.argument("input_path")
@click.argument("output_path")
@click.argument("filter_name")
def apply_filter(input_path, output_path, filter_name):
    """Apply a filter (blur, sharpen, sepia, grayscale, etc.). Use 'image filters' to list all."""
    from mediaforge.image.filters import ImageFilterEngine
    _print_result(ImageFilterEngine().apply_filter(input_path, output_path, filter_name))


@image.command()
def filters():
    """List all available image filters."""
    from mediaforge.image.filters import ImageFilterEngine
    table = Table(title="Available Filters")
    table.add_column("Filter", style="cyan")
    table.add_column("Description", style="green")
    for f in ImageFilterEngine().list_filters():
        table.add_row(f["name"], f["description"])
    console.print(table)


@image.command()
@click.argument("input_path")
def metadata(input_path):
    """Show image metadata (EXIF, dimensions, etc.)."""
    from mediaforge.image.metadata import ImageMetadataManager
    _print_json(ImageMetadataManager().read_metadata(input_path))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, default=1.0, help="Brightness factor (1.0 = no change)")
def brightness(input_path, output_path, factor):
    """Adjust image brightness."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().adjust_brightness(input_path, output_path, factor=factor))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, default=1.0, help="Contrast factor (1.0 = no change)")
def contrast(input_path, output_path, factor):
    """Adjust image contrast."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().adjust_contrast(input_path, output_path, factor=factor))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
def grayscale(input_path, output_path):
    """Convert image to grayscale."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().grayscale(input_path, output_path))


@image.command()
@click.argument("input_path")
@click.argument("output_path")
def invert(input_path, output_path):
    """Invert image colors (negative)."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().invert(input_path, output_path))


@image.command(name="auto-enhance")
@click.argument("input_path")
@click.argument("output_path")
def auto_enhance(input_path, output_path):
    """Auto-enhance image (contrast + sharpness + color)."""
    from mediaforge.image.processor import ImageProcessor
    _print_result(ImageProcessor().auto_enhance(input_path, output_path))


@image.command()
@click.argument("input_path")
def histogram(input_path):
    """Analyze image histogram (per-channel stats)."""
    from mediaforge.image.effects import ImageEffects
    _print_json(ImageEffects().histogram_analysis(input_path))


@image.command(name="color-palette")
@click.argument("input_path")
@click.option("--colors", "-n", type=int, default=8, help="Number of dominant colors to extract")
def color_palette(input_path, colors):
    """Extract dominant color palette from image."""
    from mediaforge.image.effects import ImageEffects
    _print_json(ImageEffects().extract_color_palette(input_path, n_colors=colors))


@image.command(name="blur-detect")
@click.argument("input_path")
def blur_detect(input_path):
    """Detect if an image is blurry (Laplacian variance)."""
    from mediaforge.image.effects import ImageEffects
    result = ImageEffects().detect_blur(input_path)
    _print_json(result)


@image.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--width", "-w", type=int, default=256)
@click.option("--height", "-h", type=int, default=256)
@click.option("--mode", "-m", type=click.Choice(["fit", "fill", "stretch", "pad"]), default="fit")
def thumbnail(input_path, output_path, width, height, mode):
    """Generate a thumbnail."""
    from mediaforge.image.thumbnail import ThumbnailGenerator
    _print_result(ThumbnailGenerator().generate(input_path, output_path, width=width, height=height, mode=mode))


@image.command()
@click.argument("output_path")
@click.argument("images", nargs=-1, required=True)
@click.option("--columns", "-c", type=int, default=3, help="Grid columns")
@click.option("--spacing", "-s", type=int, default=10, help="Spacing between images")
def collage(output_path, images, columns, spacing):
    """Create a grid collage from multiple images. Pass image paths after output path."""
    from mediaforge.image.collage import CollageBuilder
    _print_result(CollageBuilder().create_grid(list(images), output_path, columns=columns, spacing=spacing))


@image.command()
def formats():
    """List supported image formats."""
    from mediaforge.image.converter import ImageConverter
    for fmt in ImageConverter().supported_formats:
        console.print(f"  {fmt}")


# ═══════════════════════════════════════════════════════════════════════
#  VIDEO
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def video():
    """Video processing commands"""
    pass


@video.command()
@click.argument("input_path")
def info(input_path):
    """Show video metadata (resolution, duration, codec, etc.)."""
    from mediaforge.video.processor import VideoProcessor
    vi = VideoProcessor().get_video_info(input_path)
    table = Table(title="Video Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Format", vi.format)
    table.add_row("Resolution", f"{vi.width}x{vi.height}")
    table.add_row("Duration", f"{vi.duration:.1f}s")
    table.add_row("FPS", f"{vi.fps:.1f}")
    table.add_row("Codec", vi.codec or "N/A")
    table.add_row("Size", f"{vi.size_mb:.1f} MB")
    table.add_row("Bitrate", f"{(vi.bitrate or 0) / 1000:.0f} kbps")
    console.print(table)


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.argument("target_format")
@click.option("--quality", "-q", default="medium", help="Quality preset (low/medium/high)")
def convert(input_path, output_path, target_format, quality):
    """Convert video format (mp4, avi, mkv, mov, webm, etc.)."""
    from mediaforge.video.converter import VideoConverter
    _print_result(VideoConverter().convert(input_path, output_path, target_format, quality_preset=quality))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--crf", type=int, default=28, help="CRF value (lower = better quality, 18-28 recommended)")
@click.option("--target-size", type=float, default=None, help="Target file size in MB")
def compress(input_path, output_path, crf, target_size):
    """Compress a video file."""
    from mediaforge.video.converter import VideoConverter
    _print_result(VideoConverter().compress(input_path, output_path, target_size_mb=target_size, crf=crf))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", type=float, required=True, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, help="End time in seconds")
@click.option("--duration", "-d", type=float, default=None, help="Duration in seconds")
def trim(input_path, output_path, start, end, duration):
    """Trim/cut a video segment."""
    from mediaforge.video.editor import VideoEditor
    _print_result(VideoEditor().trim(input_path, output_path, start, end, duration))


@video.command()
@click.argument("output_path")
@click.argument("inputs", nargs=-1, required=True)
def concat(output_path, inputs):
    """Concatenate multiple videos into one. Pass video paths after output path."""
    from mediaforge.video.editor import VideoEditor
    _print_result(VideoEditor().concat(list(inputs), output_path))


@video.command(name="extract-audio")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--format", "-f", "fmt", default="mp3", help="Audio format (mp3, wav, aac, flac)")
def extract_audio(input_path, output_path, fmt):
    """Extract audio track from a video."""
    from mediaforge.video.processor import VideoProcessor
    _print_result(VideoProcessor().extract_audio(input_path, output_path, format=fmt))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", type=float, default=0, help="Start time in seconds")
@click.option("--duration", "-d", type=float, default=5, help="Duration in seconds")
@click.option("--fps", type=int, default=10, help="GIF frames per second")
@click.option("--width", "-w", type=int, default=480, help="GIF width")
def gif(input_path, output_path, start, duration, fps, width):
    """Create a GIF from video segment."""
    from mediaforge.video.processor import VideoProcessor
    _print_result(VideoProcessor().create_gif(input_path, output_path, start=start, duration=duration, fps=fps, width=width))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--text", "-t", required=True, help="Overlay text")
@click.option("--x", type=int, default=10)
@click.option("--y", type=int, default=10)
@click.option("--font-size", type=int, default=24)
@click.option("--color", default="white")
def text_overlay(input_path, output_path, text, x, y, font_size, color):
    """Add text overlay to video."""
    from mediaforge.video.editor import VideoEditor
    _print_result(VideoEditor().add_text_overlay(input_path, output_path, text=text, x=x, y=y, font_size=font_size, color=color))


@video.command(name="image-overlay")
@click.argument("input_path")
@click.argument("overlay_path")
@click.argument("output_path")
@click.option("--x", type=int, default=10)
@click.option("--y", type=int, default=10)
@click.option("--opacity", type=float, default=1.0)
def image_overlay(input_path, overlay_path, output_path, x, y, opacity):
    """Add image overlay (logo/watermark) to video."""
    from mediaforge.video.editor import VideoEditor
    _print_result(VideoEditor().add_image_overlay(input_path, overlay_path, output_path, x=x, y=y, opacity=opacity))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--segment-duration", "-d", type=float, required=True, help="Each segment duration in seconds")
def split(input_path, output_path, segment_duration):
    """Split video into equal-length segments."""
    from mediaforge.video.editor import VideoEditor
    _print_result(VideoEditor().split(input_path, output_path, segment_duration=segment_duration))


@video.command(name="filter")
@click.argument("input_path")
@click.argument("output_path")
@click.argument("filter_name")
def video_filter(input_path, output_path, filter_name):
    """Apply a video filter (grayscale, sepia, blur, etc.). Use 'video filters' to list all."""
    from mediaforge.video.effects import VideoEffects
    _print_result(VideoEffects().apply_filter(input_path, output_path, filter_name))


@video.command()
def video_filters():
    """List available video filters."""
    from mediaforge.video.effects import VideoEffects
    table = Table(title="Available Video Filters")
    table.add_column("Filter", style="cyan")
    table.add_column("Description", style="green")
    for f in VideoEffects().list_filters():
        table.add_row(f["name"], f.get("description", ""))
    console.print(table)


@video.command(name="slow-motion")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, default=2.0, help="Slowdown factor (2.0 = half speed)")
def slow_motion(input_path, output_path, factor):
    """Create slow motion video."""
    from mediaforge.video.effects import VideoEffects
    _print_result(VideoEffects().slow_motion(input_path, output_path, factor=factor))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, default=10.0, help="Speed factor (10.0 = 10x faster)")
def timelapse(input_path, output_path, factor):
    """Create timelapse from video."""
    from mediaforge.video.effects import VideoEffects
    _print_result(VideoEffects().timelapse(input_path, output_path, factor=factor))


@video.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--fade-in", type=float, default=0, help="Fade in duration (seconds)")
@click.option("--fade-out", type=float, default=0, help="Fade out duration (seconds)")
def fade(input_path, output_path, fade_in, fade_out):
    """Add fade in/out to video."""
    from mediaforge.video.effects import VideoEffects
    _print_result(VideoEffects().picture_fade(input_path, output_path, fade_in=fade_in, fade_out=fade_out))


@video.command(name="burn-subs")
@click.argument("input_path")
@click.argument("subtitle_path")
@click.argument("output_path")
def burn_subtitles(input_path, subtitle_path, output_path):
    """Burn (hardcode) subtitles into video."""
    from mediaforge.video.subtitles import SubtitleManager
    _print_result(SubtitleManager().burn_subtitles(input_path, subtitle_path, output_path))


@video.command(name="embed-subs")
@click.argument("input_path")
@click.argument("subtitle_path")
@click.argument("output_path")
def embed_subtitles(input_path, subtitle_path, output_path):
    """Embed subtitles as a selectable track (soft subs)."""
    from mediaforge.video.subtitles import SubtitleManager
    _print_result(SubtitleManager().embed_subtitles(input_path, subtitle_path, output_path))


@video.command(name="extract-subs")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--stream", "-s", type=int, default=0, help="Subtitle stream index")
def extract_subtitles(input_path, output_path, stream):
    """Extract subtitles from video to file."""
    from mediaforge.video.subtitles import SubtitleManager
    _print_result(SubtitleManager().extract_subtitles(input_path, output_path, stream_index=stream))


@video.command(name="convert-subs")
@click.argument("input_path")
@click.argument("output_path")
def convert_subtitles(input_path, output_path):
    """Convert subtitle format (SRT <-> VTT)."""
    from mediaforge.video.subtitles import SubtitleManager
    _print_result(SubtitleManager().convert_subtitle(input_path, output_path))


@video.command(name="detect-scenes")
@click.argument("input_path")
@click.option("--threshold", "-t", type=float, default=30.0, help="Scene change threshold")
def detect_scenes(input_path, threshold):
    """Detect scene changes in video."""
    from mediaforge.video.scenes import SceneDetector
    scenes = SceneDetector().detect_scenes(input_path, threshold=threshold)
    summary = SceneDetector().get_scene_summary(scenes)
    _print_json(summary)


@video.command(name="extract-frame")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--time", "-t", type=float, required=True, help="Timestamp in seconds")
def extract_frame(input_path, output_path, time):
    """Extract a single frame at a specific timestamp."""
    from mediaforge.video.thumbnail import VideoThumbnailExtractor
    _print_result(VideoThumbnailExtractor().extract_frame(input_path, output_path, timestamp=time))


@video.command(name="sprite-sheet")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--columns", "-c", type=int, default=5, help="Grid columns")
@click.option("--count", "-n", type=int, default=20, help="Number of frames")
def sprite_sheet(input_path, output_path, columns, count):
    """Create a sprite sheet (preview grid) from video."""
    from mediaforge.video.thumbnail import VideoThumbnailExtractor
    _print_result(VideoThumbnailExtractor().create_sprite_sheet(input_path, output_path, columns=columns, count=count))


@video.command()
@click.argument("input_path")
def quality(input_path):
    """Run video quality analysis."""
    from mediaforge.video.quality import VideoQualityAnalyzer
    _print_json(VideoQualityAnalyzer().generate_report(input_path))


# ═══════════════════════════════════════════════════════════════════════
#  AUDIO
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def audio():
    """Audio processing commands"""
    pass


@audio.command()
@click.argument("input_path")
def info(input_path):
    """Show audio file metadata."""
    from mediaforge.audio.processor import AudioProcessor
    _print_json(AudioProcessor().get_audio_info(input_path))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
@click.argument("target_format")
@click.option("--bitrate", "-b", default="192k", help="Audio bitrate (e.g. 128k, 192k, 320k)")
def convert(input_path, output_path, target_format, bitrate):
    """Convert audio format (mp3, wav, ogg, flac, aac, etc.)."""
    from mediaforge.audio.converter import AudioConverter
    _print_result(AudioConverter().convert(input_path, output_path, target_format, bitrate=bitrate))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--target-loudness", "-l", type=float, default=-16.0, help="Target loudness in LUFS")
def normalize(input_path, output_path, target_loudness):
    """Normalize audio loudness (EBU R128)."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().normalize(input_path, output_path, target_loudness=target_loudness))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", type=float, required=True, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, help="End time in seconds")
@click.option("--duration", "-d", type=float, default=None, help="Duration in seconds")
def trim(input_path, output_path, start, end, duration):
    """Trim/cut an audio segment."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().trim(input_path, output_path, start, end, duration))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, required=True, help="Volume factor (1.0 = no change, 2.0 = double)")
def volume(input_path, output_path, factor):
    """Change audio volume."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().change_volume(input_path, output_path, factor=factor))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--factor", "-f", type=float, required=True, help="Speed factor (0.5 = half, 2.0 = double)")
def speed(input_path, output_path, factor):
    """Change audio playback speed."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().change_speed(input_path, output_path, factor=factor))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def mono(input_path, output_path):
    """Convert audio to mono."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().to_mono(input_path, output_path))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def stereo(input_path, output_path):
    """Convert audio to stereo."""
    from mediaforge.audio.processor import AudioProcessor
    _print_result(AudioProcessor().to_stereo(input_path, output_path))


@audio.command(name="fade-in")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--duration", "-d", type=float, required=True, help="Fade duration in seconds")
def fade_in(input_path, output_path, duration):
    """Apply fade-in effect."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().fade_in(input_path, output_path, duration=duration))


@audio.command(name="fade-out")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--duration", "-d", type=float, required=True, help="Fade duration in seconds")
def fade_out(input_path, output_path, duration):
    """Apply fade-out effect."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().fade_out(input_path, output_path, duration=duration))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def echo(input_path, output_path):
    """Add echo effect."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().echo(input_path, output_path))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def reverb(input_path, output_path):
    """Add reverb effect."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().reverb(input_path, output_path))


@audio.command(name="noise-reduce")
@click.argument("input_path")
@click.argument("output_path")
def noise_reduce(input_path, output_path):
    """Reduce background noise."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().noise_reduction(input_path, output_path))


@audio.command(name="bass-boost")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--gain", "-g", type=float, default=10.0, help="Bass gain in dB")
def bass_boost(input_path, output_path, gain):
    """Boost bass frequencies."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().bass_boost(input_path, output_path, gain=gain))


@audio.command(name="treble-boost")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--gain", "-g", type=float, default=5.0, help="Treble gain in dB")
def treble_boost(input_path, output_path, gain):
    """Boost treble frequencies."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().treble_boost(input_path, output_path, gain=gain))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def reverse(input_path, output_path):
    """Reverse audio playback."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().reverse(input_path, output_path))


@audio.command(name="remove-silence")
@click.argument("input_path")
@click.argument("output_path")
def remove_silence(input_path, output_path):
    """Remove silent sections from audio."""
    from mediaforge.audio.effects import AudioEffects
    _print_result(AudioEffects().silence_remove(input_path, output_path))


@audio.command(name="concat")
@click.argument("output_path")
@click.argument("inputs", nargs=-1, required=True)
def audio_concat(output_path, inputs):
    """Concatenate multiple audio files. Pass file paths after output path."""
    from mediaforge.audio.mixer import AudioMixer
    _print_result(AudioMixer().concatenate(list(inputs), output_path))


@audio.command()
@click.argument("output_path")
@click.argument("inputs", nargs=-1, required=True)
def mix(output_path, inputs):
    """Mix multiple audio files together. Pass file paths after output path."""
    from mediaforge.audio.mixer import AudioMixer
    _print_result(AudioMixer().mix(list(inputs), output_path))


@audio.command()
@click.argument("input_path")
def loudness(input_path):
    """Analyze audio loudness (LUFS, peak, range)."""
    from mediaforge.audio.analyzer import AudioAnalyzer
    _print_json(AudioAnalyzer().analyze_loudness(input_path))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def waveform(input_path, output_path):
    """Generate waveform image (PNG)."""
    from mediaforge.audio.analyzer import AudioAnalyzer
    _print_result(AudioAnalyzer().generate_waveform(input_path, output_path))


@audio.command()
@click.argument("input_path")
@click.argument("output_path")
def spectrogram(input_path, output_path):
    """Generate spectrogram image (PNG)."""
    from mediaforge.audio.analyzer import AudioAnalyzer
    _print_result(AudioAnalyzer().generate_spectrogram(input_path, output_path))


@audio.command()
@click.argument("input_path")
def bpm(input_path):
    """Detect BPM (tempo) of audio."""
    from mediaforge.audio.analyzer import AudioAnalyzer
    result = AudioAnalyzer().detect_bpm(input_path)
    console.print(f"[bold green]BPM:[/] {result.get('bpm', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════
#  AI
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def ai():
    """AI-powered commands (requires pre-downloaded models)"""
    pass


@ai.command()
@click.argument("input_path")
@click.option("--language", "-l", default=None, help="Language code (en, tr, de, etc.)")
@click.option("--model", "-m", default="base", help="Model size (tiny/base/small/medium/large)")
def transcribe(input_path, language, model):
    """Transcribe audio/video to text (Whisper)."""
    from mediaforge.ai.transcription import WhisperTranscriber
    result = WhisperTranscriber(model_size=model).transcribe(input_path, language=language)
    console.print(f"\n[bold green]Language:[/] {result['language']}")
    console.print(f"[bold green]Text:[/]\n{result['text']}")


@ai.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--format", "-f", "fmt", default="srt", type=click.Choice(["srt", "vtt"]),
              help="Subtitle format")
@click.option("--language", "-l", default=None, help="Language code")
def subtitles(input_path, output_path, fmt, language):
    """Generate subtitle file from audio/video (Whisper)."""
    from mediaforge.ai.transcription import WhisperTranscriber
    _print_result(WhisperTranscriber().generate_subtitles(input_path, output_path, format=fmt, language=language))


@ai.command(name="detect-faces")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--blur", is_flag=True, default=False, help="Blur detected faces instead of drawing boxes")
def detect_faces(input_path, output_path, blur):
    """Detect faces in an image — draw boxes or blur them."""
    from mediaforge.ai.face_detection import FaceDetector
    detector = FaceDetector()
    if blur:
        _print_result(detector.blur_faces(input_path, output_path))
    else:
        _print_result(detector.draw_faces(input_path, output_path))


@ai.command(name="detect-objects")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--confidence", "-c", type=float, default=0.25, help="Min confidence threshold")
def detect_objects(input_path, output_path, confidence):
    """Detect objects in image (YOLOv8) and draw annotations."""
    from mediaforge.ai.object_detection import ObjectDetector
    _print_result(ObjectDetector().detect_and_draw(input_path, output_path, confidence=confidence))


@ai.command(name="remove-bg")
@click.argument("input_path")
@click.argument("output_path")
def remove_bg(input_path, output_path):
    """Remove background from image (U2Net)."""
    from mediaforge.ai.background_removal import BackgroundRemover
    _print_result(BackgroundRemover().remove_background(input_path, output_path))


@ai.command()
@click.argument("input_path")
@click.option("--engine", "-e", type=click.Choice(["tesseract", "easyocr"]), default="tesseract",
              help="OCR engine to use")
def ocr(input_path, engine):
    """Extract text from image (OCR)."""
    from mediaforge.ai.ocr import OCREngine
    result = OCREngine(engine=engine).extract_text(input_path)
    console.print(f"\n[bold green]Extracted text:[/]\n{result['text']}")


@ai.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--scale", "-s", type=int, default=4, help="Upscale factor (2, 3, 4, or 8)")
def upscale(input_path, output_path, scale):
    """Upscale image resolution (Super Resolution)."""
    from mediaforge.ai.super_resolution import SuperResolution
    _print_result(SuperResolution().upscale(input_path, output_path, scale=scale))


@ai.command(name="style-transfer")
@click.argument("content_path")
@click.argument("style_path")
@click.argument("output_path")
def style_transfer(content_path, style_path, output_path):
    """Apply neural style transfer (content + style images)."""
    from mediaforge.ai.style_transfer import StyleTransfer
    _print_result(StyleTransfer().apply_style(content_path, style_path, output_path))


# ═══════════════════════════════════════════════════════════════════════
#  SERVER
# ═══════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", "-p", type=int, default=8000, help="Bind port")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes")
def serve(host, port, reload):
    """Start the REST API server."""
    import uvicorn
    console.print(f"[bold green]Starting Media Forge API...[/]")
    console.print(f"  URL:   http://localhost:{port}")
    console.print(f"  Docs:  http://localhost:{port}/docs")
    uvicorn.run("mediaforge.api.app:create_app", host=host, port=port, reload=reload, factory=True)


# ═══════════════════════════════════════════════════════════════════════
#  SYSTEM
# ═══════════════════════════════════════════════════════════════════════

@cli.command()
def check():
    """Check system dependencies (FFmpeg, Tesseract, models, etc.)."""
    from mediaforge.utils.dependency_checker import print_dependency_report
    print_dependency_report()


if __name__ == "__main__":
    cli()
