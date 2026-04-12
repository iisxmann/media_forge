"""MediaForge configuration management."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class ImageSettings(BaseSettings):
    default_quality: int = 95
    default_format: str = "png"
    max_resolution: int = 8192
    thumbnail_size: tuple[int, int] = (256, 256)
    supported_formats: list[str] = [
        "jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp", "svg", "ico", "heic"
    ]


class VideoSettings(BaseSettings):
    default_codec: str = "libx264"
    default_audio_codec: str = "aac"
    default_format: str = "mp4"
    default_fps: int = 30
    default_bitrate: str = "5000k"
    max_resolution: int = 7680
    supported_formats: list[str] = [
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "3gp"
    ]


class AudioSettings(BaseSettings):
    default_format: str = "mp3"
    default_bitrate: str = "192k"
    default_sample_rate: int = 44100
    supported_formats: list[str] = [
        "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a"
    ]


class WhisperSettings(BaseSettings):
    model_size: str = "base"
    language: str = "tr"
    device: str = "auto"


class FaceDetectionSettings(BaseSettings):
    confidence_threshold: float = 0.5
    model: str = "haar"


class ObjectDetectionSettings(BaseSettings):
    model: str = "yolov8n"
    confidence_threshold: float = 0.25


class SuperResolutionSettings(BaseSettings):
    scale_factor: int = 4
    model: str = "EDSR"


class AISettings(BaseSettings):
    whisper: WhisperSettings = WhisperSettings()
    face_detection: FaceDetectionSettings = FaceDetectionSettings()
    object_detection: ObjectDetectionSettings = ObjectDetectionSettings()
    super_resolution: SuperResolutionSettings = SuperResolutionSettings()


class WatermarkSettings(BaseSettings):
    default_opacity: float = 0.5
    default_position: str = "bottom-right"
    default_margin: int = 10
    font_size: int = 24


class BatchSettings(BaseSettings):
    max_workers: int = 4
    chunk_size: int = 10


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 500
    cors_origins: list[str] = ["*"]


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
    rotation: str = "10 MB"
    retention: str = "30 days"


class Settings(BaseSettings):
    """Main configuration class. Holds all nested settings."""

    app_name: str = "MediaForge"
    version: str = "1.0.0"
    debug: bool = False
    env: str = Field(default="development", alias="MEDIAFORGE_ENV")

    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    cache_dir: Path = Path("./cache")
    log_dir: Path = Path("./logs")
    model_dir: Path = Path("./models")

    image: ImageSettings = ImageSettings()
    video: VideoSettings = VideoSettings()
    audio: AudioSettings = AudioSettings()
    ai: AISettings = AISettings()
    watermark: WatermarkSettings = WatermarkSettings()
    batch: BatchSettings = BatchSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()

    model_config = {"env_prefix": "MEDIAFORGE_", "env_file": ".env", "extra": "ignore"}

    def ensure_directories(self) -> None:
        """Creates required directories."""
        for dir_path in [self.output_dir, self.temp_dir, self.cache_dir, self.log_dir, self.model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Settings:
        """Loads settings from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**cls._flatten_yaml(data))

    @staticmethod
    def _flatten_yaml(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flattens nested YAML data into a flat dict."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict) and full_key in (
                "image", "video", "audio", "ai", "watermark", "batch", "api", "logging"
            ):
                result[full_key] = value
            elif isinstance(value, dict) and full_key == "paths":
                for pkey, pval in value.items():
                    result[f"{pkey}_dir"] = pval
            elif isinstance(value, dict) and full_key == "app":
                for akey, aval in value.items():
                    if akey == "name":
                        result["app_name"] = aval
                    else:
                        result[akey] = aval
            else:
                result[full_key] = value
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a singleton Settings instance."""
    config_path = Path("config/settings.yaml")
    settings = Settings.from_yaml(config_path) if config_path.exists() else Settings()
    settings.ensure_directories()
    return settings
