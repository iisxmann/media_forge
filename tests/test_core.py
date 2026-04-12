"""Core module tests."""

import pytest
from pathlib import Path

from mediaforge.core.config import Settings, get_settings
from mediaforge.core.exceptions import (
    MediaForgeError,
    UnsupportedFormatError,
    ValidationError,
)
from mediaforge.core.validators import (
    validate_quality,
    validate_opacity,
    validate_fps,
    validate_aspect_ratio,
    get_format_from_path,
)
from mediaforge.core.base import MediaInfo, ProcessingResult, MediaType, Position
from mediaforge.core.cache import FileCache


class TestConfig:
    def test_default_settings(self):
        settings = Settings()
        assert settings.app_name == "MediaForge"
        assert settings.version == "1.0.0"
        assert "jpg" in settings.image.supported_formats

    def test_image_settings(self):
        settings = Settings()
        assert settings.image.default_quality == 95
        assert settings.image.max_resolution == 8192

    def test_video_settings(self):
        settings = Settings()
        assert "mp4" in settings.video.supported_formats
        assert settings.video.default_fps == 30


class TestExceptions:
    def test_base_error(self):
        error = MediaForgeError("test error", {"key": "value"})
        assert str(error) == "test error"
        assert error.details == {"key": "value"}

    def test_unsupported_format(self):
        error = UnsupportedFormatError("xyz", ["jpg", "png"])
        assert "xyz" in str(error)
        assert error.supported_formats == ["jpg", "png"]


class TestValidators:
    def test_validate_quality_valid(self):
        assert validate_quality(50) is True
        assert validate_quality(1) is True
        assert validate_quality(100) is True

    def test_validate_quality_invalid(self):
        with pytest.raises(ValidationError):
            validate_quality(0)
        with pytest.raises(ValidationError):
            validate_quality(101)

    def test_validate_opacity(self):
        assert validate_opacity(0.5) is True
        with pytest.raises(ValidationError):
            validate_opacity(1.5)

    def test_validate_fps(self):
        assert validate_fps(30) is True
        with pytest.raises(ValidationError):
            validate_fps(0)

    def test_validate_aspect_ratio(self):
        assert validate_aspect_ratio("16:9") == (16, 9)
        assert validate_aspect_ratio("4:3") == (4, 3)
        with pytest.raises(ValidationError):
            validate_aspect_ratio("invalid")

    def test_get_format_from_path(self):
        assert get_format_from_path("test.jpg") == "jpg"
        assert get_format_from_path("test.MP4") == "mp4"


class TestBaseModels:
    def test_media_info(self):
        info = MediaInfo(
            path=Path("test.mp4"), format="mp4",
            size_bytes=1024 * 1024 * 50,
            width=1920, height=1080,
        )
        assert info.resolution == (1920, 1080)
        assert info.size_mb == 50.0
        assert info.aspect_ratio == pytest.approx(16 / 9, rel=0.01)

    def test_processing_result(self):
        result = ProcessingResult(success=True, message="OK")
        assert result.success
        assert result.message == "OK"

    def test_media_type(self):
        assert MediaType.IMAGE == "image"
        assert MediaType.VIDEO == "video"

    def test_position(self):
        assert Position.BOTTOM_RIGHT == "bottom-right"
        assert Position.CENTER == "center"


class TestCache:
    def test_cache_operations(self, tmp_path):
        cache = FileCache(cache_dir=tmp_path / "cache", ttl_seconds=60)

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        result = cache.get(test_file, "resize", {"width": 100})
        assert result is None

        cache.put(test_file, "resize", {"width": 100}, test_file)
        result = cache.get(test_file, "resize", {"width": 100})
        assert result is not None

        assert cache.stats["total_entries"] == 1

        cache.clear()
        assert cache.stats["total_entries"] == 0
