"""Image processing tests."""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from mediaforge.image.processor import ImageProcessor
from mediaforge.image.converter import ImageConverter
from mediaforge.image.filters import ImageFilterEngine
from mediaforge.image.metadata import ImageMetadataManager
from mediaforge.image.thumbnail import ThumbnailGenerator


@pytest.fixture
def sample_image(tmp_path):
    """Creates a sample image for tests."""
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def processor():
    return ImageProcessor()


@pytest.fixture
def converter():
    return ImageConverter()


class TestImageProcessor:
    def test_resize(self, processor, sample_image, tmp_path):
        output = tmp_path / "resized.png"
        result = processor.resize(sample_image, output, width=400)
        assert result.success
        assert output.exists()
        img = Image.open(output)
        assert img.size[0] == 400

    def test_crop(self, processor, sample_image, tmp_path):
        output = tmp_path / "cropped.png"
        result = processor.crop(sample_image, output, 100, 100, 500, 400)
        assert result.success
        img = Image.open(output)
        assert img.size == (400, 300)

    def test_rotate(self, processor, sample_image, tmp_path):
        output = tmp_path / "rotated.png"
        result = processor.rotate(sample_image, output, 90)
        assert result.success
        assert output.exists()

    def test_flip_horizontal(self, processor, sample_image, tmp_path):
        output = tmp_path / "flipped.png"
        result = processor.flip(sample_image, output, "horizontal")
        assert result.success

    def test_grayscale(self, processor, sample_image, tmp_path):
        output = tmp_path / "gray.png"
        result = processor.grayscale(sample_image, output)
        assert result.success
        img = Image.open(output)
        assert img.mode == "L"

    def test_auto_enhance(self, processor, sample_image, tmp_path):
        output = tmp_path / "enhanced.png"
        result = processor.auto_enhance(sample_image, output)
        assert result.success


class TestImageConverter:
    def test_png_to_jpg(self, converter, sample_image, tmp_path):
        output = tmp_path / "converted.jpg"
        result = converter.convert(sample_image, output, "jpg")
        assert result.success
        assert output.exists()

    def test_png_to_webp(self, converter, sample_image, tmp_path):
        output = tmp_path / "converted.webp"
        result = converter.convert(sample_image, output, "webp")
        assert result.success

    def test_supported_formats(self, converter):
        formats = converter.get_supported_formats()
        assert "jpg" in formats
        assert "png" in formats
        assert "webp" in formats


class TestImageFilter:
    def test_blur_filter(self, sample_image, tmp_path):
        engine = ImageFilterEngine()
        output = tmp_path / "blurred.png"
        result = engine.apply_filter(sample_image, output, "blur")
        assert result.success

    def test_sepia_filter(self, sample_image, tmp_path):
        engine = ImageFilterEngine()
        output = tmp_path / "sepia.png"
        result = engine.apply_filter(sample_image, output, "sepia")
        assert result.success

    def test_list_filters(self):
        engine = ImageFilterEngine()
        filters = engine.list_filters()
        assert len(filters) > 10

    def test_chain_filters(self, sample_image, tmp_path):
        engine = ImageFilterEngine()
        output = tmp_path / "chained.png"
        result = engine.apply_chain(sample_image, output, [
            {"name": "blur", "params": {"radius": 3}},
            {"name": "sepia"},
        ])
        assert result.success


class TestImageMetadata:
    def test_read_metadata(self, sample_image):
        manager = ImageMetadataManager()
        meta = manager.read_metadata(sample_image)
        assert "file" in meta
        assert meta["file"]["size"]["width"] == 800

    def test_strip_metadata(self, sample_image, tmp_path):
        manager = ImageMetadataManager()
        output = tmp_path / "stripped.png"
        result = manager.strip_metadata(sample_image, output)
        assert result.exists()


class TestThumbnail:
    def test_generate_thumbnail(self, sample_image, tmp_path):
        gen = ThumbnailGenerator()
        output = tmp_path / "thumb.jpg"
        result = gen.generate(sample_image, output, size=(128, 128), mode="fill")
        assert result.success
        img = Image.open(output)
        assert img.size == (128, 128)

    def test_preset_sizes(self):
        gen = ThumbnailGenerator()
        presets = gen.get_available_presets()
        assert "youtube" in presets
        assert "favicon" in presets
