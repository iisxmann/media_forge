"""MediaForge-specific exception classes."""


class MediaForgeError(Exception):
    """Base class for all MediaForge errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FileNotFoundError(MediaForgeError):
    """Raised when a file cannot be found."""
    pass


class UnsupportedFormatError(MediaForgeError):
    """Raised when an unsupported file format is used."""

    def __init__(self, format: str, supported_formats: list[str] | None = None):
        self.format = format
        self.supported_formats = supported_formats or []
        message = f"Unsupported format: {format}"
        if self.supported_formats:
            message += f". Supported formats: {', '.join(self.supported_formats)}"
        super().__init__(message, {"format": format, "supported_formats": self.supported_formats})


class InvalidResolutionError(MediaForgeError):
    """Raised when an invalid resolution value is used."""
    pass


class ProcessingError(MediaForgeError):
    """General errors that occur during processing."""
    pass


class ConversionError(MediaForgeError):
    """Errors that occur during format conversion."""
    pass


class CodecError(MediaForgeError):
    """Errors related to codecs."""
    pass


class AudioExtractionError(MediaForgeError):
    """Errors that occur during audio extraction."""
    pass


class WatermarkError(MediaForgeError):
    """Errors that occur while adding a watermark."""
    pass


class FilterError(MediaForgeError):
    """Errors that occur while applying a filter."""
    pass


class AIModelError(MediaForgeError):
    """Errors that occur while loading or running an AI model."""
    pass


class TranscriptionError(MediaForgeError):
    """Errors that occur during transcription."""
    pass


class OCRError(MediaForgeError):
    """Errors that occur during OCR."""
    pass


class BatchProcessingError(MediaForgeError):
    """Errors that occur during batch processing."""

    def __init__(self, message: str, failed_items: list[dict] | None = None):
        self.failed_items = failed_items or []
        super().__init__(message, {"failed_items": self.failed_items})


class PipelineError(MediaForgeError):
    """Errors that occur while running a pipeline."""
    pass


class StreamingError(MediaForgeError):
    """Errors that occur during real-time streaming."""
    pass


class CacheError(MediaForgeError):
    """Errors that occur during cache operations."""
    pass


class ValidationError(MediaForgeError):
    """Errors that occur during input validation."""
    pass


class ConfigurationError(MediaForgeError):
    """Configuration errors."""
    pass


class DependencyError(MediaForgeError):
    """Missing dependency errors (e.g. ffmpeg, tesseract)."""
    pass
