"""Base abstract classes. All processor, converter, and filter classes inherit from these."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from mediaforge.core.logger import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Position(str, Enum):
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


@dataclass
class MediaInfo:
    """Dataclass holding metadata for a media file."""
    path: Path
    format: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    duration: float | None = None  # seconds
    fps: float | None = None
    bitrate: int | None = None
    codec: str | None = None
    audio_codec: str | None = None
    channels: int | None = None
    sample_rate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> tuple[int, int] | None:
        if self.width and self.height:
            return (self.width, self.height)
        return None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def aspect_ratio(self) -> float | None:
        if self.width and self.height and self.height > 0:
            return self.width / self.height
        return None


@dataclass
class ProcessingResult:
    """Dataclass holding the outcome of a processing run."""
    success: bool
    output_path: Path | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    input_info: MediaInfo | None = None
    output_info: MediaInfo | None = None


class BaseProcessor(ABC):
    """Base class for all media processors."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def process(self, input_path: str | Path, output_path: str | Path, **kwargs) -> ProcessingResult:
        """Processes a media file."""
        ...

    def validate_input(self, input_path: str | Path) -> Path:
        """Validates the input file."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def prepare_output(self, output_path: str | Path) -> Path:
        """Prepares the output path and creates parent directories as needed."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_media_info(self, file_path: str | Path) -> MediaInfo:
        """Returns file metadata. Subclasses may override."""
        path = Path(file_path)
        return MediaInfo(
            path=path,
            format=path.suffix.lstrip(".").lower(),
            size_bytes=path.stat().st_size,
        )


class BaseConverter(ABC):
    """Base class for all format converters."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def convert(
        self, input_path: str | Path, output_path: str | Path, target_format: str, **kwargs
    ) -> ProcessingResult:
        """Converts a file to the target format."""
        ...

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Returns supported formats."""
        ...

    def is_format_supported(self, format: str) -> bool:
        """Returns whether the given format is supported."""
        return format.lower().lstrip(".") in self.get_supported_formats()


class BaseFilter(ABC):
    """Base class for all filters."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def apply(self, data: Any, **kwargs) -> Any:
        """Applies the filter. The type of ``data`` depends on the subclass (e.g. numpy array, PIL Image)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
