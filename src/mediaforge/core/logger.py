"""MediaForge logging. Structured logging built on Loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_dir: str | Path = "./logs",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configures logging for console and file output."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=log_format, level=level, colorize=True)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path / "mediaforge_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
    )

    logger.add(
        str(log_path / "errors_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
    )


def get_logger(name: str = "mediaforge") -> logger:
    """Returns a logger bound to the given module name."""
    return logger.bind(module=name)
