"""Progress tracking system."""

from __future__ import annotations

import time
from typing import Any, Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


class ProgressTracker:
    """Rich-based progress tracking class."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self._callbacks: list[Callable] = []

    def track_iterable(self, iterable, total: int | None = None, description: str | None = None):
        """Shows a progress bar over an iterable."""
        desc = description or self.description
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(desc, total=total or len(iterable))
            for item in iterable:
                yield item
                progress.advance(task)

    def on_progress(self, callback: Callable[[float, str], None]) -> None:
        """Adds a progress callback: callback(percentage, message)."""
        self._callbacks.append(callback)

    def update(self, percentage: float, message: str = "") -> None:
        """Updates progress and invokes callbacks."""
        for cb in self._callbacks:
            cb(percentage, message)
