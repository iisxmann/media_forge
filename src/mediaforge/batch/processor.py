"""
Batch processing module.
Process multiple files in parallel or sequentially.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import BatchProcessingError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BatchResult:
    """Data class holding batch processing results."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    results: list[ProcessingResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total * 100) if self.total > 0 else 0


class BatchProcessor:
    """Batch file processing class."""

    def __init__(self, max_workers: int = 4, show_progress: bool = True):
        """
        Args:
            max_workers: Number of parallel workers
            show_progress: Show progress bar
        """
        self.max_workers = max_workers
        self.show_progress = show_progress

    def process(
        self,
        input_paths: list[str | Path],
        processor_func: Callable,
        output_dir: str | Path | None = None,
        parallel: bool = True,
        **kwargs,
    ) -> BatchResult:
        """
        Processes a list of files with the given function.

        Args:
            input_paths: List of files to process
            processor_func: Function called for each file
                Signature: func(input_path, output_path, **kwargs) -> ProcessingResult
            output_dir: Output directory
            parallel: Parallel processing (True) or sequential (False)
        """
        start = time.time()
        batch_result = BatchResult(total=len(input_paths))

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if parallel:
            self._process_parallel(input_paths, processor_func, output_dir, batch_result, **kwargs)
        else:
            self._process_sequential(input_paths, processor_func, output_dir, batch_result, **kwargs)

        batch_result.duration_seconds = time.time() - start
        logger.info(
            f"Batch processing complete: {batch_result.successful}/{batch_result.total} succeeded "
            f"({batch_result.duration_seconds:.1f}s)"
        )
        return batch_result

    def process_directory(
        self,
        input_dir: str | Path,
        processor_func: Callable,
        output_dir: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = False,
        parallel: bool = True,
        **kwargs,
    ) -> BatchResult:
        """
        Processes all files in a directory.

        Args:
            input_dir: Input directory
            extensions: File extensions to filter (e.g. ['.jpg', '.png'])
            recursive: Include subdirectories
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise BatchProcessingError(f"Directory not found: {input_dir}")

        pattern = "**/*" if recursive else "*"
        files = []
        for f in input_dir.glob(pattern):
            if f.is_file():
                if extensions is None or f.suffix.lower() in extensions:
                    files.append(f)

        if not files:
            logger.warning(f"No files to process: {input_dir}")
            return BatchResult()

        logger.info(f"{len(files)} files found: {input_dir}")
        return self.process(files, processor_func, output_dir, parallel, **kwargs)

    def _process_parallel(
        self,
        input_paths: list[str | Path],
        processor_func: Callable,
        output_dir: Path | None,
        batch_result: BatchResult,
        **kwargs,
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for path in input_paths:
                path = Path(path)
                out_path = (output_dir / path.name) if output_dir else None
                future = executor.submit(processor_func, path, out_path, **kwargs)
                futures[future] = path

            iterator = as_completed(futures)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Processing")

            for future in iterator:
                path = futures[future]
                try:
                    result = future.result()
                    batch_result.results.append(result)
                    if result.success:
                        batch_result.successful += 1
                    else:
                        batch_result.failed += 1
                        batch_result.errors.append({"file": str(path), "error": result.message})
                except Exception as e:
                    batch_result.failed += 1
                    batch_result.errors.append({"file": str(path), "error": str(e)})

    def _process_sequential(
        self,
        input_paths: list[str | Path],
        processor_func: Callable,
        output_dir: Path | None,
        batch_result: BatchResult,
        **kwargs,
    ) -> None:
        iterator = input_paths
        if self.show_progress:
            iterator = tqdm(iterator, desc="Processing")

        for path in iterator:
            path = Path(path)
            out_path = (output_dir / path.name) if output_dir else None
            try:
                result = processor_func(path, out_path, **kwargs)
                batch_result.results.append(result)
                if result.success:
                    batch_result.successful += 1
                else:
                    batch_result.failed += 1
                    batch_result.errors.append({"file": str(path), "error": result.message})
            except Exception as e:
                batch_result.failed += 1
                batch_result.errors.append({"file": str(path), "error": str(e)})
