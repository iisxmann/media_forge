"""File-based cache system. Avoids reprocessing media that has already been handled."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class FileCache:
    """File-based cache system."""

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        max_size_mb: int = 1024,
        ttl_seconds: int = 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.metadata_file = self.cache_dir / "_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def _generate_key(self, file_path: str | Path, operation: str, params: dict) -> str:
        """Builds a unique cache key from file path, operation, and parameters."""
        path = Path(file_path)
        stat = path.stat()
        raw = f"{path.absolute()}|{stat.st_size}|{stat.st_mtime}|{operation}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, file_path: str | Path, operation: str, params: dict) -> Path | None:
        """Returns a cached file path, or None if missing or expired."""
        key = self._generate_key(file_path, operation, params)
        if key not in self._metadata:
            return None

        entry = self._metadata[key]
        cached_path = Path(entry["cached_path"])

        if not cached_path.exists():
            del self._metadata[key]
            self._save_metadata()
            return None

        if time.time() - entry["timestamp"] > self.ttl_seconds:
            self.invalidate(file_path, operation, params)
            return None

        logger.debug(f"Cache hit: {key[:12]}... -> {cached_path}")
        return cached_path

    def put(
        self, file_path: str | Path, operation: str, params: dict, result_path: str | Path
    ) -> Path:
        """Stores an operation result in the cache."""
        self._enforce_size_limit()

        key = self._generate_key(file_path, operation, params)
        result = Path(result_path)
        cached_path = self.cache_dir / f"{key}{result.suffix}"

        shutil.copy2(result, cached_path)

        self._metadata[key] = {
            "original_path": str(file_path),
            "cached_path": str(cached_path),
            "operation": operation,
            "params": params,
            "timestamp": time.time(),
            "size_bytes": cached_path.stat().st_size,
        }
        self._save_metadata()

        logger.debug(f"Cache put: {key[:12]}... -> {cached_path}")
        return cached_path

    def invalidate(self, file_path: str | Path, operation: str, params: dict) -> bool:
        """Removes a specific cache entry."""
        key = self._generate_key(file_path, operation, params)
        if key in self._metadata:
            cached_path = Path(self._metadata[key]["cached_path"])
            if cached_path.exists():
                cached_path.unlink()
            del self._metadata[key]
            self._save_metadata()
            return True
        return False

    def clear(self) -> int:
        """Clears the entire cache. Returns the number of files removed."""
        count = 0
        for key, entry in list(self._metadata.items()):
            cached_path = Path(entry["cached_path"])
            if cached_path.exists():
                cached_path.unlink()
                count += 1
        self._metadata.clear()
        self._save_metadata()
        logger.info(f"Cache cleared: {count} file(s) removed")
        return count

    def get_size_mb(self) -> float:
        """Returns total cache size in megabytes."""
        total = sum(
            Path(entry["cached_path"]).stat().st_size
            for entry in self._metadata.values()
            if Path(entry["cached_path"]).exists()
        )
        return total / (1024 * 1024)

    def _enforce_size_limit(self) -> None:
        """Enforces the cache size limit by evicting oldest entries (LRU-style)."""
        while self.get_size_mb() > self.max_size_mb and self._metadata:
            oldest_key = min(self._metadata, key=lambda k: self._metadata[k]["timestamp"])
            entry = self._metadata[oldest_key]
            cached_path = Path(entry["cached_path"])
            if cached_path.exists():
                cached_path.unlink()
            del self._metadata[oldest_key]

        self._save_metadata()

    @property
    def stats(self) -> dict[str, Any]:
        """Returns cache statistics."""
        return {
            "total_entries": len(self._metadata),
            "total_size_mb": round(self.get_size_mb(), 2),
            "max_size_mb": self.max_size_mb,
            "ttl_seconds": self.ttl_seconds,
        }
