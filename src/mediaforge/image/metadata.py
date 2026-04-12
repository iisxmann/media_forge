"""
Image metadata manager.
Read, write, delete, and edit EXIF data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from mediaforge.core.logger import get_logger
from mediaforge.core.exceptions import ProcessingError

logger = get_logger(__name__)


class ImageMetadataManager:
    """Manager for image metadata (EXIF/IPTC/XMP)."""

    def read_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Reads all metadata from the image."""
        path = Path(file_path)
        if not path.exists():
            raise ProcessingError(f"File not found: {path}")

        try:
            img = Image.open(path)
            metadata = {
                "file": {
                    "name": path.name,
                    "format": img.format,
                    "mode": img.mode,
                    "size": {"width": img.size[0], "height": img.size[1]},
                    "file_size_bytes": path.stat().st_size,
                },
                "exif": self._read_exif(img),
                "info": {k: str(v) for k, v in img.info.items() if isinstance(v, (str, int, float, bytes))},
            }
            return metadata
        except Exception as e:
            raise ProcessingError(f"Metadata read error: {e}")

    def read_exif(self, file_path: str | Path) -> dict[str, Any]:
        """Reads EXIF data only."""
        img = Image.open(file_path)
        return self._read_exif(img)

    def read_gps(self, file_path: str | Path) -> dict[str, Any] | None:
        """Reads GPS information and converts to coordinates."""
        exif = self.read_exif(file_path)
        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None

        try:
            gps = {}
            for key, val in gps_info.items():
                tag = GPSTAGS.get(key, key)
                gps[tag] = val

            if "GPSLatitude" in gps and "GPSLongitude" in gps:
                lat = self._convert_gps_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
                lon = self._convert_gps_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
                gps["latitude_decimal"] = lat
                gps["longitude_decimal"] = lon
                gps["google_maps_url"] = f"https://maps.google.com/?q={lat},{lon}"

            return gps
        except Exception:
            return None

    def strip_metadata(
        self, input_path: str | Path, output_path: str | Path, quality: int = 95
    ) -> Path:
        """Creates a clean copy with all metadata removed. For privacy protection."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(input_path)
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))
            clean.save(output_path, quality=quality)
            logger.info(f"Metadata stripped: {input_path.name}")
            return output_path
        except Exception as e:
            raise ProcessingError(f"Metadata strip error: {e}")

    def copy_metadata(
        self, source_path: str | Path, target_path: str | Path, output_path: str | Path
    ) -> Path:
        """Copies metadata from one image to another."""
        try:
            import piexif

            source_exif = piexif.load(str(source_path))
            target_img = Image.open(target_path)

            exif_bytes = piexif.dump(source_exif)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            target_img.save(output_path, exif=exif_bytes)

            return output_path
        except ImportError:
            raise ProcessingError("piexif package required: pip install piexif")
        except Exception as e:
            raise ProcessingError(f"Metadata copy error: {e}")

    def export_metadata(self, file_path: str | Path, output_json: str | Path) -> Path:
        """Exports metadata to a JSON file."""
        metadata = self.read_metadata(file_path)
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        return output_json

    def compare_metadata(self, file1: str | Path, file2: str | Path) -> dict[str, Any]:
        """Compares metadata between two images."""
        meta1 = self.read_metadata(file1)
        meta2 = self.read_metadata(file2)

        differences = {}
        all_keys = set(list(meta1.get("exif", {}).keys()) + list(meta2.get("exif", {}).keys()))

        for key in all_keys:
            val1 = meta1.get("exif", {}).get(key)
            val2 = meta2.get("exif", {}).get(key)
            if val1 != val2:
                differences[key] = {"file1": str(val1), "file2": str(val2)}

        return {
            "file1": str(file1),
            "file2": str(file2),
            "differences": differences,
            "total_differences": len(differences),
        }

    def _read_exif(self, img: Image.Image) -> dict[str, Any]:
        exif_data = img.getexif()
        if not exif_data:
            return {}

        result = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            try:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                result[tag_name] = value
            except Exception:
                result[tag_name] = str(value)

        return result

    @staticmethod
    def _convert_gps_to_decimal(coords, ref: str) -> float:
        degrees = float(coords[0])
        minutes = float(coords[1])
        seconds = float(coords[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
