"""Dependency checker module. Verifies presence of external tools."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def check_dependencies() -> dict[str, Any]:
    """Checks all dependencies and returns a report."""
    results = {}

    external_tools = {
        "ffmpeg": "Video/audio processing (required)",
        "ffprobe": "Media analysis (required)",
        "tesseract": "OCR - text extraction (optional)",
    }

    for tool, desc in external_tools.items():
        path = shutil.which(tool)
        version = None
        if path:
            try:
                result = subprocess.run([tool, "-version"], capture_output=True, text=True, timeout=5)
                first_line = result.stdout.split("\n")[0] if result.stdout else ""
                version = first_line.strip()
            except Exception:
                version = "unknown"

        results[tool] = {
            "available": path is not None,
            "path": path,
            "version": version,
            "description": desc,
        }

    python_packages = {
        "PIL": ("Pillow", "Image processing"),
        "cv2": ("opencv-python", "Computer vision"),
        "numpy": ("numpy", "Numerical computing"),
        "moviepy": ("moviepy", "Video editing"),
        "whisper": ("openai-whisper", "Speech transcription"),
        "torch": ("torch", "Deep learning"),
        "ultralytics": ("ultralytics", "YOLO object detection"),
        "rembg": ("rembg", "Background removal"),
        "pytesseract": ("pytesseract", "Tesseract OCR"),
        "easyocr": ("easyocr", "EasyOCR"),
        "fastapi": ("fastapi", "REST API"),
        "click": ("click", "CLI"),
        "loguru": ("loguru", "Logging"),
        "librosa": ("librosa", "Audio analysis"),
    }

    for module, (package, desc) in python_packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            results[package] = {"available": True, "version": version, "description": desc}
        except ImportError:
            results[package] = {"available": False, "version": None, "description": desc}

    return results


def print_dependency_report() -> None:
    """Prints the dependency report as a table."""
    deps = check_dependencies()

    table = Table(title="MediaForge Dependency Report")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Version", style="dim")
    table.add_column("Description")

    for name, info in deps.items():
        status = "[green]Installed[/green]" if info["available"] else "[red]Missing[/red]"
        version = info.get("version") or "-"
        table.add_row(name, status, str(version), info["description"])

    console.print(table)

    missing = [n for n, i in deps.items() if not i["available"]]
    if missing:
        console.print(f"\n[yellow]Missing packages:[/yellow] {', '.join(missing)}")
        console.print("[dim]Install with: pip install <package_name>[/dim]")
    else:
        console.print("\n[bold green]All dependencies are installed![/bold green]")
