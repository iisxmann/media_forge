"""
Whisper-based audio transcription module.
Speech-to-text, subtitle generation, language detection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mediaforge.core.base import ProcessingResult
from mediaforge.core.exceptions import AIModelError, TranscriptionError
from mediaforge.core.logger import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """
    OpenAI Whisper-based transcription class.
    Produces text and subtitles from audio/video files.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: str | None = None,
    ):
        """
        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v3)
            device: Compute device (auto, cpu, cuda)
            language: Target language (None=auto-detect, 'tr', 'en', etc.)
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import whisper
            import torch

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            logger.info(f"Loading Whisper model: {self.model_size} ({device})")
            self._model = whisper.load_model(self.model_size, device=device)
            logger.info("Whisper model loaded")
        except ImportError:
            raise AIModelError("openai-whisper package required: pip install openai-whisper")
        except Exception as e:
            raise AIModelError(f"Whisper model load error: {e}")

    def transcribe(
        self,
        file_path: str | Path,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribes an audio/video file to text.

        Args:
            language: Language code (None=auto, 'tr'=Turkish, 'en'=English)
            task: 'transcribe' (text) or 'translate' (translate to English)
            word_timestamps: Word-level timing
            initial_prompt: Hint text that gives the model context
        """
        self._load_model()
        start = time.time()

        try:
            options = {
                "task": task,
                "word_timestamps": word_timestamps,
            }
            if language or self.language:
                options["language"] = language or self.language
            if initial_prompt:
                options["initial_prompt"] = initial_prompt

            result = self._model.transcribe(str(file_path), **options)

            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": [
                    {
                        "id": seg["id"],
                        "start": round(seg["start"], 3),
                        "end": round(seg["end"], 3),
                        "text": seg["text"].strip(),
                        "confidence": round(seg.get("avg_logprob", 0), 4),
                        "words": seg.get("words", []) if word_timestamps else [],
                    }
                    for seg in result.get("segments", [])
                ],
                "duration_seconds": round(time.time() - start, 2),
            }
        except Exception as e:
            raise TranscriptionError(f"Transcription error: {e}")

    def generate_subtitles(
        self,
        file_path: str | Path,
        output_path: str | Path,
        format: str = "srt",
        language: str | None = None,
        max_line_length: int = 42,
    ) -> ProcessingResult:
        """
        Creates a subtitle file from an audio/video file.

        Args:
            format: Subtitle format ('srt' or 'vtt')
            max_line_length: Maximum characters per line
        """
        start = time.time()
        result = self.transcribe(file_path, language=language)
        segments = result["segments"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "srt":
            content = self._segments_to_srt(segments, max_line_length)
        elif format == "vtt":
            content = self._segments_to_vtt(segments, max_line_length)
        else:
            raise TranscriptionError(f"Unsupported format: {format}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return ProcessingResult(
            success=True,
            output_path=output_path,
            message=f"Subtitles created ({len(segments)} segments, {format})",
            duration_seconds=time.time() - start,
            details={
                "language": result["language"],
                "segments": len(segments),
                "format": format,
            },
        )

    def detect_language(self, file_path: str | Path) -> dict[str, Any]:
        """Detects the language of an audio file."""
        self._load_model()

        try:
            import whisper

            audio = whisper.load_audio(str(file_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self._model.device)

            _, probs = self._model.detect_language(mel)
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "detected_language": top_languages[0][0],
                "confidence": round(top_languages[0][1], 4),
                "top_5": [{"language": lang, "probability": round(prob, 4)} for lang, prob in top_languages],
            }
        except Exception as e:
            raise TranscriptionError(f"Language detection error: {e}")

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _segments_to_srt(self, segments: list[dict], max_len: int) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            text = self._wrap_text(seg["text"], max_len)
            lines.append(str(i))
            lines.append(
                f"{self._format_timestamp_srt(seg['start'])} --> {self._format_timestamp_srt(seg['end'])}"
            )
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _segments_to_vtt(self, segments: list[dict], max_len: int) -> str:
        lines = ["WEBVTT", ""]
        for seg in segments:
            text = self._wrap_text(seg["text"], max_len)
            lines.append(
                f"{self._format_timestamp_vtt(seg['start'])} --> {self._format_timestamp_vtt(seg['end'])}"
            )
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _wrap_text(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > max_len:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        return "\n".join(lines)
