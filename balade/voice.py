"""Voice questions: speech to text with faster-whisper (optional dependency).

faster-whisper decodes audio with PyAV, so unlike openai-whisper it does not need ffmpeg
installed on the system (which is what blocked voice input on Windows in the first version).
"""
from __future__ import annotations

import importlib.util
import io

from balade import config


def is_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


class Transcriber:
    def __init__(self, model_size: str = config.WHISPER_MODEL, device: str = config.WHISPER_DEVICE):
        from faster_whisper import WhisperModel

        # int8 on CPU is fast enough for a short question; float16 on an NVIDIA GPU.
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes) -> str:
        """Text of a recorded question (language detected automatically). Empty if nothing was said."""
        segments, _info = self.model.transcribe(io.BytesIO(audio), vad_filter=True, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()
