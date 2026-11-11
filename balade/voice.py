"""Voice questions: speech to text with Whisper, running locally.

Two backends, tried in this order (WHISPER_BACKEND=auto):

1. faster-whisper: fast, light, decodes audio with PyAV (no ffmpeg needed).
2. transformers: Whisper through PyTorch, which the embeddings already use. This is the fallback
   when Windows refuses to load faster-whisper's libraries ("WinError 4551": Smart App Control
   blocks DLLs that are not signed). The recording is decoded with Python's own `wave` module,
   so nothing else has to be loaded.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import wave

from balade import config

log = logging.getLogger("balade")
SAMPLE_RATE = 16000   # what Whisper expects


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def is_available() -> bool:
    return _has("faster_whisper") or (_has("transformers") and _has("torch"))


def decode_wav(audio: bytes):
    """WAV bytes -> (mono float32 samples between -1 and 1, sample rate)."""
    import numpy as np

    with wave.open(io.BytesIO(audio)) as file:
        channels, width, rate = file.getnchannels(), file.getsampwidth(), file.getframerate()
        frames = file.readframes(file.getnframes())
    if width != 2:
        raise ValueError(f"Unsupported recording format ({8 * width}-bit audio).")
    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def resample(samples, rate: int, target: int = SAMPLE_RATE):
    """Linear resampling, enough for speech. Does nothing when the rate is already right."""
    import numpy as np

    if rate == target or len(samples) == 0:
        return samples
    positions = np.linspace(0, len(samples) - 1, int(len(samples) * target / rate))
    return np.interp(positions, np.arange(len(samples)), samples).astype("float32")


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, model_size: str, device: str):
        from faster_whisper import WhisperModel

        # int8 on CPU is fast enough for a short question; float16 on an NVIDIA GPU.
        self.model = WhisperModel(model_size, device=device,
                                  compute_type="float16" if device == "cuda" else "int8")

    def transcribe(self, audio: bytes) -> str:
        segments, _info = self.model.transcribe(io.BytesIO(audio), vad_filter=True, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_size: str, device: str):
        import torch
        from transformers import pipeline

        use_gpu = device == "cuda" and torch.cuda.is_available()
        self.pipeline = pipeline("automatic-speech-recognition", model=f"openai/whisper-{model_size}",
                                 device=0 if use_gpu else -1, chunk_length_s=30)

    def transcribe(self, audio: bytes) -> str:
        samples, rate = decode_wav(audio)
        samples = resample(samples, rate)
        if len(samples) < SAMPLE_RATE // 4:           # less than a quarter of a second: nothing said
            return ""
        result = self.pipeline({"raw": samples, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()


BACKENDS = {"faster-whisper": FasterWhisperBackend, "transformers": TransformersBackend}


class Transcriber:
    """Speech to text. With backend="auto", falls back to the next backend when one cannot be
    loaded or fails on its first use (missing package, or libraries blocked by Windows)."""

    def __init__(self, model_size: str = config.WHISPER_MODEL, device: str = config.WHISPER_DEVICE,
                 backend: str = config.WHISPER_BACKEND):
        self.model_size = model_size
        self.device = device
        self.candidates = list(BACKENDS) if backend == "auto" else [backend]
        self.backend = None

    def _load_next(self) -> None:
        errors = []
        while self.candidates:
            name = self.candidates.pop(0)
            try:
                self.backend = BACKENDS[name](self.model_size, self.device)
                log.info("Speech recognition: using %s", name)
                return
            except Exception as error:               # not installed, or blocked by the system
                log.warning("Speech recognition: %s is unavailable (%s)", name, error)
                errors.append(f"{name}: {error}")
        self.backend = None
        raise RuntimeError("No speech recognition backend could be loaded. " + " | ".join(errors))

    def transcribe(self, audio: bytes) -> str:
        """Text of a recorded question (language detected automatically). Empty if nothing was said."""
        if self.backend is None:
            self._load_next()
        try:
            return self.backend.transcribe(audio)
        except OSError as error:
            # Typically WinError 4551: the backend loaded, but one of its libraries is blocked
            # when first used. Try the next backend with the same recording.
            if not self.candidates:
                raise
            log.warning("Speech recognition: %s failed (%s), trying the next backend",
                        self.backend.name, error)
            self.backend = None
            return self.transcribe(audio)
