"""Apple-silicon GPU transcription through MLX (macOS only).

faster-whisper's CTranslate2 has no Metal backend, so on macOS it runs on the
CPU. ``mlx-whisper`` runs Whisper on the Apple GPU instead. This module keeps
that engine behind the same small surface the app uses for faster-whisper:
load a model directory once, then ``transcribe(audio, language) -> text``.

MLX models are separate downloads (``mlx-community`` repos) stored next to the
CTranslate2 models under ``models-mlx/<name>``.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import types

import numpy as np

# name -> (Hugging Face repo, approximate megabytes on disk)
MLX_MODELS = {
    "base.en": ("mlx-community/whisper-base.en-mlx", 145),
    "small.en": ("mlx-community/whisper-small.en-mlx", 480),
    "medium.en": ("mlx-community/whisper-medium.en-mlx", 1525),
    "large-v3": ("mlx-community/whisper-large-v3-mlx", 3085),
}

MLX_DIRNAME = "models-mlx"


def mlx_supported() -> bool:
    """True on an Apple-silicon Mac with the MLX runtime importable."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    if importlib.util.find_spec("mlx") is None:
        return False
    if importlib.util.find_spec("mlx_whisper") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


def mlx_model_is_complete(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and any(
        os.path.isfile(os.path.join(path, f))
        for f in ("weights.npz", "weights.safetensors")
    )


def _stub_word_timestamps() -> None:
    """Let ``mlx_whisper`` import without numba and scipy.

    ``mlx_whisper.transcribe`` imports its word-timestamp module eagerly, and
    that module needs numba (plus llvmlite) and scipy, about 200 MB that the
    app never uses because it doesn't ask for word timestamps. When they are
    absent, register a stand-in so the import succeeds.
    """
    if "mlx_whisper.timing" in sys.modules:
        return
    if importlib.util.find_spec("numba") and importlib.util.find_spec("scipy"):
        return
    stub = types.ModuleType("mlx_whisper.timing")

    def add_word_timestamps(*_args, **_kwargs):
        raise RuntimeError("word timestamps are not available in this build")

    stub.add_word_timestamps = add_word_timestamps
    sys.modules["mlx_whisper.timing"] = stub


def fetch_mlx_model(name: str, dest_root: str) -> str:
    """Download an MLX model into ``<dest_root>/models-mlx/<name>``."""
    if name not in MLX_MODELS:
        raise ValueError(f"unknown model '{name}' (choose from {list(MLX_MODELS)})")
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    root = os.path.join(dest_root, MLX_DIRNAME)
    out = os.path.join(root, name)
    if mlx_model_is_complete(out):
        return out
    os.makedirs(root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{name}-", dir=root)
    try:
        snapshot_download(
            MLX_MODELS[name][0],
            local_dir=staging,
            allow_patterns=["config.json", "weights.npz", "weights.safetensors"],
        )
        if not mlx_model_is_complete(staging):
            raise RuntimeError(f"MLX model '{name}' downloaded incompletely")
        # snapshot_download leaves a .cache folder of metadata; not needed.
        shutil.rmtree(os.path.join(staging, ".cache"), ignore_errors=True)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.replace(staging, out)
        staging = None
    finally:
        if staging and os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return out


def trim_to_speech(audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """Keep only the speech, using faster-whisper's Silero VAD.

    Mirrors ``vad_filter=True`` on the CPU path so silence doesn't make Whisper
    invent text ("Thank you.") and long pauses don't slow decoding.
    """
    from faster_whisper.vad import collect_chunks, get_speech_timestamps

    chunks = get_speech_timestamps(audio, sampling_rate=sampling_rate)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pieces, _ = collect_chunks(audio, chunks, sampling_rate=sampling_rate)
    return np.concatenate(pieces).astype(np.float32, copy=False)


class MlxWhisper:
    """A loaded MLX Whisper model."""

    def __init__(self, model_dir: str):
        _stub_word_timestamps()
        import mlx_whisper  # noqa: F401  (import errors surface here)

        self.model_dir = model_dir
        # mlx_whisper loads weights lazily on first use; do it now with a
        # second of silence so the first dictation isn't slowed by loading.
        self._decode(np.zeros(16000, dtype=np.float32), language="en")

    def _decode(
        self, audio: np.ndarray, language: str | None, prompt: str | None = None
    ) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_dir,
            language=language or None,
            verbose=None,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        return str(result.get("text", "")).strip()

    def transcribe(
        self, audio: np.ndarray, language: str | None, prompt: str | None = None
    ) -> str:
        """``prompt``: preceding text, for consistent style across pieces."""
        speech = trim_to_speech(np.asarray(audio, dtype=np.float32))
        if speech.size < 1600:  # under 0.1 s of speech
            return ""
        return self._decode(speech, language, prompt)
