"""Transcribe dictation in pieces while the user is still speaking.

Whisper needs a whole stretch of audio, so true word-by-word streaming would
cost accuracy. Instead, while recording, this watches the audio for natural
pauses and transcribes everything up to each pause in the background. When the
user stops, only the speech after the last pause is left, so the wait no
longer grows with the length of the dictation.

Each piece is given the text so far as Whisper's prompt, which keeps
punctuation and capitalisation consistent across the cuts.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

# Transcribe(audio, prompt) -> text
TranscribeFn = Callable[[np.ndarray, Optional[str]], str]
# Vad(audio) -> [{"start": sample, "end": sample}, ...] speech regions
VadFn = Callable[[np.ndarray], List[dict]]


def silero_vad(pause_s: float, sample_rate: int = 16000) -> VadFn:
    """faster-whisper's Silero VAD, merging gaps shorter than ``pause_s``."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # A region only closes after a full ``pause_s`` of silence; a region still
    # in progress runs to the end of the audio. Light padding keeps word
    # edges without pushing closed regions back to the end.
    options = VadOptions(
        min_silence_duration_ms=int(pause_s * 1000), speech_pad_ms=200
    )

    def vad(audio: np.ndarray) -> List[dict]:
        return get_speech_timestamps(audio, options, sampling_rate=sample_rate)

    return vad


class StreamingTranscriber:
    """Runs alongside a recording and transcribes it at each pause.

    ``get_audio`` returns everything recorded so far. ``transcribe`` is called
    from one background thread at a time; callers that share the model with
    other threads must serialise it themselves.
    """

    def __init__(
        self,
        get_audio: Callable[[], np.ndarray],
        transcribe: TranscribeFn,
        vad: VadFn,
        sample_rate: int = 16000,
        pause_s: float = 0.7,
        poll_s: float = 0.4,
        min_piece_s: float = 1.0,
        prompt_chars: int = 200,
    ):
        self._get_audio = get_audio
        self._transcribe = transcribe
        self._vad = vad
        self._sr = sample_rate
        self._pause = int(pause_s * sample_rate)
        self._poll_s = poll_s
        self._min_piece = int(min_piece_s * sample_rate)
        self._prompt_chars = prompt_chars
        self._committed = 0  # samples already transcribed (or dropped as silence)
        self._parts: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WhisperTypeStreaming"
        )
        self._thread.start()

    def cancel(self) -> None:
        """Stop watching without transcribing anything further."""
        self._stop.set()

    def finish(self, audio: np.ndarray) -> str:
        """Stop watching, transcribe what is left of ``audio``, return all text."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            # A background piece failed; redo the rest in one go rather than
            # returning a transcript with a hole in it.
            print(f"[stream] background piece failed ({self._error}); "
                  f"finishing the remainder in one pass")
        tail = audio[self._committed:]
        if tail.size:
            t0 = time.time()
            text = self._transcribe(tail, self._prompt())
            print(f"[stream] last piece {tail.size / self._sr:.1f}s "
                  f"→ {time.time() - t0:.1f}s")
            if text:
                self._parts.append(text)
        return self.text

    @property
    def text(self) -> str:
        return " ".join(p.strip() for p in self._parts if p.strip())

    # -- background loop -----------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop.wait(self._poll_s):
                self.step()
        except BaseException as e:  # keep recording; finish() covers the rest
            self._error = e

    def step(self) -> None:
        """Transcribe up to the latest finished pause, if there is one."""
        audio = self._get_audio()
        pending = audio[self._committed:]
        if pending.size < self._min_piece + self._pause:
            return
        regions = self._vad(pending)
        if not regions:
            # Only silence so far: drop all but the last pause's worth, so a
            # long quiet start isn't re-scanned forever.
            drop = pending.size - self._pause
            if drop > 0:
                self._committed += drop
            return
        # The VAD only closes a region after a full pause; a region still in
        # progress runs to the end of the audio. So any region ending clearly
        # before the end is finished speech followed by a pause.
        open_margin = self._sr // 4
        cut = None
        for region in regions:
            if region["end"] <= pending.size - open_margin:
                cut = region["end"]
        if cut is None or cut < self._min_piece:
            return
        piece = pending[:cut]
        t0 = time.time()
        text = self._transcribe(piece, self._prompt())
        print(f"[stream] piece {len(self._parts) + 1}: {piece.size / self._sr:.1f}s "
              f"→ {time.time() - t0:.1f}s {text!r}")
        if text:
            self._parts.append(text)
        self._committed += cut

    def _prompt(self) -> Optional[str]:
        text = self.text
        return text[-self._prompt_chars:] if text else None
