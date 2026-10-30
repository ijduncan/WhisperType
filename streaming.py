"""Transcribe dictation in pieces while the user is still speaking.

Whisper needs a whole stretch of audio, so true word-by-word streaming would
cost accuracy. Instead, while recording, this watches the audio for natural
pauses and transcribes everything up to each pause in the background. When the
user stops, only the speech after the last pause is left, so the wait no
longer grows with the length of the dictation.

Each piece is given the text so far as Whisper's prompt, which keeps
punctuation and capitalisation consistent across the cuts.

Whisper treats the end of its audio as the end of what was said, so a piece
cut at a mid-sentence pause comes back as a finished sentence ("and provide.")
and the next would start with a capital. To undo that, the prompt leaves off
the full stop a cut may have added. Whisper then starts the next piece in
lowercase if it carries on the sentence, and the full stop is dropped; a
capital means a new sentence, and the full stop stays.
"""

from __future__ import annotations

import re
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
        self._text = ""
        self._pieces = 0
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
            self._add(text)
        if self._pieces > 1:
            self._text = finish_sentence(self._text)
        return self.text

    @property
    def text(self) -> str:
        return self._text

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
        print(f"[stream] piece {self._pieces + 1}: {piece.size / self._sr:.1f}s "
              f"→ {time.time() - t0:.1f}s {text!r}")
        self._add(text)
        self._committed += cut

    def _add(self, text: str) -> None:
        if text and text.strip():
            self._text = join_piece(self._text, text)
            self._pieces += 1

    def _prompt(self) -> Optional[str]:
        text = _CUT_STOP.sub("", self._text)
        return text[-self._prompt_chars:] if text else None


# A full stop or ellipsis that may only be there because the audio was cut.
_CUT_STOP = re.compile(r"(\.\.\.|…|\.)$")
_LEADING_ELLIPSIS = re.compile(r"^(\.\.\.|…)\s*")
# Capitalised whatever the sentence, so they don't show a new one started.
_ALWAYS_CAPITAL = {"I", "I'm", "I'll", "I've", "I'd"}


def join_piece(text: str, piece: str) -> str:
    """Append a newly transcribed piece, fixing the punctuation at the cut."""
    piece = _LEADING_ELLIPSIS.sub("", piece.strip())
    if not text:
        return piece
    if not piece:
        return text
    first = piece.split()[0]
    if piece[0].islower():
        # Carries on the sentence: the full stop came from the cut.
        text = _CUT_STOP.sub("", text)
        if text.endswith(("?", "!")):
            piece = piece[0].upper() + piece[1:]
    elif piece[0].isupper() and first not in _ALWAYS_CAPITAL:
        text = finish_sentence(text)
    return f"{text} {piece}"


def finish_sentence(text: str) -> str:
    """End ``text`` with a full stop unless it already ends a sentence."""
    text = text.rstrip()
    if text.endswith(("...", "…")):
        return _CUT_STOP.sub(".", text)
    if text and text[-1].isalnum():
        return text + "."
    return text
