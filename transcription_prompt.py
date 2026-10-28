"""Fit style examples and recent transcript context into Whisper's prompt.

Whisper treats its prompt as preceding transcript text, not instructions.  A
short, punctuated example can guide its writing style without a second model.
Both transcription engines keep at most 223 preceding-text tokens, so reserve
some of that space for the latest streaming context instead of letting a long
custom prompt or transcript silently displace the other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


DEFAULT_TRANSCRIPTION_PROMPT = (
    "Hello, how are you? I'm doing well. Let's review the details, "
    "make a decision, and move forward."
)
MAX_PROMPT_TOKENS = 223
PREVIOUS_TEXT_TOKENS = 64

Encode = Callable[[str], Sequence[int]]


def normalize_transcription_prompt(value: object) -> str:
    """Normalize a saved setting without turning nulls or objects into prose."""
    return value.strip() if isinstance(value, str) else ""


def _fit_text(text: str, budget: int, encode: Encode, *, tail: bool) -> str:
    """Keep a fitting prefix/suffix, cutting characters rather than token bytes.

    Decoding a truncated token sequence can split a Unicode character.  Search
    over character boundaries and measure each candidate with the native
    tokenizer instead.  BPE counts need not be perfectly monotonic; the search
    may leave a little unused space, but every retained candidate is measured.
    """
    if not text or budget <= 0:
        return ""
    if len(encode(text)) <= budget:
        return text

    low, high = 0, len(text)
    fitted = ""
    while low < high:
        length = (low + high + 1) // 2
        candidate = text[-length:].lstrip() if tail else text[:length].rstrip()
        if len(encode(candidate)) <= budget:
            low = length
            fitted = candidate
        else:
            high = length - 1
    return fitted


def build_transcription_prompt(
    style_prompt: object,
    previous_text: object,
    encode: Encode,
) -> str | None:
    """Combine an editable style/vocabulary example with recent dictation.

    ``encode`` must use the transcription model's native tokenizer and include
    any prefix the engine adds before tokenization.  With a style prompt, keep
    up to 64 tokens from the end of the prior transcript and fit the start of
    the style prompt into the remaining space.  Measure the joined string, not
    just its fragments, because tokenization can change at the separator.
    With an empty style prompt, the full budget is available to prior text.
    """
    style = normalize_transcription_prompt(style_prompt)
    previous = normalize_transcription_prompt(previous_text)
    if not style:
        return _fit_text(previous, MAX_PROMPT_TOKENS, encode, tail=True) or None
    if not previous:
        return _fit_text(style, MAX_PROMPT_TOKENS, encode, tail=False) or None

    history = _fit_text(previous, PREVIOUS_TEXT_TOKENS, encode, tail=True)
    if not history:
        return _fit_text(style, MAX_PROMPT_TOKENS, encode, tail=False) or None

    def encode_with_history(candidate: str) -> Sequence[int]:
        return encode(f"{candidate}\n{history}")

    fitted_style = _fit_text(style, MAX_PROMPT_TOKENS, encode_with_history, tail=False)
    if not fitted_style:
        return _fit_text(history, MAX_PROMPT_TOKENS, encode, tail=True) or None
    return f"{fitted_style}\n{history}"
