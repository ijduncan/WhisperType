import unittest

from transcription_prompt import (
    DEFAULT_TRANSCRIPTION_PROMPT,
    MAX_PROMPT_TOKENS,
    PREVIOUS_TEXT_TOKENS,
    build_transcription_prompt,
    normalize_transcription_prompt,
)


def byte_tokens(text):
    """A conservative tokenizer that exposes multi-byte character boundaries."""
    return list(text.encode("utf-8"))


class TranscriptionPromptTests(unittest.TestCase):
    def test_default_example_is_short_and_kept_intact(self):
        result = build_transcription_prompt(DEFAULT_TRANSCRIPTION_PROMPT, None, byte_tokens)
        self.assertEqual(result, DEFAULT_TRANSCRIPTION_PROMPT)
        self.assertLess(len(byte_tokens(result)), MAX_PROMPT_TOKENS)

    def test_normalization_preserves_text_but_ignores_invalid_saved_values(self):
        self.assertEqual(normalize_transcription_prompt("  Alice, Bob.\n "), "Alice, Bob.")
        for value in (None, False, 42, ["example"], {"prompt": "example"}):
            with self.subTest(value=value):
                self.assertEqual(normalize_transcription_prompt(value), "")

    def test_empty_inputs_disable_initial_prompt(self):
        self.assertIsNone(build_transcription_prompt(" \n", None, byte_tokens))

    def test_both_style_and_latest_context_survive_a_long_transcript(self):
        style = "WhisperType, Schrödinger. A clear sentence, with punctuation."
        previous = "Earlier words. " * 100 + "The latest sentence is still relevant."
        result = build_transcription_prompt(style, previous, byte_tokens)
        self.assertTrue(result.startswith(style + "\n"))
        self.assertTrue(result.endswith("The latest sentence is still relevant."))
        self.assertNotIn("Earlier words. Earlier words. Earlier words. Earlier words.", result)
        self.assertLessEqual(len(byte_tokens(result)), MAX_PROMPT_TOKENS)

    def test_no_style_gives_the_whole_budget_to_recent_transcript(self):
        previous = "a" * 500 + " End."
        result = build_transcription_prompt("", previous, byte_tokens)
        self.assertEqual(result, previous[-MAX_PROMPT_TOKENS:])

    def test_unicode_fragments_are_not_corrupted_when_trimmed(self):
        style = "東京、明日の予定。🌱 " * 100
        previous = "過去の文章。" * 100 + " 最後の言葉🌷"
        result = build_transcription_prompt(style, previous, byte_tokens)
        fitted_style, history = result.split("\n", 1)
        self.assertTrue(style.startswith(fitted_style))
        self.assertTrue(previous.endswith(history))
        self.assertTrue(result.endswith("最後の言葉🌷"))
        self.assertNotIn("\ufffd", result)
        self.assertLessEqual(len(byte_tokens(history)), PREVIOUS_TEXT_TOKENS)
        self.assertLessEqual(len(byte_tokens(result)), MAX_PROMPT_TOKENS)

    def test_joined_budget_counts_separator_and_engine_added_prefix(self):
        def encode(text):
            # Tokenization across the join can have overhead not present in
            # separately encoded fragments. Also mimic an engine's leading
            # space so that its token is included in the budget.
            return byte_tokens(" " + text) + ([0] * 8 if "\n" in text else [])

        result = build_transcription_prompt("s" * 500, "c" * 500, encode)
        self.assertEqual(len(encode(result)), MAX_PROMPT_TOKENS)
        style, history = result.split("\n", 1)
        self.assertTrue(style)
        self.assertTrue(history)
        self.assertLessEqual(len(encode(history)), PREVIOUS_TEXT_TOKENS)


if __name__ == "__main__":
    unittest.main()
