import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import mlx_backend
import whispertype
from streaming import StreamingTranscriber
from transcription_prompt import DEFAULT_TRANSCRIPTION_PROMPT


def byte_tokens(text):
    return list(text.encode("utf-8"))


class PromptIntegrationTests(unittest.TestCase):
    def make_app(self, engine, style, responses):
        cfg = dict(whispertype.DEFAULT_CONFIG, transcription_prompt=style)
        app = whispertype.WhisperType(cfg)
        app.engine = engine
        app.model = mock.Mock()
        if engine == "mlx":
            app.model.encode_prompt.side_effect = (
                lambda text: byte_tokens(" " + text.strip())
            )
            app.model.transcribe.side_effect = responses
        else:
            app.model.hf_tokenizer.encode.side_effect = (
                lambda text, **_kwargs: SimpleNamespace(ids=byte_tokens(text))
            )
            app.model.transcribe.side_effect = [
                (iter([SimpleNamespace(text=text)]), None)
                for text in responses
            ]
        return app

    def prompts(self, app):
        if app.engine == "mlx":
            return [call.args[2] for call in app.model.transcribe.call_args_list]
        return [
            call.kwargs["initial_prompt"]
            for call in app.model.transcribe.call_args_list
        ]

    def test_one_shot_passes_custom_style_to_each_engine(self):
        style = "WhisperType, Qwen. Clear sentences, with punctuation."
        audio = np.zeros(16000, dtype=np.float32)
        for engine in ("ct2", "mlx"):
            with self.subTest(engine=engine):
                app = self.make_app(engine, style, [" A spoken sentence."])
                result = app._transcribe_audio(audio)
                self.assertEqual(self.prompts(app), [style])
                self.assertEqual(result.strip(), "A spoken sentence.")
                self.assertNotIn(style, result)
                if engine == "ct2":
                    self.assertTrue(all(
                        call.kwargs.get("add_special_tokens") is False
                        for call in app.model.hf_tokenizer.encode.call_args_list
                    ))

    def test_streaming_keeps_style_and_previous_speech_for_the_tail(self):
        style = "WhisperType. Clear sentences, with commas."
        first = "First phrase."
        last = "Second phrase."
        for engine in ("ct2", "mlx"):
            with self.subTest(engine=engine):
                app = self.make_app(engine, style, [first, last])
                audio = np.zeros(16000 * 2, dtype=np.float32)
                streamer = StreamingTranscriber(
                    get_audio=lambda: audio,
                    transcribe=app._transcribe_audio,
                    vad=lambda _pending: [{"start": 0, "end": 16000}],
                )
                streamer.step()
                transcript = streamer.finish(np.zeros(16000 * 3, dtype=np.float32))
                initial, following = self.prompts(app)
                self.assertEqual(initial, style)
                self.assertTrue(following.startswith(style + "\n"))
                self.assertTrue(following.endswith(first))
                self.assertEqual(transcript, f"{first} {last}")

    def test_empty_style_disables_the_seed_but_preserves_stream_context(self):
        audio = np.zeros(16000, dtype=np.float32)
        context = "The preceding spoken words."
        for engine in ("ct2", "mlx"):
            with self.subTest(engine=engine):
                app = self.make_app(engine, "", ["First.", "Second."])
                app._transcribe_audio(audio)
                app._transcribe_audio(audio, context)
                self.assertEqual(self.prompts(app), [None, context])

    def test_mlx_decode_forwards_composed_prompt_without_adding_it_to_output(self):
        prompt = "A punctuated example.\nEarlier spoken words."
        model = mlx_backend.MlxWhisper.__new__(mlx_backend.MlxWhisper)
        model.model_dir = "/local/WhisperType/model"
        fake_mlx = SimpleNamespace(
            transcribe=mock.Mock(return_value={"text": " Actual speech. "})
        )
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
            result = model._decode(np.zeros(16000, dtype=np.float32), "en", prompt)
        self.assertEqual(fake_mlx.transcribe.call_args.kwargs["initial_prompt"], prompt)
        self.assertEqual(result, "Actual speech.")

    def test_mlx_prompt_encoding_matches_each_models_local_vocabulary(self):
        for n_vocab, multilingual, num_languages in (
            (51864, False, 99),
            (51865, True, 99),
            (51866, True, 100),
        ):
            with self.subTest(n_vocab=n_vocab), tempfile.TemporaryDirectory() as folder:
                with open(os.path.join(folder, "config.json"), "w") as config_file:
                    json.dump({"n_vocab": n_vocab}, config_file)
                tokenizer = SimpleNamespace(encode=mock.Mock(side_effect=byte_tokens))
                get_tokenizer = mock.Mock(return_value=tokenizer)
                model = mlx_backend.MlxWhisper.__new__(mlx_backend.MlxWhisper)
                model.model_dir = folder
                with mock.patch.dict(sys.modules, {
                    "mlx_whisper": SimpleNamespace(),
                    "mlx_whisper.tokenizer": SimpleNamespace(get_tokenizer=get_tokenizer),
                }):
                    self.assertEqual(model.encode_prompt("  Zoë, Qwen.  "), byte_tokens(" Zoë, Qwen."))
                    model.encode_prompt("Next phrase.")
                get_tokenizer.assert_called_once_with(
                    multilingual,
                    num_languages=num_languages,
                    language="en", task="transcribe",
                )
                self.assertEqual(tokenizer.encode.call_args_list[0].args, (" Zoë, Qwen.",))


class PromptConfigTests(unittest.TestCase):
    def load_at(self, path):
        with (
            mock.patch.object(whispertype, "CONFIG_PATH", path),
            mock.patch.object(whispertype, "IS_WINDOWS", True),
            mock.patch.object(whispertype, "IS_MACOS", False),
            mock.patch.object(
                whispertype.KeyboardBackend, "validate", side_effect=lambda value: value
            ),
        ):
            return whispertype.load_config()

    def test_existing_config_gains_the_default_example_without_losing_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump({"hotkey": "ctrl+alt+d", "output_mode": "clipboard"}, config_file)
            config = self.load_at(path)
            self.assertEqual(config["transcription_prompt"], DEFAULT_TRANSCRIPTION_PROMPT)
            self.assertEqual(config["hotkey"], "ctrl+alt+d")
            self.assertEqual(config["output_mode"], "clipboard")
            with open(path, encoding="utf-8") as config_file:
                saved = json.load(config_file)
            self.assertEqual(saved["transcription_prompt"], DEFAULT_TRANSCRIPTION_PROMPT)

    def test_custom_unicode_and_disabled_prompts_round_trip_through_save(self):
        for value in ("Schrödinger, Zoë.\n你好，世界。", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "config.json")
                with open(path, "w", encoding="utf-8") as config_file:
                    json.dump({"transcription_prompt": value}, config_file, ensure_ascii=False)
                config = self.load_at(path)
                app = whispertype.WhisperType(config)
                with mock.patch.object(whispertype, "CONFIG_PATH", path):
                    app.save_config()
                reloaded = self.load_at(path)
                self.assertEqual(reloaded["transcription_prompt"], value)

    def test_invalid_saved_prompt_disables_it_instead_of_becoming_spoken_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump({"transcription_prompt": None}, config_file)
            self.assertEqual(self.load_at(path)["transcription_prompt"], "")


if __name__ == "__main__":
    unittest.main()
