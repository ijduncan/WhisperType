import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import whispertype


class _FakeIcon:
    def __init__(self):
        self.icon = None
        self.title = None
        self.menu_updates = 0

    def update_menu(self):
        self.menu_updates += 1


class _FakeRoot:
    def __init__(self):
        self.callbacks = []

    def after(self, delay_ms, callback):
        self.callbacks.append((delay_ms, callback))
        return len(self.callbacks)


class MacRuntimeTests(unittest.TestCase):
    def make_app(self):
        return whispertype.WhisperType(dict(whispertype.DEFAULT_CONFIG))

    def test_macos_icon_updates_are_marshaled_to_ui_thread(self):
        app = self.make_app()
        callbacks = []
        app.ui = SimpleNamespace(call=callbacks.append)
        app.icon = _FakeIcon()

        with mock.patch.object(whispertype, "IS_MACOS", True):
            app.set_state("idle")

        self.assertEqual(app.state, "idle")
        self.assertIsNone(app.icon.icon)
        self.assertEqual(len(callbacks), 1)
        callbacks[0]()
        self.assertIs(app.icon.icon, whispertype.ICONS["idle"])
        self.assertEqual(app.icon.menu_updates, 1)

    def test_accessibility_poll_registers_fresh_listener_after_grant(self):
        app = self.make_app()
        app.cfg["hotkey"] = "cmd+shift+space"
        root = _FakeRoot()
        app.ui = SimpleNamespace(root=root, call=lambda callback: callback())
        app.keyboard = mock.Mock()
        app.keyboard.request_accessibility.side_effect = [False, True]
        handle = object()
        app.keyboard.register.return_value = handle

        with mock.patch.object(whispertype, "IS_MACOS", True):
            app._schedule_accessibility_retry(delay_ms=1)
            root.callbacks.pop(0)[1]()
            self.assertIsNone(app._hotkey_handle)
            root.callbacks.pop(0)[1]()

        self.assertIs(app._hotkey_handle, handle)
        self.assertIsNone(app.hotkey_error)
        app.keyboard.register.assert_called_once_with(
            app.cfg["hotkey"], app.on_hotkey
        )

    def test_macos_recording_uses_blocking_input_without_callback(self):
        app = self.make_app()
        app.set_state = mock.Mock()
        app.overlay = mock.Mock()
        stream = mock.Mock()

        with (
            mock.patch.object(whispertype, "IS_MACOS", True),
            mock.patch.object(whispertype.sd, "InputStream", return_value=stream) as ctor,
            mock.patch.object(whispertype.threading.Thread, "start", return_value=None),
        ):
            app.start_recording()

        kwargs = ctor.call_args.kwargs
        self.assertNotIn("callback", kwargs)
        self.assertEqual(kwargs["blocksize"], int(app.cfg["sample_rate"] * 0.03))
        stream.start.assert_called_once_with()
        self.assertTrue(app.recording)
        self.assertIsNotNone(app._capture_thread)

    def test_hotkey_callback_only_dispatches_worker(self):
        app = self.make_app()
        with mock.patch.object(
            whispertype.threading.Thread,
            "start",
            return_value=None,
        ) as start:
            app.on_hotkey()
        start.assert_called_once_with()

    def test_incomplete_model_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "model.bin"), "wb") as model_file:
                model_file.write(b"partial")
            self.assertFalse(whispertype._model_is_complete(temp_dir))
            with open(os.path.join(temp_dir, "tokenizer.json"), "w") as tokenizer:
                tokenizer.write("{}")
            self.assertTrue(whispertype._model_is_complete(temp_dir))

    def test_model_download_replaces_partial_directory_after_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = os.path.join(temp_dir, "models", "small.en")
            os.makedirs(final_dir)
            with open(os.path.join(final_dir, "model.bin"), "wb") as partial:
                partial.write(b"old-partial")

            def download(_name, output_dir):
                with open(os.path.join(output_dir, "model.bin"), "wb") as model_file:
                    model_file.write(b"complete")
                with open(os.path.join(output_dir, "tokenizer.json"), "w") as tokenizer:
                    tokenizer.write("{}")

            with mock.patch("faster_whisper.download_model", side_effect=download):
                whispertype.fetch_model("small.en", temp_dir)

            self.assertTrue(whispertype._model_is_complete(final_dir))
            with open(os.path.join(final_dir, "model.bin"), "rb") as model_file:
                self.assertEqual(model_file.read(), b"complete")

    def test_failed_model_download_preserves_previous_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = os.path.join(temp_dir, "models", "small.en")
            os.makedirs(final_dir)
            marker = os.path.join(final_dir, "keep-me")
            with open(marker, "w") as marker_file:
                marker_file.write("partial")

            def incomplete_download(_name, output_dir):
                with open(os.path.join(output_dir, "model.bin"), "wb") as model_file:
                    model_file.write(b"still-partial")

            with mock.patch(
                "faster_whisper.download_model",
                side_effect=incomplete_download,
            ), self.assertRaisesRegex(RuntimeError, "vocabulary"):
                whispertype.fetch_model("small.en", temp_dir)

            self.assertTrue(os.path.isfile(marker))
            self.assertFalse(whispertype._model_is_complete(final_dir))


if __name__ == "__main__":
    unittest.main()
