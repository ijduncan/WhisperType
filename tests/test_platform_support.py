import os
import tempfile
import unittest
from unittest import mock

from platform_support import (
    KeyboardBackend,
    default_acceleration,
    default_hotkey,
    default_model,
    normalize_hotkey,
    to_pynput_hotkey,
    user_data_dir,
)


class PlatformSupportTests(unittest.TestCase):
    def test_platform_defaults(self):
        self.assertEqual(default_hotkey("darwin"), "cmd+shift+space")
        self.assertEqual(default_model("darwin"), "small.en")
        self.assertEqual(default_acceleration("darwin"), ("cpu", "int8"))
        self.assertEqual(default_acceleration("win32"), ("cuda", "float16"))

    def test_hotkey_aliases_are_normalized(self):
        self.assertEqual(
            normalize_hotkey("Control + Option + Command + Space", "darwin"),
            "ctrl+option+cmd+space",
        )
        self.assertEqual(
            normalize_hotkey("ctrl+option+cmd+space", "win32"),
            "ctrl+alt+windows+space",
        )

    def test_hotkey_converts_to_pynput_syntax(self):
        self.assertEqual(
            to_pynput_hotkey("cmd+shift+space"),
            "<cmd>+<shift>+<space>",
        )

    def test_hotkey_rejects_multiple_primary_keys(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            normalize_hotkey("cmd+a+b", "darwin")

    def test_windows_backend_preserves_keyboard_library_syntax(self):
        parsed = []
        fake_keyboard = mock.Mock()
        fake_keyboard.parse_hotkey.side_effect = parsed.append
        with mock.patch.dict("sys.modules", {"keyboard": fake_keyboard}):
            value = KeyboardBackend("win32").validate("Ctrl+Shift+A, B")
        self.assertEqual(value, "ctrl+shift+a, b")
        self.assertEqual(parsed, ["ctrl+shift+a, b"])

    def test_macos_data_directory(self):
        self.assertEqual(
            user_data_dir(platform_name="darwin", home="/Users/example", env={}),
            "/Users/example/Library/Application Support/WhisperType",
        )

    def test_data_directory_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = os.path.abspath(temp_dir)
            self.assertEqual(
                user_data_dir(platform_name="darwin", env={"WHISPERTYPE_DATA_DIR": temp_dir}),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
