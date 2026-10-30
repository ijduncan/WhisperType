import os
import sys
import unittest
from unittest import mock

import whispertype


class StartupTests(unittest.TestCase):
    def test_second_launch_exits_before_config_model_and_hotkey_setup(self):
        with (
            mock.patch.object(sys, "argv", ["whispertype.py"]),
            mock.patch.object(whispertype.multiprocessing, "freeze_support") as freeze,
            mock.patch.object(whispertype, "user_data_dir", return_value="user-data"),
            mock.patch.object(whispertype, "SingleInstance") as guard,
            mock.patch.object(whispertype, "load_config") as load_config,
            mock.patch.object(whispertype, "WhisperType") as app,
        ):
            guard.return_value.acquire.return_value = False
            whispertype.main()
            freeze.assert_called_once_with()
            guard.assert_called_once_with(os.path.join("user-data", "instance.lock"))
            load_config.assert_not_called()
            app.assert_not_called()
            guard.return_value.release.assert_not_called()

    def test_primary_holds_lock_until_app_exits_including_on_error(self):
        for error in (None, RuntimeError("app failed")):
            with (
                self.subTest(error=error),
                mock.patch.object(sys, "argv", ["whispertype.py"]),
                mock.patch.object(whispertype.multiprocessing, "freeze_support"),
                mock.patch.object(whispertype, "user_data_dir", return_value="user-data"),
                mock.patch.object(whispertype, "SingleInstance") as guard,
                mock.patch.object(whispertype, "load_config", return_value={}),
                mock.patch.object(whispertype, "WhisperType") as app,
            ):
                guard.return_value.acquire.return_value = True

                def run():
                    guard.return_value.acquire.assert_called_once_with()
                    guard.return_value.release.assert_not_called()
                    if error is not None:
                        raise error

                app.return_value.run.side_effect = run
                if error is None:
                    whispertype.main()
                else:
                    with self.assertRaisesRegex(RuntimeError, "app failed"):
                        whispertype.main()
                app.assert_called_once_with({})
                guard.return_value.release.assert_called_once_with()

    def test_maintenance_commands_can_run_while_dictation_is_open(self):
        for flag, handler in (
            ("--check", "check"),
            ("--fetch", "fetch_main"),
            ("--overlaytest", "overlaytest"),
            ("--menutest", "menutest"),
            ("--selftest", "selftest"),
        ):
            with (
                self.subTest(flag=flag),
                mock.patch.object(sys, "argv", ["whispertype.py", flag]),
                mock.patch.object(whispertype.multiprocessing, "freeze_support"),
                mock.patch.object(whispertype, handler, return_value=0) as command,
                mock.patch.object(whispertype, "SingleInstance") as guard,
                mock.patch.object(whispertype, "load_config") as load_config,
                mock.patch.object(whispertype, "WhisperType") as app,
            ):
                if flag in ("--check", "--fetch"):
                    with self.assertRaises(SystemExit) as exited:
                        whispertype.main()
                    self.assertEqual(exited.exception.code, 0)
                else:
                    whispertype.main()
                command.assert_called_once()
                guard.assert_not_called()
                load_config.assert_not_called()
                app.assert_not_called()


if __name__ == "__main__":
    unittest.main()
