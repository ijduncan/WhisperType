import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import mlx_backend
import whispertype


class MlxBackendTests(unittest.TestCase):
    def test_word_timestamp_stub_is_used_without_numba(self):
        real_find_spec = mlx_backend.importlib.util.find_spec

        def find_spec(name, *args):
            if name in ("numba", "scipy"):
                return None
            return real_find_spec(name, *args)

        with mock.patch.dict(sys.modules), \
                mock.patch.object(mlx_backend.importlib.util, "find_spec", find_spec):
            sys.modules.pop("mlx_whisper.timing", None)
            mlx_backend._stub_word_timestamps()
            stub = sys.modules["mlx_whisper.timing"]
            with self.assertRaises(RuntimeError):
                stub.add_word_timestamps()

    def test_silence_is_trimmed_to_nothing(self):
        silence = np.zeros(16000 * 3, dtype=np.float32)
        self.assertEqual(mlx_backend.trim_to_speech(silence).size, 0)

    def test_model_completeness_needs_config_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(mlx_backend.mlx_model_is_complete(d))
            open(os.path.join(d, "config.json"), "w").close()
            self.assertFalse(mlx_backend.mlx_model_is_complete(d))
            open(os.path.join(d, "weights.npz"), "w").close()
            self.assertTrue(mlx_backend.mlx_model_is_complete(d))

    def test_every_cpu_model_has_an_apple_gpu_build(self):
        self.assertEqual(set(mlx_backend.MLX_MODELS), set(whispertype.MODELS))


class MlxConfigTests(unittest.TestCase):
    def load(self, saved):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(saved, f)
            with mock.patch.object(whispertype, "CONFIG_PATH", path), \
                    mock.patch.object(whispertype, "IS_MACOS", True):
                return whispertype.load_config()

    def test_macos_keeps_apple_gpu_choice(self):
        cfg = self.load({"device": "mlx", "compute_type": "float16"})
        self.assertEqual(cfg["device"], "mlx")

    def test_macos_migrates_cuda_config_to_cpu(self):
        cfg = self.load({"device": "cuda", "compute_type": "float16"})
        self.assertEqual((cfg["device"], cfg["compute_type"]), ("cpu", "int8"))

    def test_failed_apple_gpu_load_falls_back_to_cpu(self):
        app = whispertype.WhisperType(dict(whispertype.DEFAULT_CONFIG, device="mlx"))
        with mock.patch.object(app, "_load_mlx_model", side_effect=RuntimeError("no gpu")), \
                mock.patch.object(app, "_load_ct2_model") as ct2:
            app.load_model()
        ct2.assert_called_once()

    def test_mlx_choice_downloads_only_the_mlx_model(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(whispertype, "_data_dir", return_value=d), \
                mock.patch.object(whispertype, "_local_mlx_models", return_value={}), \
                mock.patch.object(whispertype, "fetch_mlx_model",
                                  side_effect=lambda n, dest: calls.append(("mlx", n))), \
                mock.patch.object(whispertype, "fetch_model",
                                  side_effect=lambda n, dest: calls.append(("ct2", n))):
            whispertype.ensure_components(model="small.en", want_mlx=True)
        self.assertEqual(calls, [("mlx", "small.en")])


if __name__ == "__main__":
    unittest.main()
