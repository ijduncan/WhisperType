"""
WhisperType - Offline voice-to-text for Windows.

Press a global hotkey to start recording, press again to stop. Your speech is
transcribed locally with Whisper (faster-whisper) and pasted into the focused
app. No internet is used after the model has been downloaded once.
"""

import json
import os
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _config_path():
    """Config lives next to the script when running from source, but in a
    writable per-user folder when installed (Program Files is read-only)."""
    if getattr(sys, "frozen", False):
        d = os.path.join(os.environ.get("APPDATA", APP_DIR), "WhisperType")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "config.json")
    return os.path.join(APP_DIR, "config.json")


CONFIG_PATH = _config_path()


def _bundle_dir():
    """Directory holding bundled data: the PyInstaller bundle when frozen,
    otherwise this script's folder."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return APP_DIR


def _nvidia_roots():
    roots = []
    # Frozen bundle: DLLs collected under <bundle>/nvidia/<lib>/bin
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(_bundle_dir(), "nvidia"))
    else:
        try:
            import importlib.util
            spec = importlib.util.find_spec("nvidia")
            if spec and spec.submodule_search_locations:
                roots.append(spec.submodule_search_locations[0])
        except Exception:
            pass
    return roots


def _register_cuda_dlls():
    """Make the bundled NVIDIA cuBLAS/cuDNN DLLs discoverable so CTranslate2
    can run on the GPU. No-op off Windows or if the libs aren't present."""
    if os.name != "nt":
        return
    try:
        for root in _nvidia_roots():
            for sub in ("cublas", "cudnn", "cuda_nvrtc"):
                binp = os.path.join(root, sub, "bin")
                if os.path.isdir(binp):
                    try:
                        os.add_dll_directory(binp)
                    except Exception:
                        pass
                    # CTranslate2 loads cuBLAS/cuDNN lazily with a plain library
                    # name, which only searches PATH — so prepend to PATH too.
                    os.environ["PATH"] = binp + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"[cuda] Could not register CUDA DLL dirs: {e}")


def _resolve_model(name):
    """Prefer a model bundled next to the app (offline) over the HF cache /
    download by name."""
    local = os.path.join(_bundle_dir(), "models", name)
    if os.path.isdir(local):
        return local
    return name

DEFAULT_CONFIG = {
    "hotkey": "ctrl+alt+windows+space",  # global toggle hotkey
    "model": "medium.en",          # whisper model (medium.en = accurate, English)
    "language": "en",              # transcription language
    "device": "cuda",              # "cuda" (NVIDIA GPU) or "cpu"
    "compute_type": "float16",     # float16 on GPU; use "int8" on CPU
    "output_mode": "paste",        # "paste" into focused app, or "clipboard" only
    "sample_rate": 16000,          # Whisper expects 16 kHz
    "min_seconds": 0.3,            # ignore recordings shorter than this
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[config] Could not read config.json ({e}); using defaults.")
    # Write back so the user has a complete file to edit.
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass
    return cfg


# ---------------------------------------------------------------------------
# Tray icon images (simple colored circles for each state)
# ---------------------------------------------------------------------------

def _icon(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color)
    return img


ICONS = {
    "loading": _icon((150, 150, 150, 255)),   # gray
    "idle": _icon((40, 120, 220, 255)),        # blue
    "recording": _icon((220, 50, 50, 255)),    # red
    "working": _icon((230, 150, 30, 255)),     # orange
}


class WhisperType:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.state = "loading"
        self.lock = threading.Lock()
        self.recording = False
        self.frames = []
        self.stream = None
        self.icon = None
        self._hotkey_handle = None
        self._settings_open = False

    # -- config -------------------------------------------------------------
    def save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
        except Exception as e:
            print(f"[cfg] Could not save config: {e}")

    def apply_hotkey(self, new_hotkey):
        """Re-register the global hotkey live and persist it."""
        try:
            if self._hotkey_handle is not None:
                keyboard.remove_hotkey(self._hotkey_handle)
        except Exception:
            pass
        self._hotkey_handle = keyboard.add_hotkey(new_hotkey, self.on_hotkey)
        self.cfg["hotkey"] = new_hotkey
        self.save_config()
        print(f"[cfg] hotkey = {new_hotkey}")
        if self.icon is not None:
            try:
                self.icon.update_menu()
            except Exception:
                pass

    # -- tray helpers -------------------------------------------------------
    def set_state(self, state):
        self.state = state
        if self.icon is not None:
            self.icon.icon = ICONS[state]
            self.icon.title = f"WhisperType — {state}"
            try:
                self.icon.update_menu()
            except Exception:
                pass

    # -- model --------------------------------------------------------------
    def load_model(self):
        _register_cuda_dlls()
        from faster_whisper import WhisperModel
        device = self.cfg["device"]
        compute = self.cfg["compute_type"]
        model_ref = _resolve_model(self.cfg["model"])
        bundled = os.path.isdir(model_ref)
        print(f"[model] Loading '{self.cfg['model']}' ({device}/{compute})… "
              + ("using bundled model." if bundled else "downloads on first run."))
        t0 = time.time()
        try:
            self.model = WhisperModel(
                model_ref, device=device, compute_type=compute
            )
        except Exception as e:
            if device == "cuda":
                print(f"[model] GPU load failed ({e}); falling back to CPU/int8.")
                device, compute = "cpu", "int8"
                self.model = WhisperModel(
                    model_ref, device=device, compute_type=compute
                )
            else:
                raise
        print(f"[model] Ready on {device} in {time.time() - t0:.1f}s.")
        self.set_state("idle")

    # -- recording ----------------------------------------------------------
    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.frames.append(indata.copy())

    def start_recording(self):
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.cfg["sample_rate"],
            channels=1,
            dtype="float32",
            callback=self._audio_cb,
        )
        self.stream.start()
        self.recording = True
        self.set_state("recording")
        print("[rec] Recording… press the hotkey again to stop.")

    def stop_recording(self):
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
        if not self.frames:
            self.set_state("idle")
            return None
        audio = np.concatenate(self.frames, axis=0).flatten().astype(np.float32)
        secs = len(audio) / self.cfg["sample_rate"]
        print(f"[rec] Stopped ({secs:.1f}s).")
        if secs < self.cfg["min_seconds"]:
            print("[rec] Too short, ignoring.")
            self.set_state("idle")
            return None
        return audio

    # -- transcription ------------------------------------------------------
    def transcribe_and_output(self, audio):
        self.set_state("working")
        print("[stt] Transcribing…")
        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            vad_filter=True,
            beam_size=5,
        )
        text = "".join(s.text for s in segments).strip()
        print(f"[stt] {time.time() - t0:.1f}s → {text!r}")
        if text:
            self.output(text)
        self.set_state("idle")

    def output(self, text):
        pyperclip.copy(text)
        if self.cfg["output_mode"] == "paste":
            time.sleep(0.08)  # let modifiers from the hotkey fully release
            keyboard.send("ctrl+v")
            print("[out] Pasted into focused app.")
        else:
            print("[out] Copied to clipboard.")

    # -- hotkey dispatch ----------------------------------------------------
    def on_hotkey(self):
        # Runs in the keyboard hook thread; keep it snappy by handing the
        # heavy work to a worker thread.
        with self.lock:
            if self.state == "loading":
                print("[hotkey] Model still loading, please wait…")
                return
            if not self.recording:
                self.start_recording()
            else:
                audio = self.stop_recording()
                if audio is not None:
                    threading.Thread(
                        target=self.transcribe_and_output,
                        args=(audio,),
                        daemon=True,
                    ).start()

    # -- lifecycle ----------------------------------------------------------
    def toggle_output_mode(self, icon, item):
        self.cfg["output_mode"] = (
            "clipboard" if self.cfg["output_mode"] == "paste" else "paste"
        )
        self.save_config()
        print(f"[cfg] output_mode = {self.cfg['output_mode']}")

    def open_settings(self, icon, item):
        # Run the Tk dialog on its own thread so it doesn't fight pystray's
        # message loop. Only one window at a time.
        if self._settings_open:
            return
        self._settings_open = True
        threading.Thread(target=self._run_settings_window, daemon=True).start()

    def build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(lambda item: f"Status: {self.state}", None, enabled=False),
            pystray.MenuItem(lambda item: f"Hotkey: {self.cfg['hotkey']}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Auto-paste into focused app",
                self.toggle_output_mode,
                checked=lambda item: self.cfg["output_mode"] == "paste",
            ),
            pystray.MenuItem("Settings…", self.open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit),
        )

    # -- settings window ----------------------------------------------------
    def _run_settings_window(self):
        import tkinter as tk
        from tkinter import ttk, messagebox

        try:
            root = tk.Tk()
            root.title("WhisperType Settings")
            root.resizable(False, False)
            try:
                ico = os.path.join(_bundle_dir(), "icon.ico")
                if os.path.isfile(ico):
                    root.iconbitmap(ico)
            except Exception:
                pass

            frm = ttk.Frame(root, padding=16)
            frm.grid(sticky="nsew")

            # --- Hotkey ---
            ttk.Label(frm, text="Global hotkey").grid(row=0, column=0, sticky="w")
            hotkey_var = tk.StringVar(value=self.cfg["hotkey"])
            hk_entry = ttk.Entry(frm, textvariable=hotkey_var, width=28)
            hk_entry.grid(row=1, column=0, sticky="we", pady=(2, 0))

            record_btn = ttk.Button(frm, text="Record")

            def record_keys():
                record_btn.config(text="Press keys…", state="disabled")
                root.update_idletasks()

                def capture():
                    try:
                        combo = keyboard.read_hotkey(suppress=False)
                    except Exception:
                        combo = None
                    def done():
                        if combo:
                            hotkey_var.set(combo)
                        record_btn.config(text="Record", state="normal")
                    root.after(0, done)

                threading.Thread(target=capture, daemon=True).start()

            record_btn.config(command=record_keys)
            record_btn.grid(row=1, column=1, padx=(8, 0), pady=(2, 0))
            ttk.Label(
                frm, text="e.g. ctrl+alt+windows+space", foreground="#777"
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 10))

            # --- Output mode ---
            ttk.Label(frm, text="Output").grid(row=3, column=0, sticky="w")
            output_var = tk.StringVar(value=self.cfg["output_mode"])
            ttk.Combobox(
                frm, textvariable=output_var, state="readonly", width=26,
                values=["paste", "clipboard"],
            ).grid(row=4, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Language ---
            ttk.Label(frm, text="Language (ISO code, e.g. en)").grid(
                row=5, column=0, sticky="w")
            lang_var = tk.StringVar(value=self.cfg["language"])
            ttk.Entry(frm, textvariable=lang_var, width=28).grid(
                row=6, column=0, columnspan=2, sticky="we", pady=(2, 14))

            # --- Buttons ---
            btns = ttk.Frame(frm)
            btns.grid(row=7, column=0, columnspan=2, sticky="e")

            def on_save():
                new_hotkey = hotkey_var.get().strip().lower()
                if not new_hotkey:
                    messagebox.showerror("WhisperType", "Hotkey cannot be empty.")
                    return
                # Validate the hotkey before committing to it.
                if new_hotkey != self.cfg["hotkey"]:
                    try:
                        h = keyboard.add_hotkey(new_hotkey, self.on_hotkey)
                        keyboard.remove_hotkey(h)
                    except Exception as e:
                        messagebox.showerror(
                            "WhisperType", f"'{new_hotkey}' is not a valid hotkey.\n\n{e}")
                        return
                    self.apply_hotkey(new_hotkey)
                self.cfg["output_mode"] = output_var.get()
                self.cfg["language"] = lang_var.get().strip() or "en"
                self.save_config()
                if self.icon is not None:
                    try:
                        self.icon.update_menu()
                    except Exception:
                        pass
                print("[cfg] Settings saved.")
                root.destroy()

            ttk.Button(btns, text="Cancel", command=root.destroy).grid(
                row=0, column=0, padx=(0, 8))
            ttk.Button(btns, text="Save", command=on_save).grid(row=0, column=1)

            root.update_idletasks()
            root.eval("tk::PlaceWindow . center")
            root.attributes("-topmost", True)
            root.after(100, lambda: root.attributes("-topmost", False))
            hk_entry.focus_set()
            root.mainloop()
        except Exception as e:
            print(f"[settings] error: {e}")
        finally:
            self._settings_open = False

    def quit(self, icon, item):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        icon.stop()
        os._exit(0)

    def run(self):
        # Register the global hotkey (tracks the handle for live re-binding).
        self._hotkey_handle = keyboard.add_hotkey(self.cfg["hotkey"], self.on_hotkey)
        print(f"[ready] Hotkey: {self.cfg['hotkey']}  "
              f"(output: {self.cfg['output_mode']})")

        # Load the model in the background so the tray appears immediately.
        threading.Thread(target=self.load_model, daemon=True).start()

        self.icon = pystray.Icon(
            "WhisperType", ICONS["loading"], "WhisperType — loading", self.build_menu()
        )
        self.icon.run()


def selftest():
    """Load the model and run one transcription — no GUI. Used to verify a
    frozen build. Writes a log file and exits non-zero on failure."""
    import tempfile
    import traceback
    logp = os.path.join(tempfile.gettempdir(), "whispertype_selftest.log")

    def log(msg):
        print(msg)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(logp, "w").close()
    try:
        import numpy as np
        cfg = load_config()
        log(f"frozen={getattr(sys, 'frozen', False)} bundle={_bundle_dir()}")
        log(f"model_ref={_resolve_model(cfg['model'])}")
        app = WhisperType(cfg)
        app.load_model()
        tone = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(
            0, 2, cfg["sample_rate"] * 2, dtype=np.float32))).astype(np.float32)
        segments, _ = app.model.transcribe(
            tone, language=cfg["language"], vad_filter=False)
        list(segments)
        log("SELFTEST OK — model loaded and GPU transcribe path ran.")
    except Exception:
        log("SELFTEST FAILED:\n" + traceback.format_exc())
        sys.exit(1)


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    cfg = load_config()
    WhisperType(cfg).run()


if __name__ == "__main__":
    main()
