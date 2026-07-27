"""
WhisperType - Offline voice-to-text for Windows.

Press a global hotkey to start recording, press again to stop. Your speech is
transcribed locally with Whisper (faster-whisper) and pasted into the focused
app. No internet is used after the model has been downloaded once.
"""

import json
import math
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


def _data_dir():
    """Writable per-user folder for components fetched after install
    (models, CUDA libraries). Survives upgrades and needs no admin rights."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or APP_DIR
        d = os.path.join(base, "WhisperType")
    else:
        d = APP_DIR
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


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
    """Places CUDA DLLs may live, most-specific first: downloaded at install
    time, bundled in the package, or pip-installed in the dev venv."""
    roots = [os.path.join(_data_dir(), "nvidia")]
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
    """Prefer a locally-present model (downloaded at install time, or bundled)
    over fetching by name from Hugging Face."""
    for root in (_data_dir(), _bundle_dir()):
        local = os.path.join(root, "models", name)
        if os.path.isdir(local) and os.path.isfile(os.path.join(local, "model.bin")):
            return local
    return name

DEFAULT_CONFIG = {
    "hotkey": "ctrl+alt+windows+space",  # global toggle hotkey
    "model": "medium.en",          # whisper model (medium.en = accurate, English)
    "language": "en",              # transcription language
    "device": "cuda",              # "cuda" (NVIDIA GPU) or "cpu"
    "compute_type": "float16",     # float16 on GPU; use "int8" on CPU
    "output_mode": "paste",        # "paste" into focused app, or "clipboard" only
    "show_overlay": True,          # live waveform panel while recording
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


# ---------------------------------------------------------------------------
# Recording overlay: a small always-on-top waveform panel at bottom-center.
# ---------------------------------------------------------------------------

class Overlay:
    """Borderless, non-activating panel showing a live mic waveform.

    Runs on the shared Tk UI thread. It must never take focus, or the
    transcribed text would paste into the overlay's app instead of whatever the
    user was typing in — hence the WS_EX_NOACTIVATE/TRANSPARENT styles.
    """

    W, H = 260, 74
    BARS = 34

    def __init__(self, ui):
        self.ui = ui              # UiThread
        self.win = None
        self.canvas = None
        self.levels = [0.0] * self.BARS
        self.status = ""
        self._visible = False
        self._anim = None
        self._phase = 0.0

    # -- win32 window styling ----------------------------------------------
    def _make_click_through(self):
        try:
            import ctypes
            from ctypes import wintypes
            hwnd = int(self.win.frame(), 16)
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            u32 = ctypes.windll.user32
            u32.GetWindowLongW.restype = ctypes.c_long
            style = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                style | WS_EX_LAYERED | WS_EX_TRANSPARENT
                | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            )
        except Exception as e:
            print(f"[overlay] could not set click-through styles: {e}")

    def _round_corners(self, radius=16):
        try:
            import ctypes
            hwnd = int(self.win.frame(), 16)
            g = ctypes.windll.gdi32
            rgn = g.CreateRoundRectRgn(0, 0, self.W + 1, self.H + 1, radius, radius)
            ctypes.windll.user32.SetWindowRgn(hwnd, rgn, True)
        except Exception as e:
            print(f"[overlay] could not round corners: {e}")

    def _build(self):
        import tkinter as tk
        self.win = tk.Toplevel(self.ui.root)
        self.win.overrideredirect(True)          # no title bar / borders
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.92)
        BG = "#14161a"
        self.win.configure(bg=BG)
        self.canvas = tk.Canvas(
            self.win, width=self.W, height=self.H,
            bg=BG, highlightthickness=0, bd=0,
        )
        self.canvas.pack()
        # Position: bottom-center, a little above the taskbar.
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = int((sw - self.W) / 2)
        y = int(sh - self.H - 96)
        self.win.geometry(f"{self.W}x{self.H}+{x}+{y}")
        self.win.withdraw()
        self.win.update_idletasks()
        self._make_click_through()
        self._round_corners()

    # -- drawing ------------------------------------------------------------
    def _draw(self):
        c = self.canvas
        if c is None:
            return
        c.delete("all")
        pad = 14
        label_h = 18
        area_top = 8
        area_h = self.H - label_h - area_top - 8
        mid = area_top + area_h / 2
        usable = self.W - pad * 2
        step = usable / self.BARS
        bw = max(2, step * 0.55)
        listening = self.status == "Listening"
        accent = "#e5484d" if listening else "#f0a020"
        for i, lv in enumerate(self.levels):
            if not listening:
                # Transcribing: bars settle low and a pulse sweeps across, so
                # the panel reads as "working" rather than frozen.
                phase = (self._phase - i / self.BARS) % 1.0
                lv = 0.10 + 0.42 * max(0.0, math.sin(math.pi * phase) ** 8)
            h = max(2.0, lv * area_h * 0.92)
            x = pad + i * step + step / 2
            c.create_rectangle(
                x - bw / 2, mid - h / 2, x + bw / 2, mid + h / 2,
                fill=accent, outline="",
            )
        c.create_text(
            self.W / 2, self.H - label_h / 2 - 4,
            text=self.status, fill="#9aa0a6",
            font=("Segoe UI", 9),
        )

    def _tick(self):
        if not self._visible:
            return
        self._phase = (self._phase + 0.035) % 1.0
        self._draw()
        self._anim = self.ui.root.after(33, self._tick)   # ~30 fps

    # -- public API (thread-safe: call from anywhere) ----------------------
    def show(self, status="Listening"):
        def go():
            if self.win is None:
                self._build()
            self.status = status
            self.levels = [0.0] * self.BARS
            self._visible = True
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self._tick()
        self.ui.call(go)

    def set_status(self, status):
        def go():
            self.status = status
        self.ui.call(go)

    def push_level(self, level):
        """Append one amplitude sample (0..1) to the scrolling waveform."""
        self.levels.append(max(0.0, min(1.0, float(level))))
        if len(self.levels) > self.BARS:
            del self.levels[:-self.BARS]

    def hide(self):
        def go():
            self._visible = False
            if self._anim is not None:
                try:
                    self.ui.root.after_cancel(self._anim)
                except Exception:
                    pass
                self._anim = None
            if self.win is not None:
                self.win.withdraw()
        self.ui.call(go)


class UiThread:
    """Owns a single hidden Tk root on a dedicated thread, so tkinter work is
    always marshalled to one place (pystray owns the main thread)."""

    def __init__(self):
        self.root = None
        self._ready = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(timeout=10)

    def _run(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self._ready.set()
        self.root.mainloop()

    def call(self, fn):
        """Run fn on the UI thread."""
        if self.root is None:
            return
        try:
            self.root.after(0, fn)
        except Exception:
            pass


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
        self.ui = UiThread()
        self.overlay = Overlay(self.ui)

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
        if self.cfg.get("show_overlay", True):
            # RMS -> perceptual-ish level for the waveform bars.
            rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
            self.overlay.push_level(min(1.0, (rms ** 0.5) * 3.2))

    def start_recording(self):
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.cfg["sample_rate"],
            channels=1,
            dtype="float32",
            blocksize=int(self.cfg["sample_rate"] * 0.03),   # ~30 ms per bar
            callback=self._audio_cb,
        )
        self.stream.start()
        self.recording = True
        self.set_state("recording")
        if self.cfg.get("show_overlay", True):
            self.overlay.show("Listening")
        print("[rec] Recording… press the hotkey again to stop.")

    def stop_recording(self):
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
        if not self.frames:
            self.overlay.hide()
            self.set_state("idle")
            return None
        audio = np.concatenate(self.frames, axis=0).flatten().astype(np.float32)
        secs = len(audio) / self.cfg["sample_rate"]
        print(f"[rec] Stopped ({secs:.1f}s).")
        if secs < self.cfg["min_seconds"]:
            print("[rec] Too short, ignoring.")
            self.overlay.hide()
            self.set_state("idle")
            return None
        # Keep the panel up through transcription so the user sees progress.
        self.overlay.set_status("Transcribing…")
        return audio

    # -- transcription ------------------------------------------------------
    def transcribe_and_output(self, audio):
        self.set_state("working")
        print("[stt] Transcribing…")
        t0 = time.time()
        text = ""
        try:
            segments, _ = self.model.transcribe(
                audio,
                language=self.cfg["language"],
                vad_filter=True,
                beam_size=5,
            )
            text = "".join(s.text for s in segments).strip()
            print(f"[stt] {time.time() - t0:.1f}s → {text!r}")
        finally:
            # Always drop the overlay before pasting, so focus/z-order is clean.
            self.overlay.hide()
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
        self.ui.call(self._run_settings_window)

    def toggle_overlay(self, icon, item):
        self.cfg["show_overlay"] = not self.cfg.get("show_overlay", True)
        if not self.cfg["show_overlay"]:
            self.overlay.hide()
        self.save_config()
        print(f"[cfg] show_overlay = {self.cfg['show_overlay']}")

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
            pystray.MenuItem(
                "Show waveform overlay",
                self.toggle_overlay,
                checked=lambda item: self.cfg.get("show_overlay", True),
            ),
            pystray.MenuItem("Settings…", self.open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit),
        )

    # -- settings window ----------------------------------------------------
    def _run_settings_window(self):
        """Builds the settings window. Must run on the shared UI thread."""
        import tkinter as tk
        from tkinter import ttk, messagebox

        try:
            root = tk.Toplevel(self.ui.root)
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
                row=6, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Overlay ---
            overlay_var = tk.BooleanVar(value=self.cfg.get("show_overlay", True))
            ttk.Checkbutton(
                frm, text="Show waveform overlay while recording",
                variable=overlay_var,
            ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 14))

            # --- Buttons ---
            btns = ttk.Frame(frm)
            btns.grid(row=8, column=0, columnspan=2, sticky="e")

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
                self.cfg["show_overlay"] = bool(overlay_var.get())
                if not self.cfg["show_overlay"]:
                    self.overlay.hide()
                self.save_config()
                if self.icon is not None:
                    try:
                        self.icon.update_menu()
                    except Exception:
                        pass
                print("[cfg] Settings saved.")
                close()

            def close():
                self._settings_open = False
                try:
                    root.destroy()
                except Exception:
                    pass

            ttk.Button(btns, text="Cancel", command=close).grid(
                row=0, column=0, padx=(0, 8))
            ttk.Button(btns, text="Save", command=on_save).grid(row=0, column=1)
            root.protocol("WM_DELETE_WINDOW", close)

            root.update_idletasks()
            # Center on screen (Toplevel: compute manually).
            w, h = root.winfo_width(), root.winfo_height()
            x = (root.winfo_screenwidth() - w) // 2
            y = (root.winfo_screenheight() - h) // 2
            root.geometry(f"+{x}+{y}")
            root.attributes("-topmost", True)
            root.after(100, lambda: root.attributes("-topmost", False))
            root.lift()
            root.focus_force()
            hk_entry.focus_set()
        except Exception as e:
            print(f"[settings] error: {e}")
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
        # Single Tk thread shared by the overlay and the settings window.
        self.ui.start()

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


# ---------------------------------------------------------------------------
# Component downloader — used by the installer to fetch the model and the
# optional GPU libraries, so the installer itself stays small.
# ---------------------------------------------------------------------------

MODELS = {
    "base.en":   ("Systran/faster-whisper-base.en",   "~150 MB"),
    "small.en":  ("Systran/faster-whisper-small.en",  "~490 MB"),
    "medium.en": ("Systran/faster-whisper-medium.en", "~1.5 GB"),
    "large-v3":  ("Systran/faster-whisper-large-v3",  "~3.1 GB"),
}

# Immutable PyPI wheels holding the CUDA runtime libraries CTranslate2 needs.
CUDA_WHEELS = [
    ("nvidia-cublas-cu12", "12.9.2.10"),
    ("nvidia-cudnn-cu12", "9.25.0.15"),
    ("nvidia-cuda-nvrtc-cu12", "12.9.86"),
]


def _pypi_wheel_url(package, version):
    """Resolve the win_amd64 wheel URL for an exact package version."""
    import json as _json
    import urllib.request
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = _json.loads(r.read().decode("utf-8"))
    for f in data["urls"]:
        name = f.get("filename", "")
        if name.endswith(".whl") and "win_amd64" in name:
            return f["url"], f.get("size", 0)
    raise RuntimeError(f"no win_amd64 wheel for {package}=={version}")


def _download(url, dest, label=""):
    """Stream a URL to disk, reporting progress on stdout."""
    import urllib.request
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last and pct % 2 == 0:
                        last = pct
                        print(f"[fetch] {label} {pct}%", flush=True)
    os.replace(tmp, dest)
    return dest


def fetch_cuda(dest_root):
    """Download + unpack the CUDA DLLs into <dest_root>/nvidia/<lib>/bin."""
    import tempfile
    import zipfile
    out = os.path.join(dest_root, "nvidia")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for pkg, ver in CUDA_WHEELS:
            url, _size = _pypi_wheel_url(pkg, ver)
            whl = os.path.join(td, f"{pkg}-{ver}.whl")
            print(f"[fetch] downloading {pkg} {ver}", flush=True)
            _download(url, whl, label=pkg)
            print(f"[fetch] extracting {pkg}", flush=True)
            with zipfile.ZipFile(whl) as z:
                for m in z.namelist():
                    # keep only nvidia/<lib>/bin/*.dll
                    parts = m.split("/")
                    if (len(parts) >= 4 and parts[0] == "nvidia"
                            and parts[2] == "bin" and m.lower().endswith(".dll")):
                        target = os.path.join(out, parts[1], "bin", parts[3])
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with z.open(m) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            os.remove(whl)
    print(f"[fetch] CUDA libraries ready in {out}", flush=True)


def fetch_model(name, dest_root):
    """Download a faster-whisper model into <dest_root>/models/<name>."""
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}' (choose from {list(MODELS)})")
    repo, _size = MODELS[name]
    out = os.path.join(dest_root, "models", name)
    os.makedirs(out, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    files = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]
    for fn in files:
        target = os.path.join(out, fn)
        if os.path.isfile(target) and fn != "model.bin":
            continue
        print(f"[fetch] downloading {name}/{fn}", flush=True)
        try:
            _download(base + fn, target, label=fn)
        except Exception as e:
            # vocabulary.txt is absent for some repos; tokenizer covers it.
            if fn in ("vocabulary.txt", "tokenizer.json"):
                print(f"[fetch] skip {fn} ({e})", flush=True)
                continue
            raise
    print(f"[fetch] model '{name}' ready in {out}", flush=True)


def fetch_main(argv):
    """`--fetch [--model NAME] [--cuda]` — used by the installer."""
    import traceback
    dest = _data_dir()
    model = None
    want_cuda = "--cuda" in argv
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 < len(argv):
            model = argv[i + 1]
    try:
        if want_cuda:
            fetch_cuda(dest)
        if model and model.lower() != "none":
            fetch_model(model, dest)
            # Record the choice so the app uses it on first launch.
            cfg = load_config()
            cfg["model"] = model
            if not want_cuda:
                cfg["device"], cfg["compute_type"] = "cpu", "int8"
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        print("[fetch] DONE", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("[fetch] FAILED", flush=True)
        return 1


def overlaytest(seconds=4.0):
    """Show the overlay with simulated levels — verifies the panel works in a
    frozen build (Tk + win32 styling) without needing the mic or hotkey."""
    import random
    import tempfile
    import traceback
    logp = os.path.join(tempfile.gettempdir(), "whispertype_overlaytest.log")

    def log(msg):
        print(msg)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(logp, "w").close()
    try:
        log(f"frozen={getattr(sys, 'frozen', False)}")
        ui = UiThread()
        ui.start()
        ov = Overlay(ui)
        ov.show("Listening")
        t0 = time.time()
        while time.time() - t0 < seconds * 0.6:
            t = time.time() - t0
            ov.push_level(min(1.0, abs(math.sin(t * 2.2)) * (0.5 + 0.5 * random.random())))
            time.sleep(0.03)
        info = {}

        def probe():
            info["geometry"] = ov.win.geometry()
            info["mapped"] = bool(ov.win.winfo_ismapped())
            import ctypes
            ex = ctypes.windll.user32.GetWindowLongW(int(ov.win.frame(), 16), -20)
            info["noactivate"] = bool(ex & 0x08000000)
            info["transparent"] = bool(ex & 0x00000020)

        ui.call(probe)
        time.sleep(0.6)
        for k, v in info.items():
            log(f"  {k} = {v}")
        ov.set_status("Transcribing…")
        time.sleep(seconds * 0.4)
        ov.hide()
        time.sleep(0.4)
        ok = info.get("mapped") and info.get("noactivate") and info.get("transparent")
        log("OVERLAYTEST OK" if ok else "OVERLAYTEST FAILED — window/styles wrong")
        if not ok:
            sys.exit(1)
    except Exception:
        log("OVERLAYTEST FAILED:\n" + traceback.format_exc())
        sys.exit(1)


def main():
    if "--fetch" in sys.argv:
        sys.exit(fetch_main(sys.argv))
    if "--overlaytest" in sys.argv:
        overlaytest()
        return
    if "--selftest" in sys.argv:
        selftest()
        return
    cfg = load_config()
    WhisperType(cfg).run()


if __name__ == "__main__":
    main()
