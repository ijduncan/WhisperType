"""
WhisperType - Offline voice-to-text for Windows and macOS.

Press a global hotkey to start recording, press again to stop. Your speech is
transcribed locally with Whisper (faster-whisper) and pasted into the focused
app. No internet is used after the model has been downloaded once.
"""

import json
import math
import multiprocessing
import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import pyperclip
import pystray
from PIL import Image, ImageDraw

from platform_support import (
    IS_MACOS,
    IS_WINDOWS,
    KeyboardBackend,
    default_acceleration,
    default_hotkey,
    default_model,
    normalize_hotkey,
    open_accessibility_settings,
    user_data_dir,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_dir():
    """Writable per-user folder for components fetched after install
    (models, CUDA libraries). Survives upgrades and needs no admin rights."""
    if getattr(sys, "frozen", False) or os.environ.get("WHISPERTYPE_DATA_DIR"):
        d = user_data_dir()
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
    if os.environ.get("WHISPERTYPE_DATA_DIR"):
        d = user_data_dir()
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "config.json")
    if getattr(sys, "frozen", False):
        # Preserve the existing Windows roaming-config location while macOS
        # keeps all writable state in Application Support.
        if IS_WINDOWS and os.environ.get("APPDATA"):
            d = os.path.join(os.environ["APPDATA"], "WhisperType")
        else:
            d = user_data_dir()
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


def _model_is_complete(path):
    return (
        os.path.isfile(os.path.join(path, "model.bin"))
        and any(
            os.path.isfile(os.path.join(path, filename))
            for filename in ("vocabulary.json", "vocabulary.txt", "tokenizer.json")
        )
    )


def _local_models():
    """Every model present on disk, as {name: path}."""
    found = {}
    for root in (_data_dir(), _bundle_dir()):
        mdir = os.path.join(root, "models")
        if not os.path.isdir(mdir):
            continue
        for n in os.listdir(mdir):
            if n.startswith("."):
                continue
            p = os.path.join(mdir, n)
            if n not in found and _model_is_complete(p):
                found[n] = p
    return found


def _resolve_model(name):
    """Prefer a locally-present model over downloading from Hugging Face.

    If the configured model isn't installed but another one is, use that —
    otherwise the app sits on 'loading' silently pulling gigabytes, which
    looks like a hang.
    """
    local = _local_models()
    if name in local:
        return local[name]
    if local:
        alt = sorted(local)[0]
        print(f"[model] '{name}' is not installed; using '{alt}' instead. "
              f"Installed: {sorted(local)}")
        return local[alt]
    if IS_MACOS and getattr(sys, "frozen", False) and name in MODELS:
        # Keep downloads outside the signed .app bundle so they survive app
        # upgrades and do not invalidate the bundle signature.
        print(f"[model] First launch — downloading '{name}' to {_data_dir()}.")
        fetch_model(name, _data_dir())
        installed = _local_models().get(name)
        if installed:
            return installed
    print(f"[model] '{name}' is not installed locally — downloading from "
          f"Hugging Face, which can take several minutes.")
    return name

DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE = default_acceleration()

DEFAULT_CONFIG = {
    "hotkey": default_hotkey(),      # global toggle hotkey
    "model": default_model(),        # smaller CPU default on macOS
    "language": "en",              # transcription language
    "device": DEFAULT_DEVICE,        # NVIDIA CUDA on Windows, CPU on macOS
    "compute_type": DEFAULT_COMPUTE_TYPE,
    "output_mode": "paste",        # "paste" into focused app, or "clipboard" only
    "show_overlay": True,          # live waveform panel while recording
    "sample_rate": 16000,          # Whisper expects 16 kHz
    "min_seconds": 0.3,            # ignore recordings shorter than this
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            # utf-8-sig so a BOM (Notepad, PowerShell Set-Content) doesn't make
            # the file unreadable and silently reset every setting.
            with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[config] Could not read config.json ({e}); using defaults.")
            try:
                bad = CONFIG_PATH + ".bad"
                os.replace(CONFIG_PATH, bad)
                print(f"[config] Kept the unreadable file as {bad}")
            except Exception:
                pass
    # CTranslate2's macOS wheels are CPU-only.  Migrate a copied Windows config
    # instead of repeatedly attempting a CUDA load on every launch.
    if IS_MACOS and cfg.get("device") != "cpu":
        cfg["device"], cfg["compute_type"] = "cpu", "int8"
    try:
        configured_hotkey = str(cfg.get("hotkey", default_hotkey()))
        if IS_WINDOWS:
            cfg["hotkey"] = KeyboardBackend("win32").validate(configured_hotkey)
        else:
            cfg["hotkey"] = normalize_hotkey(configured_hotkey)
    except Exception:
        cfg["hotkey"] = default_hotkey()

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
    "error": _icon((120, 40, 140, 255)),       # purple
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
        self._nonactivating = False

    # -- win32 window styling ----------------------------------------------
    def _make_click_through(self):
        if IS_MACOS:
            try:
                # Tk's Aqua backend maps this to a non-activating help panel.
                # ignoreClicks keeps the overlay from intercepting the app the
                # user is dictating into.
                self.win.tk.call(
                    "::tk::unsupported::MacWindowStyle",
                    "style", self.win._w, "help",
                    "noActivates ignoreClicks doesNotHide",
                )
                self._nonactivating = True
            except Exception as e:
                print(f"[overlay] could not set macOS panel style: {e}")
            return
        if not IS_WINDOWS:
            return
        try:
            import ctypes
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
            self._nonactivating = True
        except Exception as e:
            print(f"[overlay] could not set click-through styles: {e}")

    def _round_corners(self, radius=16):
        if not IS_WINDOWS:
            return
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
        self.win.withdraw()
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
            font=(("Helvetica Neue", 11) if IS_MACOS else ("Segoe UI", 9)),
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
    """Own a single hidden Tk root and marshal all Tk work through a queue.

    Windows lets pystray own the main thread and runs Tk in a worker.  On
    macOS, Tk must own the main thread, so pystray is attached to that native
    run loop with ``run_detached``.
    """

    def __init__(self):
        self.root = None
        self._ready = threading.Event()
        self._jobs = queue.Queue()
        self._on_main_thread = False
        self._thread_id = None

    def start(self, on_main_thread=False):
        self._on_main_thread = on_main_thread
        if on_main_thread:
            self._init_root()
        else:
            threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(timeout=10)

    def _init_root(self):
        import tkinter as tk
        self._thread_id = threading.get_ident()
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.after(16, self._drain)
        self._ready.set()

    def _run(self):
        self._init_root()
        self.root.mainloop()

    def run_mainloop(self):
        if self.root is not None:
            self.root.mainloop()

    def _drain(self):
        while True:
            try:
                fn = self._jobs.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                print(f"[ui] callback failed: {e}")
        if self.root is not None:
            try:
                self.root.after(16, self._drain)
            except Exception:
                pass

    def call(self, fn):
        """Run fn on the UI thread.

        Always queued, even from the UI thread itself: on macOS the menu-bar
        callbacks arrive on the main thread while tkinter's mainloop still
        holds its Tcl lock inside Tcl_DoOneEvent, so touching Tk inline there
        deadlocks (the Settings item hung the app with a spinning cursor).
        """
        if self.root is None:
            return
        self._jobs.put(fn)

    def stop(self):
        def go():
            if self.root is None:
                return
            root, self.root = self.root, None
            try:
                root.quit()
                root.destroy()
            except Exception:
                pass

        self.call(go)


class WhisperType:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.state = "loading"
        self.lock = threading.Lock()
        self.recording = False
        self.frames = []
        self.stream = None
        self._capture_thread = None
        self._capture_stop = None
        self.icon = None
        self._hotkey_handle = None
        self._accessibility_retry_scheduled = False
        self._settings_open = False
        self._quitting = False
        self.keyboard = KeyboardBackend()
        self.hotkey_error = None
        self.ui = UiThread()
        self.overlay = Overlay(self.ui)
        self.load_error = None

    # -- config -------------------------------------------------------------
    def save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
        except Exception as e:
            print(f"[cfg] Could not save config: {e}")

    def apply_hotkey(self, new_hotkey):
        """Re-register the global hotkey live and persist it."""
        normalized = self.keyboard.validate(new_hotkey)
        if IS_MACOS and not self.keyboard.request_accessibility(prompt=False):
            self.keyboard.unregister(self._hotkey_handle)
            self._hotkey_handle = None
            self.cfg["hotkey"] = normalized
            self.hotkey_error = "Accessibility permission required for the hotkey"
            self.save_config()
            self._refresh_menu()
            self._schedule_accessibility_retry()
            return
        new_handle = self.keyboard.register(normalized, self.on_hotkey)
        old_handle, self._hotkey_handle = self._hotkey_handle, new_handle
        self.keyboard.unregister(old_handle)
        self.cfg["hotkey"] = normalized
        self.hotkey_error = None
        self.save_config()
        print(f"[cfg] hotkey = {normalized}")
        self._refresh_menu()

    # -- tray helpers -------------------------------------------------------
    def _on_icon_thread(self, callback):
        """Keep AppKit-backed pystray mutations on macOS's main thread."""
        if IS_MACOS:
            self.ui.call(callback)
        else:
            callback()

    def _refresh_menu(self):
        def refresh():
            if self.icon is not None:
                try:
                    self.icon.update_menu()
                except Exception:
                    pass

        self._on_icon_thread(refresh)

    def set_state(self, state):
        self.state = state

        def refresh():
            if self.icon is not None:
                current = self.state
                self.icon.icon = ICONS[current]
                self.icon.title = f"WhisperType — {current}"
                try:
                    self.icon.update_menu()
                except Exception:
                    pass

        self._on_icon_thread(refresh)

    # -- model --------------------------------------------------------------
    def _load_model_guarded(self):
        """Never leave the tray stuck on 'loading' if the model can't load."""
        try:
            self.load_model()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.load_error = str(e)
            self.set_state("error")

    def reload_model(self):
        """Swap to the model/device currently in cfg, without restarting."""
        self.load_error = None
        self.set_state("loading")
        old = self.model
        self.model = None
        del old
        self._load_model_guarded()

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
    def _store_audio(self, indata, status=None):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.frames.append(indata.copy())
        if self.cfg.get("show_overlay", True):
            # RMS -> perceptual-ish level for the waveform bars.
            rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
            self.overlay.push_level(min(1.0, (rms ** 0.5) * 3.2))

    def _audio_cb(self, indata, frames, time_info, status):
        self._store_audio(indata, status)

    def _read_audio_blocks(self, stream, stop_event, blocksize):
        """Read microphone frames without a CFFI callback on macOS.

        CFFI callbacks require write+execute memory in frozen applications,
        which conflicts with Apple's hardened runtime.  PortAudio's blocking
        read API avoids that entitlement and keeps the bundle hardened.
        """
        try:
            while not stop_event.is_set():
                indata, overflowed = stream.read(blocksize)
                self._store_audio(
                    indata,
                    "input overflow" if overflowed else None,
                )
        except Exception as e:
            if not stop_event.is_set() and not self._quitting:
                self.recording = False
                try:
                    stream.abort()
                    stream.close()
                except Exception:
                    pass
                if self.stream is stream:
                    self.stream = None
                self.load_error = f"Microphone unavailable: {e}"
                self.overlay.hide()
                self.set_state("error")
                print(f"[audio] {self.load_error}")

    def _stop_capture_reader(self):
        stop_event = self._capture_stop
        capture_thread = self._capture_thread
        if stop_event is not None:
            stop_event.set()
        if (capture_thread is not None
                and capture_thread is not threading.current_thread()):
            capture_thread.join(timeout=1.0)
        self._capture_stop = None
        self._capture_thread = None

    def start_recording(self):
        self.frames = []
        blocksize = int(self.cfg["sample_rate"] * 0.03)   # ~30 ms per bar
        stream = None
        try:
            kwargs = {
                "samplerate": self.cfg["sample_rate"],
                "channels": 1,
                "dtype": "float32",
                "blocksize": blocksize,
            }
            if not IS_MACOS:
                kwargs["callback"] = self._audio_cb
            stream = sd.InputStream(**kwargs)
            stream.start()
        except Exception:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise

        self.stream = stream
        self.recording = True
        if IS_MACOS:
            self._capture_stop = threading.Event()
            self._capture_thread = threading.Thread(
                target=self._read_audio_blocks,
                args=(stream, self._capture_stop, blocksize),
                daemon=True,
                name="WhisperTypeAudioReader",
            )
            self._capture_thread.start()
        self.set_state("recording")
        if self.cfg.get("show_overlay", True):
            self.overlay.show("Listening")
        print("[rec] Recording… press the hotkey again to stop.")

    def stop_recording(self):
        self.recording = False
        self._stop_capture_reader()
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
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.load_error = f"Transcription failed: {e}"
            self.set_state("error")
            return
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
            if self.keyboard.paste():
                self.hotkey_error = None
                print("[out] Pasted into focused app.")
            else:
                self.hotkey_error = (
                    "Accessibility permission required; text copied to clipboard"
                )
                print("[out] Accessibility access is missing; copied to clipboard.")
                self._refresh_menu()
        else:
            print("[out] Copied to clipboard.")

    # -- hotkey dispatch ----------------------------------------------------
    def on_hotkey(self):
        # Never run audio or NumPy work inside pynput's listener callback.  An
        # exception escaping that callback permanently stops the listener.
        threading.Thread(
            target=self._handle_hotkey,
            daemon=True,
            name="WhisperTypeHotkey",
        ).start()

    def _handle_hotkey(self):
        with self.lock:
            try:
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
                            name="WhisperTypeTranscription",
                        ).start()
            except Exception as e:
                import traceback

                traceback.print_exc()
                self.recording = False
                self._stop_capture_reader()
                stream, self.stream = self.stream, None
                if stream is not None:
                    try:
                        stream.abort()
                        stream.close()
                    except Exception:
                        pass
                self.load_error = f"Microphone unavailable: {e}"
                self.overlay.hide()
                self.set_state("error")
                print(f"[audio] {self.load_error}")

    # -- lifecycle ----------------------------------------------------------
    def _activate_configured_hotkey(self):
        if self._hotkey_handle is not None:
            return True
        try:
            self._hotkey_handle = self.keyboard.register(
                self.cfg["hotkey"], self.on_hotkey
            )
            self.hotkey_error = None
            self._refresh_menu()
            print(f"[hotkey] Active: {self.cfg['hotkey']}")
            return True
        except Exception as e:
            self.hotkey_error = f"Hotkey unavailable: {e}"
            self._refresh_menu()
            print(f"[hotkey] {self.hotkey_error}")
            return False

    def _schedule_accessibility_retry(self, delay_ms=1500):
        if (not IS_MACOS or self._quitting
                or self._accessibility_retry_scheduled
                or self.ui.root is None):
            return
        self._accessibility_retry_scheduled = True

        def poll():
            self._accessibility_retry_scheduled = False
            if self._quitting or self._hotkey_handle is not None:
                return
            if self.keyboard.request_accessibility(prompt=False):
                self._activate_configured_hotkey()
            else:
                self.hotkey_error = "Accessibility permission required for the hotkey"
                self._refresh_menu()
                self._schedule_accessibility_retry()

        self.ui.root.after(delay_ms, poll)

    def toggle_output_mode(self, icon, item):
        self.cfg["output_mode"] = (
            "clipboard" if self.cfg["output_mode"] == "paste" else "paste"
        )
        self.save_config()
        print(f"[cfg] output_mode = {self.cfg['output_mode']}")

    def open_settings(self, icon, item):
        # Marshal the Tk dialog to its owning UI thread. Only one window at a
        # time is allowed.
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
        def status_text(item):
            if self.load_error:
                return f"Error: {self.load_error[:60]}"
            if self.hotkey_error:
                return self.hotkey_error[:70]
            return f"Status: {self.state}"

        items = [
            pystray.MenuItem(
                status_text,
                None, enabled=False),
            pystray.MenuItem(lambda item: f"Model: {self.cfg['model']}", None, enabled=False),
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
        ]
        if IS_MACOS:
            items.append(pystray.MenuItem(
                "Accessibility Settings…",
                lambda icon, item: open_accessibility_settings(),
            ))
        items.extend([pystray.Menu.SEPARATOR, pystray.MenuItem("Quit", self.quit)])
        return pystray.Menu(*items)

    # -- settings window ----------------------------------------------------
    def _run_settings_window(self):
        """Builds the settings window. Must run on the shared UI thread."""
        import tkinter as tk
        from tkinter import ttk, messagebox

        try:
            root = tk.Toplevel(self.ui.root)
            root.title("WhisperType Settings")
            root.resizable(False, False)
            if IS_MACOS:
                try:
                    from AppKit import NSApp
                    NSApp.activateIgnoringOtherApps_(True)
                except Exception:
                    pass
            try:
                ico = os.path.join(_bundle_dir(), "icon.ico")
                if IS_WINDOWS and os.path.isfile(ico):
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
                        combo = self.keyboard.capture()
                    except Exception as e:
                        print(f"[hotkey] capture failed: {e}")
                        combo = None

                    def done():
                        if combo:
                            hotkey_var.set(combo)
                        record_btn.config(text="Record", state="normal")

                    self.ui.call(done)

                threading.Thread(target=capture, daemon=True).start()

            record_btn.config(command=record_keys)
            record_btn.grid(row=1, column=1, padx=(8, 0), pady=(2, 0))
            ttk.Label(
                frm, text=f"e.g. {default_hotkey()}", foreground="#777"
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 10))

            # --- Speech model ---
            installed = _local_models()

            def model_label(name):
                note = MODEL_NOTES.get(name, "")
                status = ("installed" if name in installed
                          else f"download {_human_mb(MODELS[name][1])}")
                return f"{name} — {note} ({status})"

            model_names = list(MODELS)
            labels = [model_label(n) for n in model_names]
            ttk.Label(frm, text="Speech model").grid(row=3, column=0, sticky="w")
            model_var = tk.StringVar(
                value=model_label(self.cfg["model"])
                if self.cfg["model"] in MODELS else labels[0])
            ttk.Combobox(
                frm, textvariable=model_var, state="readonly", width=50,
                values=labels,
            ).grid(row=4, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Acceleration ---
            if IS_MACOS:
                gpu_label = None
                cpu_label = "CPU — supported on macOS"
                accel_values = [cpu_label]
                accel_value = cpu_label
            else:
                gpu_ready = cuda_installed()
                gpu_label = ("GPU — NVIDIA (installed)" if gpu_ready
                             else f"GPU — NVIDIA (download {_human_mb(CUDA_MB)})")
                cpu_label = "CPU — works everywhere, slower"
                accel_values = [gpu_label, cpu_label]
                accel_value = gpu_label if self.cfg["device"] == "cuda" else cpu_label
            ttk.Label(frm, text="Acceleration").grid(row=5, column=0, sticky="w")
            accel_var = tk.StringVar(value=accel_value)
            ttk.Combobox(
                frm, textvariable=accel_var, state="readonly", width=50,
                values=accel_values,
            ).grid(row=6, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Output mode ---
            ttk.Label(frm, text="Output").grid(row=7, column=0, sticky="w")
            output_var = tk.StringVar(value=self.cfg["output_mode"])
            ttk.Combobox(
                frm, textvariable=output_var, state="readonly", width=26,
                values=["paste", "clipboard"],
            ).grid(row=8, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Language ---
            ttk.Label(frm, text="Language (ISO code, e.g. en)").grid(
                row=9, column=0, sticky="w")
            lang_var = tk.StringVar(value=self.cfg["language"])
            ttk.Entry(frm, textvariable=lang_var, width=28).grid(
                row=10, column=0, columnspan=2, sticky="we", pady=(2, 10))

            # --- Overlay ---
            overlay_var = tk.BooleanVar(value=self.cfg.get("show_overlay", True))
            ttk.Checkbutton(
                frm, text="Show waveform overlay while recording",
                variable=overlay_var,
            ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(0, 8))

            # --- Download progress (only shown when something must be fetched) ---
            prog_label = ttk.Label(frm, text="", foreground="#555")
            prog_bar = ttk.Progressbar(frm, mode="determinate", maximum=100)

            # --- Buttons ---
            btns = ttk.Frame(frm)
            btns.grid(row=14, column=0, columnspan=2, sticky="e", pady=(6, 0))

            def on_save():
                requested_hotkey = hotkey_var.get().strip().lower()
                if not requested_hotkey:
                    messagebox.showerror("WhisperType", "Hotkey cannot be empty.")
                    return
                # Validate the hotkey before committing to it.
                try:
                    new_hotkey = self.keyboard.validate(requested_hotkey)
                except Exception as e:
                    messagebox.showerror(
                        "WhisperType", f"'{requested_hotkey}' is not a valid hotkey.\n\n{e}")
                    return
                if new_hotkey != self.cfg["hotkey"]:
                    try:
                        self.apply_hotkey(new_hotkey)
                    except Exception as e:
                        messagebox.showerror(
                            "WhisperType", f"Could not register '{new_hotkey}'.\n\n{e}")
                        return
                self.cfg["output_mode"] = output_var.get()
                self.cfg["language"] = lang_var.get().strip() or "en"
                self.cfg["show_overlay"] = bool(overlay_var.get())
                if not self.cfg["show_overlay"]:
                    self.overlay.hide()

                # Model / acceleration may require a download before they apply.
                new_model = model_names[labels.index(model_var.get())]
                want_cuda = bool(gpu_label and accel_var.get() == gpu_label)
                model_changed = new_model != self.cfg["model"]
                device_changed = want_cuda != (self.cfg["device"] == "cuda")
                needs_dl = (new_model not in _local_models()
                            or (want_cuda and not cuda_installed()))

                if not (model_changed or device_changed):
                    self.save_config()
                    self._refresh_menu()
                    print("[cfg] Settings saved.")
                    close()
                    return

                def apply_choice():
                    self.cfg["model"] = new_model
                    if want_cuda:
                        self.cfg["device"] = "cuda"
                        self.cfg["compute_type"] = "float16"
                    else:
                        self.cfg["device"] = "cpu"
                        self.cfg["compute_type"] = "int8"
                    self.save_config()
                    self._refresh_menu()

                if not needs_dl:
                    apply_choice()
                    print(f"[cfg] Switching to {new_model} on {self.cfg['device']}")
                    threading.Thread(target=self.reload_model, daemon=True).start()
                    close()
                    return

                # Download first, keeping the window open with progress.
                save_btn.config(state="disabled")
                cancel_btn.config(state="disabled")
                prog_label.grid(row=12, column=0, columnspan=2, sticky="w")
                prog_bar.grid(row=13, column=0, columnspan=2, sticky="we", pady=(4, 0))

                base_text = {"msg": "Preparing…"}

                def set_status(msg):
                    base_text["msg"] = msg
                    self.ui.call(lambda: prog_label.config(text=msg))

                def set_progress(done, total):
                    pct = max(0.0, min(100.0, (done / total * 100) if total else 0))
                    text = f"{base_text['msg']}  {_human_mb(done)} of {_human_mb(total)}"
                    self.ui.call(lambda: (prog_bar.config(value=pct),
                                          prog_label.config(text=text)))

                def worker():
                    try:
                        ensure_components(
                            model=new_model, want_cuda=want_cuda,
                            on_status=set_status, on_progress=set_progress)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        error_text = str(e)
                        self.ui.call(lambda error_text=error_text: (
                            messagebox.showerror(
                                "WhisperType",
                                f"Could not download the components.\n\n{error_text}"),
                            save_btn.config(state="normal"),
                            cancel_btn.config(state="normal"),
                            prog_label.grid_remove(),
                            prog_bar.grid_remove()))
                        return
                    applied = threading.Event()

                    def commit_choice():
                        try:
                            apply_choice()
                            close()
                        finally:
                            applied.set()

                    self.ui.call(commit_choice)
                    if not applied.wait(timeout=10):
                        print("[cfg] Timed out applying downloaded components")
                        return
                    print(f"[cfg] Switching to {new_model} on {self.cfg['device']}")
                    self.reload_model()

                threading.Thread(target=worker, daemon=True).start()

            def close():
                self._settings_open = False
                try:
                    root.destroy()
                except Exception:
                    pass

            cancel_btn = ttk.Button(btns, text="Cancel", command=close)
            cancel_btn.grid(row=0, column=0, padx=(0, 8))
            save_btn = ttk.Button(btns, text="Save", command=on_save)
            save_btn.grid(row=0, column=1)
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
        self._quitting = True
        self.recording = False
        self._stop_capture_reader()
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.keyboard.unregister(self._hotkey_handle)
        self._hotkey_handle = None
        # On macOS pystray and Tk share NSApplication.  Stop the menu-bar
        # integration before tearing down Tk's root so pystray can wake and
        # exit the still-valid native event loop cleanly.
        icon.stop()
        self.ui.stop()

    def run(self):
        # Aqua/Tk must own the main thread.  Windows keeps its existing split
        # where pystray owns main and Tk runs in a dedicated UI thread.
        self.ui.start(on_main_thread=IS_MACOS)

        # Register the global hotkey (tracks the handle for live re-binding).
        accessibility_ready = True
        if IS_MACOS:
            accessibility_ready = self.keyboard.request_accessibility(prompt=True)
        if not accessibility_ready:
            self.hotkey_error = "Accessibility permission required for the hotkey"
        else:
            self._activate_configured_hotkey()
        print(f"[ready] Hotkey: {self.cfg['hotkey']}  "
              f"(output: {self.cfg['output_mode']})")

        icon_kwargs = {}
        if IS_MACOS:
            from AppKit import NSApplication
            icon_kwargs["darwin_nsapplication"] = NSApplication.sharedApplication()
        self.icon = pystray.Icon(
            "WhisperType", ICONS["loading"], "WhisperType — loading",
            self.build_menu(), **icon_kwargs,
        )
        if IS_MACOS and not accessibility_ready:
            self._schedule_accessibility_retry()

        # Load the model in the background so the menu-bar/tray icon appears
        # immediately, even while a first-run model download is in progress.
        threading.Thread(target=self._load_model_guarded, daemon=True).start()

        if IS_MACOS:
            self.icon.run_detached(
                setup=lambda icon: self.ui.call(
                    lambda: setattr(icon, "visible", True)
                )
            )
            self.ui.run_mainloop()
        else:
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

# name -> (hugging-face repo, approximate megabytes on disk)
MODELS = {
    "base.en":   ("Systran/faster-whisper-base.en",    150),
    "small.en":  ("Systran/faster-whisper-small.en",   490),
    "medium.en": ("Systran/faster-whisper-medium.en", 1530),
    "large-v3":  ("Systran/faster-whisper-large-v3",  2950),
}

MODEL_NOTES = {
    "base.en":   "fastest, least accurate",
    "small.en":  "fast, good accuracy",
    "medium.en": "slower, very accurate",
    "large-v3":  "best accuracy, multilingual",
}

# Approximate size of the unpacked CUDA libraries.
CUDA_MB = 2080


def _human_mb(mb):
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{int(mb)} MB"


def _dir_size_mb(path):
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def cuda_installed():
    """True if the CUDA runtime libraries are already unpacked."""
    if not IS_WINDOWS:
        return False
    for root in _nvidia_roots():
        if all(os.path.isfile(os.path.join(root, *s.split("/")))
               for _p, _v, s in CUDA_WHEELS):
            return True
    return False

# Immutable PyPI wheels holding the CUDA runtime libraries CTranslate2 needs,
# with a sentinel DLL used to tell whether the wheel is already unpacked.
CUDA_WHEELS = [
    ("nvidia-cublas-cu12", "12.9.2.10", "cublas/bin/cublas64_12.dll"),
    ("nvidia-cudnn-cu12", "9.25.0.15", "cudnn/bin/cudnn64_9.dll"),
    ("nvidia-cuda-nvrtc-cu12", "12.9.86", "cuda_nvrtc/bin/nvrtc64_120_0.dll"),
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
    if not IS_WINDOWS:
        raise RuntimeError("NVIDIA CUDA acceleration is only supported on Windows")
    import tempfile
    import zipfile
    out = os.path.join(dest_root, "nvidia")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for pkg, ver, sentinel in CUDA_WHEELS:
            if os.path.isfile(os.path.join(out, *sentinel.split("/"))):
                print(f"[fetch] {pkg} already installed", flush=True)
                continue
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
    """Download a faster-whisper model into <dest_root>/models/<name>.

    Uses faster_whisper's own downloader rather than a hand-written file list:
    the required files differ per model (large-v3 ships vocabulary.json where
    medium.en ships vocabulary.txt), and getting that wrong produces a model
    directory that fails to load with "Cannot load the vocabulary".
    """
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}' (choose from {list(MODELS)})")
    import shutil
    import tempfile
    import uuid

    models_root = os.path.join(dest_root, "models")
    out = os.path.join(models_root, name)
    if _model_is_complete(out):
        print(f"[fetch] model '{name}' already installed", flush=True)
        return

    staging_root = os.path.join(models_root, ".downloads")
    os.makedirs(staging_root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f"{name}-", dir=staging_root)
    backup = None
    print(f"[fetch] downloading model '{name}' (this can take a few minutes)",
          flush=True)
    try:
        from faster_whisper import download_model

        download_model(name, output_dir=staging)
        _verify_model(staging, name)
        if os.path.exists(out):
            backup = os.path.join(
                staging_root,
                f"{name}-previous-{uuid.uuid4().hex}",
            )
            os.replace(out, backup)
        os.replace(staging, out)
        staging = None
    except BaseException:
        if backup and os.path.exists(backup) and not os.path.exists(out):
            os.replace(backup, out)
            backup = None
        raise
    finally:
        if staging and os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
        if backup and os.path.exists(backup):
            shutil.rmtree(backup, ignore_errors=True)
    print(f"[fetch] model '{name}' ready in {out}", flush=True)


def _verify_model(path, name):
    """Fail loudly if a model directory is missing what CTranslate2 needs."""
    if not os.path.isfile(os.path.join(path, "model.bin")):
        raise RuntimeError(f"model '{name}' is missing model.bin")
    if not _model_is_complete(path):
        raise RuntimeError(
            f"model '{name}' has no vocabulary/tokenizer file — it would fail "
            f"to load")


def ensure_components(model=None, want_cuda=False, on_status=None, on_progress=None):
    """Download whatever is missing for the requested setup.

    on_status(text) and on_progress(done_mb, total_mb) let a GUI show what is
    happening; progress is estimated by watching the target folder grow, since
    the underlying downloaders don't expose a byte count.
    """
    def status(msg):
        print(f"[components] {msg}", flush=True)
        if on_status:
            on_status(msg)

    dest = _data_dir()
    jobs = []
    if want_cuda and not cuda_installed():
        jobs.append(("cuda", os.path.join(dest, "nvidia"), CUDA_MB))
    if model and model not in _local_models():
        if model not in MODELS:
            raise ValueError(f"unknown model '{model}'")
        jobs.append((model, os.path.join(dest, "models", model), MODELS[model][1]))

    for kind, path, total_mb in jobs:
        label = "GPU acceleration files" if kind == "cuda" else f"model '{kind}'"
        status(f"Downloading {label} ({_human_mb(total_mb)})…")
        stop = threading.Event()

        def watch(p=path, t=total_mb):
            while not stop.wait(0.5):
                if on_progress:
                    try:
                        on_progress(_dir_size_mb(p) if os.path.isdir(p) else 0, t)
                    except Exception:
                        pass

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            if kind == "cuda":
                fetch_cuda(dest)
            else:
                fetch_model(kind, dest)
        finally:
            stop.set()
        if on_progress:
            on_progress(total_mb, total_mb)
    status("Ready.")


def fetch_main(argv):
    """`--fetch [--model NAME] [--cuda]` — used by the installer."""
    import traceback
    dest = _data_dir()
    model = None
    want_cuda = "--cuda" in argv
    if want_cuda and not IS_WINDOWS:
        print("[fetch] CUDA acceleration is only supported on Windows", flush=True)
        return 2
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 < len(argv):
            model = argv[i + 1]
    print(f"[fetch] data dir : {dest}", flush=True)
    print(f"[fetch] config   : {CONFIG_PATH}", flush=True)
    print(f"[fetch] model={model} cuda={want_cuda}", flush=True)
    try:
        if want_cuda:
            fetch_cuda(dest)
        if model and model.lower() != "none":
            fetch_model(model, dest)
        # Record the choice so the app uses it on first launch. Done even if
        # only CUDA was requested, so device/compute_type stay consistent.
        cfg = load_config()
        if model and model.lower() != "none":
            cfg["model"] = model
        if want_cuda:
            cfg["device"], cfg["compute_type"] = "cuda", "float16"
        else:
            cfg["device"], cfg["compute_type"] = "cpu", "int8"
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[fetch] wrote config: model={cfg['model']} device={cfg['device']}",
              flush=True)
        print("[fetch] DONE", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("[fetch] FAILED", flush=True)
        return 1


def overlaytest(seconds=4.0):
    """Show the overlay with simulated levels — verifies the panel works in a
    frozen build without needing the mic or hotkey."""
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
        ui.start(on_main_thread=IS_MACOS)
        ov = Overlay(ui)
        info = {}
        finished = threading.Event()

        def exercise():
            ov.show("Listening")
            t0 = time.time()
            while time.time() - t0 < seconds * 0.6:
                t = time.time() - t0
                level = min(
                    1.0,
                    abs(math.sin(t * 2.2)) * (0.5 + 0.5 * random.random()),
                )
                ov.push_level(level)
                time.sleep(0.03)

            probed = threading.Event()

            def probe():
                info["geometry"] = ov.win.geometry()
                info["mapped"] = bool(ov.win.winfo_ismapped())
                info["nonactivating"] = ov._nonactivating
                probed.set()

            ui.call(probe)
            probed.wait(timeout=2)
            ov.set_status("Transcribing…")
            time.sleep(seconds * 0.4)
            ov.hide()
            time.sleep(0.4)
            finished.set()
            ui.stop()

        worker = threading.Thread(target=exercise, daemon=True)
        worker.start()
        if IS_MACOS:
            ui.run_mainloop()
        finished.wait(timeout=seconds + 4)
        for k, v in info.items():
            log(f"  {k} = {v}")
        ok = info.get("mapped") and info.get("nonactivating")
        log("OVERLAYTEST OK" if ok else "OVERLAYTEST FAILED — window/styles wrong")
        if not ok:
            sys.exit(1)
    except Exception:
        log("OVERLAYTEST FAILED:\n" + traceback.format_exc())
        sys.exit(1)


def menutest(seconds=1.0):
    """Exercise pystray's detached AppKit integration and clean shutdown."""
    if not IS_MACOS:
        print("MENUTEST skipped — macOS only")
        return

    import traceback

    ui = None
    icon = None
    try:
        from AppKit import NSApplication

        ui = UiThread()
        ui.start(on_main_thread=True)
        icon = pystray.Icon(
            "WhisperTypeSmokeTest",
            ICONS["idle"],
            "WhisperType smoke test",
            pystray.Menu(pystray.MenuItem("Smoke test", None, enabled=False)),
            darwin_nsapplication=NSApplication.sharedApplication(),
        )
        result = {"visible": False, "stopped": False}

        def finish():
            result["visible"] = bool(icon.visible)
            icon.stop()
            result["stopped"] = True
            ui.stop()

        ui.root.after(int(seconds * 1000), finish)
        icon.run_detached(
            setup=lambda tray_icon: ui.call(
                lambda: setattr(tray_icon, "visible", True)
            )
        )
        ui.run_mainloop()
        if not all(result.values()):
            raise RuntimeError(f"menu-bar lifecycle incomplete: {result}")
        print("MENUTEST OK")
    except Exception:
        traceback.print_exc()
        print("MENUTEST FAILED")
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        if ui is not None:
            ui.stop()
        sys.exit(1)


def check():
    """Import the heavy dependencies and report versions, without loading a
    model or touching the network. Used by CI to catch a frozen build that is
    missing hidden imports or bundled data files."""
    import traceback
    try:
        print(f"frozen   : {getattr(sys, 'frozen', False)}")
        print(f"bundle   : {_bundle_dir()}")
        import ctranslate2
        print(f"ctranslate2: {ctranslate2.__version__}")
        import faster_whisper
        print(f"faster_whisper: {faster_whisper.__version__}")
        from faster_whisper import WhisperModel  # noqa: F401
        import sounddevice, pystray, pyperclip  # noqa: F401
        if IS_MACOS:
            import AppKit, Quartz, pynput  # noqa: F401
            from pynput import keyboard as pynput_keyboard  # noqa: F401
            print("keyboard : pynput/darwin")
        else:
            import keyboard  # noqa: F401
            print("keyboard : keyboard/win32")
        import tkinter
        from tkinter import ttk  # noqa: F401
        r = tkinter.Tk(); r.withdraw(); r.destroy()
        print("tkinter  : ok")
        import numpy
        print(f"numpy    : {numpy.__version__}")
        print(f"models known: {list(MODELS)}")
        print("CHECK OK")
        return 0
    except Exception:
        traceback.print_exc()
        print("CHECK FAILED")
        return 1


def main():
    # A frozen app re-executes its own binary for multiprocessing helpers
    # (faster-whisper's resource tracker); without this each helper launches
    # another full copy of WhisperType.
    multiprocessing.freeze_support()
    if "--check" in sys.argv:
        sys.exit(check())
    if "--fetch" in sys.argv:
        sys.exit(fetch_main(sys.argv))
    if "--overlaytest" in sys.argv:
        overlaytest()
        return
    if "--menutest" in sys.argv:
        menutest()
        return
    if "--selftest" in sys.argv:
        selftest()
        return
    cfg = load_config()
    WhisperType(cfg).run()


if __name__ == "__main__":
    main()
