# WhisperType — Offline Voice-to-Text for Windows

A tiny tray app that turns your speech into text **entirely on your machine**,
similar to Windows dictation (Win+H) — but offline, powered by
[Whisper](https://github.com/openai/whisper) via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Download

**[⬇️ Download the Windows installer (Google Drive, ~2.3 GB)](https://drive.google.com/file/d/1at1otQpeNee90Rez587gc_9DLl__SVr4/view?usp=drive_link)**

The installer needs no Python and bundles GPU support and the `medium.en` model,
so it works fully offline from the first launch. It installs per-user (no admin
prompt) and offers a "start at sign-in" option. Because the `.exe` is unsigned,
Windows SmartScreen may warn on first run — click *More info → Run anyway*.

Prefer to build it yourself, or run without installing? See
[Running from source](#running-from-source) and
[Building the installer](#building-the-installer) below.

## How it works

1. Press the global hotkey (**Ctrl+Alt+Win+Space** by default) to start recording.
2. Speak.
3. Press the hotkey again to stop. Your speech is transcribed locally and
   **pasted into whatever app is focused** (or copied to the clipboard).

The tray icon shows the current state:

| Color  | Meaning        |
|--------|----------------|
| Gray   | Loading model  |
| Blue   | Idle / ready   |
| Red    | Recording      |
| Orange | Transcribing   |

## Settings

Right-click the tray icon for:

- **Auto-paste into focused app** — toggle between pasting and clipboard-only.
- **Settings…** — a small window to change the **global hotkey** (type a combo or
  click *Record* and press the keys), the output mode, and the language. Hotkey
  and output changes apply immediately. Settings are saved to `config.json`.

## Running from source

Requires **Python 3.12** (64-bit) and, for GPU, an NVIDIA GPU with a recent
driver. The CUDA cuBLAS/cuDNN libraries are installed as pip packages
(see `requirements.txt`) — no separate CUDA Toolkit needed.

1. Double-click **`setup.bat`** (or run `python -m venv .venv` and
   `.venv\Scripts\python -m pip install -r requirements.txt`). This creates a
   virtual environment and installs the dependencies (a few minutes).
2. Start the app:
   - **`run.bat`** — with a console window (useful to see logs).
   - **`run-hidden.vbs`** — quietly, no console window.
3. The **first launch** downloads the `medium.en` model (~1.5 GB) to your
   Hugging Face cache. After that it works fully offline.

To launch automatically at login, press `Win+R`, type `shell:startup`, and put a
shortcut to `run-hidden.vbs` in that folder. (The installer offers this as a
checkbox instead.)

If you don't have an NVIDIA GPU, set `"device": "cpu"` and
`"compute_type": "int8"` in `config.json` (the app also falls back to CPU
automatically if GPU init fails).

## Building the installer

The app can be packaged into a single Windows installer
(`WhisperType-Setup.exe`) that needs no Python and bundles the GPU libraries and
the `medium.en` model, so it works offline from first launch.

1. Complete [Running from source](#running-from-source) first (the build reuses
   the `.venv`).
2. Install the build tools once:
   - `.venv\Scripts\python.exe -m pip install pyinstaller`
   - `winget install JRSoftware.InnoSetup`
3. Fetch the model to bundle (it's git-ignored, so a fresh clone won't have it):
   ```
   .venv\Scripts\python -c "from faster_whisper import download_model; download_model('medium.en', output_dir=r'models/medium.en')"
   ```
4. Run **`build_installer.bat`**. It runs PyInstaller then Inno Setup and writes
   the installer to `C:\WhisperTypeBuild\Output\WhisperType-Setup.exe`.

The build outputs to `C:\WhisperTypeBuild` (outside OneDrive) on purpose —
OneDrive locks files mid-build and also would try to sync the ~3 GB output.

The installer installs per-user (no admin prompt), adds Start Menu / optional
desktop shortcuts, and offers a **"Start WhisperType automatically when I sign
in"** checkbox (a per-user `Run` registry entry, removed on uninstall).

## Configuration

Edit **`config.json`** (created on first run):

| Key            | Default            | Notes |
|----------------|--------------------|-------|
| `hotkey`       | `ctrl+alt+windows+space` | Any [keyboard](https://github.com/boppreh/keyboard) combo, e.g. `ctrl+alt+d`. |
| `model`        | `medium.en`        | `base.en`, `small.en`, `medium.en`, or multilingual `small`, `medium`, `large-v3`. |
| `language`     | `en`               | Transcription language. |
| `device`       | `cuda`             | `cuda` uses your NVIDIA GPU; set to `cpu` if you don't have one. |
| `compute_type` | `float16`          | `float16` on GPU; use `int8` on CPU. |
| `output_mode`  | `paste`            | `paste` into focused app, or `clipboard` only. |

You can also toggle auto-paste from the tray icon's right-click menu.

## Notes

- **GPU:** the app runs on an NVIDIA GPU by default (`device: cuda`). The
  required cuBLAS/cuDNN libraries ship in the venv, and their DLL paths are wired
  up automatically at startup. If GPU init ever fails, it falls back to CPU
  automatically. On the GPU, `medium.en` transcribes a sentence in well under a
  second.
- On CPU, `medium.en` takes a few seconds per sentence; switch to `small.en` for
  faster results with slightly lower accuracy.
- The `keyboard` library captures a global hotkey; if the hotkey ever doesn't
  register, try running `run.bat` as administrator.
