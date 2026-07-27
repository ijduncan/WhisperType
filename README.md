# WhisperType — Offline Voice-to-Text for Windows

A tiny tray app that turns your speech into text **entirely on your machine**,
similar to Windows dictation (Win+H) — but offline, powered by
[Whisper](https://github.com/openai/whisper) via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Download

**[⬇️ Download the Windows installer (~75 MB)](https://github.com/ijduncan/WhisperType/releases/latest)**

The installer needs no Python. During setup you pick which **speech model** to
install and whether to add **GPU acceleration**, and it downloads just those:

| Choice | Download |
|--------|----------|
| `base.en` — fastest, least accurate | ~150 MB |
| `small.en` — fast, good accuracy | ~490 MB |
| `medium.en` — slower, very accurate *(default)* | ~1.5 GB |
| `large-v3` — best accuracy, multilingual | ~3.1 GB |
| GPU acceleration (NVIDIA only) | +~1.4 GB |

Internet is needed **during installation only** — afterwards WhisperType runs
entirely offline. It installs per-user (no admin prompt) and offers a "start at
sign-in" option. Because the `.exe` is unsigned, Windows SmartScreen may warn on
first run — click *More info → Run anyway*.

Prefer to build it yourself, or run without installing? See
[Running from source](#running-from-source) and
[Building the installer](#building-the-installer) below.

## How it works

1. Press the global hotkey (**Ctrl+Alt+Win+Space** by default) to start recording.
   A small **waveform panel** appears at the bottom of the screen, showing your
   voice live so you can see it's listening.
2. Speak.
3. Press the hotkey again to stop. The panel switches to a "Transcribing…" pulse
   while your speech is transcribed locally, then the text is **pasted into
   whatever app is focused** (or copied to the clipboard).

The overlay never takes keyboard focus and is click-through, so it can't
interfere with the app you're typing into. Turn it off from the tray menu or
Settings if you'd rather just have the tray dot.

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
- **Show waveform overlay** — toggle the recording panel.
- **Settings…** — a small window to change:
  - **Global hotkey** (type a combo, or click *Record* and press the keys)
  - **Speech model** — switch between `base.en`, `small.en`, `medium.en` and
    `large-v3`. Each entry shows whether it is installed or how large the
    download is; picking one that isn't installed downloads it with a progress
    bar and then swaps to it **without restarting the app**.
  - **Acceleration** — GPU (NVIDIA) or CPU. Choosing GPU when the CUDA libraries
    aren't present downloads those too.
  - Output mode, language, and whether to show the waveform overlay

  So if a model turns out to be too slow, or GPU was chosen on a machine with no
  NVIDIA card, it can be changed here instead of reinstalling. Settings are
  saved to `config.json`.

## Running from source

Requires **Python 3.12** (64-bit) and, for GPU, an NVIDIA GPU with a recent
driver. The CUDA cuBLAS/cuDNN libraries are installed as pip packages
(`requirements-gpu.txt`) — no separate CUDA Toolkit needed.

1. Double-click **`setup.bat`**. This creates a virtual environment and installs
   `requirements.txt` plus the ~1.3 GB CUDA libraries from
   `requirements-gpu.txt`. Run `setup.bat cpu` to skip the CUDA download.
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

The app can be packaged into a Windows installer (`WhisperType-Setup.exe`) that
needs no Python.

1. Complete [Running from source](#running-from-source) first (the build reuses
   the `.venv`).
2. Install the build tools once:
   - `.venv\Scripts\python.exe -m pip install pyinstaller`
   - `winget install JRSoftware.InnoSetup`
3. Run **`build_installer.bat`**. It runs PyInstaller then Inno Setup and writes
   the installer to `C:\WhisperTypeBuild\Output\WhisperType-Setup.exe`.

That produces the **slim** installer (~75 MB): the model and CUDA libraries are
downloaded during installation by `WhisperTypeFetch.exe`, based on what the user
picks in the setup wizard.

To build a **fully offline** installer instead (~2.3 GB, nothing downloaded),
run `build_installer.bat bundled`. It sets `WHISPERTYPE_BUNDLE=1`, which makes
`whispertype.spec` embed the model and the CUDA DLLs.

The build outputs to `C:\WhisperTypeBuild` (outside OneDrive) on purpose —
OneDrive locks files mid-build and would try to sync the output.

### Automated builds

[`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml)
builds the slim installer on GitHub Actions. It runs on pushes to `main`, on
pull requests, on `v*` tags, and on demand from the Actions tab.

- The built installer is uploaded as a workflow **artifact** on every run.
- On a **`v*` tag** it is also attached to the matching GitHub Release, so
  `git tag v1.3.0 && git push --tags` publishes a build.
- CI installs only `requirements.txt` — not the CUDA libraries — because the
  slim installer downloads those at install time. It then runs
  `WhisperTypeFetch.exe --check`, which imports faster-whisper, CTranslate2 and
  tkinter inside the frozen bundle to catch packaging breakage.

Cross-building is not possible: PyInstaller bundles the host OS's interpreter
and native libraries, so a macOS or Linux build has to run on that OS (a CI
runner is enough — you don't need the hardware). The app is currently
Windows-only in any case: the global hotkey, the clipboard paste, and the
non-activating overlay window all use Win32 APIs.

### Fetching components manually

`WhisperTypeFetch.exe` (or `python whispertype.py`) accepts:

```
WhisperTypeFetch.exe --fetch --model medium.en --cuda
```

It downloads into `%LOCALAPPDATA%\WhisperType\` — `models\<name>\` for models and
`nvidia\<lib>\bin\` for the CUDA DLLs. The app prefers those over anything
bundled, so you can add or swap models on an installed copy.

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
| `show_overlay` | `true`             | Live waveform panel while recording. |

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

## License

Released under the [MIT License](LICENSE).
