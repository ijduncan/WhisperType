# WhisperType — Offline Voice-to-Text for Windows and macOS

A tiny tray/menu-bar app that turns your speech into text **entirely on your
machine**, similar to built-in dictation — but offline, powered by
[Whisper](https://github.com/openai/whisper) via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Download

### Windows

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

### macOS — Apple silicon preview

The Apple silicon preview is built as the `WhisperType-macOS-arm64` artifact by
the macOS GitHub Actions workflow. A public download will be added after the app
has a Developer ID signature and Apple notarization.

The macOS build supports M-series Macs on macOS 14 or later. Open the DMG and
drag WhisperType to Applications. The current preview is ad-hoc signed, so on
first launch you may need to right-click the app and choose **Open**, or approve
it under **System Settings → Privacy & Security**. A Developer ID signature and
notarization are still required before calling the macOS build production-ready.

macOS asks for two permissions:

- **Microphone** — records only while the listening panel is visible.
- **Accessibility** — listens for the global shortcut and sends `Cmd+V` for
  auto-paste. If this is not granted, transcribed text remains on the clipboard.

Prefer to build it yourself, or run without installing? See
[Running from source](#running-from-source) and
[Building distributable apps](#building-distributable-apps) below.

## How it works

1. Press the global hotkey (**Ctrl+Alt+Win+Space** on Windows or
   **Cmd+Shift+Space** on macOS) to start recording.
   A small **waveform panel** appears at the bottom of the screen, showing your
   voice live so you can see it's listening.
2. Speak.
3. Press the hotkey again to stop. The panel switches to a "Transcribing…" pulse
   while your speech is transcribed locally, then the text is **pasted into
   whatever app is focused** (or copied to the clipboard).

The overlay never takes keyboard focus and is click-through, so it can't
interfere with the app you're typing into. Turn it off from the tray menu or
Settings if you'd rather just have the tray dot.

The tray/menu-bar icon shows the current state:

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
  - **Acceleration** — GPU (NVIDIA) or CPU on Windows; CPU on macOS. Choosing
    GPU on Windows when the CUDA libraries aren't present downloads those too.
  - Output mode, language, and whether to show the waveform overlay

  So if a model turns out to be too slow, or GPU was chosen on a machine with no
  NVIDIA card, it can be changed here instead of reinstalling. Settings are
  saved to `config.json`.

## Running from source

Requires **Python 3.12** (64-bit).

### Windows

For GPU acceleration, use an NVIDIA GPU with a recent driver. The CUDA
cuBLAS/cuDNN libraries are installed as pip packages (`requirements-gpu.txt`) —
no separate CUDA Toolkit is needed.

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

### macOS — Apple silicon

1. On macOS 14 or later, install a native Apple silicon build of Python 3.12.
2. Run `bash setup_macos.sh`.
3. Start the app with `.venv/bin/python whispertype.py`.
4. Approve Microphone and Accessibility access when macOS asks. When running
   from a terminal, macOS may list that terminal or Python under the permission
   rather than WhisperType.

The macOS build uses CPU/int8 transcription and defaults to the smaller
`small.en` model for a faster first download. CTranslate2 does not provide its
CUDA backend on macOS.

## Building distributable apps

### Windows

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

### macOS — Apple silicon

On an Apple silicon Mac with native Python 3.12:

1. Run `bash setup_macos.sh` once.
2. Run `bash build_macos.sh`.
3. Find the app and DMG under `dist/macos/arm64/`.

The build is native `arm64`, keeps speech models outside the signed app under
`~/Library/Application Support/WhisperType`, creates a menu-bar-only `.app`,
ad-hoc signs it for development, and packages
`WhisperType-macOS-arm64.dmg`. Public distribution still needs an Apple
Developer ID certificate and notarization.

#### Mac validation checklist

After copying the app to Applications:

1. Launch it and approve Microphone and Accessibility access. If Accessibility
   is granted while WhisperType is open, the app detects it and enables the
   hotkey automatically.
2. Confirm WhisperType appears only in the menu bar, not the Dock, and finishes
   downloading the default `small.en` model on first launch.
3. In Notes or TextEdit, press **Cmd+Shift+Space**, speak, and press the shortcut
   again. Confirm the overlay never takes focus and the text is pasted.
4. Disable auto-paste from the menu, dictate again, and confirm the result is on
   the clipboard without being pasted.
5. Open Settings, change the shortcut, save, and confirm the new shortcut works.
6. Quit and relaunch. Confirm the settings and downloaded model are reused.

### Automated builds

The GitHub Actions workflows build the slim Windows installer and Apple silicon
macOS DMG. They run on pushes to `main`, pull requests, `v*` tags, and on demand
from the Actions tab:

- [`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml)
- [`.github/workflows/build-macos.yml`](.github/workflows/build-macos.yml)

- Each build is uploaded as a workflow **artifact** on every run.
- On a **`v*` tag**, the Windows installer is attached to the matching GitHub
  Release. The ad-hoc-signed macOS preview remains an Actions artifact until
  Developer ID signing and notarization are configured.
- Windows CI runs `WhisperTypeFetch.exe --check`. macOS CI runs the frozen
  app's `--check`, verifies its code signature and property list, then inspects
  every bundled Mach-O file for an arm64 slice and a macOS 14-compatible
  deployment target.

Cross-building is not possible: PyInstaller bundles the host OS's interpreter
and native libraries, so each app must be built on its target OS and CPU
architecture. The macOS workflow therefore runs natively on Apple silicon.

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
| `hotkey`       | Windows: `ctrl+alt+windows+space`; macOS: `cmd+shift+space` | Global recording shortcut. |
| `model`        | Windows: `medium.en`; macOS: `small.en` | `base.en`, `small.en`, `medium.en`, or multilingual `small`, `medium`, `large-v3`. |
| `language`     | `en`               | Transcription language. |
| `device`       | Windows: `cuda`; macOS: `cpu` | `cuda` uses an NVIDIA GPU on Windows; macOS is CPU-only. |
| `compute_type` | Windows: `float16`; macOS: `int8` | `float16` on GPU; `int8` on CPU. |
| `output_mode`  | `paste`            | `paste` into focused app, or `clipboard` only. |
| `show_overlay` | `true`             | Live waveform panel while recording. |

You can also toggle auto-paste from the tray icon's right-click menu.

## Notes

- **Windows GPU:** the app runs on an NVIDIA GPU by default (`device: cuda`). The
  required cuBLAS/cuDNN libraries ship in the venv, and their DLL paths are wired
  up automatically at startup. If GPU init ever fails, it falls back to CPU
  automatically. On the GPU, `medium.en` transcribes a sentence in well under a
  second.
- On CPU, `medium.en` takes a few seconds per sentence; switch to `small.en` for
  faster results with slightly lower accuracy.
- On macOS, WhisperType stores its settings and downloaded models in
  `~/Library/Application Support/WhisperType`.
- Windows uses the `keyboard` library for its global hotkey; if it does not
  register, try running `run.bat` as administrator. macOS uses `pynput` and
  requires Accessibility permission instead.

## License

Released under the [MIT License](LICENSE).
