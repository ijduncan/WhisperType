# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for WhisperType (GPU + bundled medium.en model).
import os
import importlib.util
from PyInstaller.utils.hooks import collect_all

SPECPATH = os.path.abspath(os.getcwd())

datas, binaries, hiddenimports = [], [], ["pystray._win32"]

# Packages that need full collection (code + data + native libs).
for pkg in ("ctranslate2", "faster_whisper", "onnxruntime", "av"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


def collect_dir(src_root, dest_root, skip=()):
    out = []
    for dp, _dn, fn in os.walk(src_root):
        if any(s in dp for s in skip):
            continue
        for f in fn:
            full = os.path.join(dp, f)
            rel = os.path.relpath(dp, src_root)
            dest = dest_root if rel == "." else os.path.join(dest_root, rel)
            out.append((full, dest))
    return out


# App icon (used by the Settings window).
if os.path.isfile(os.path.join(SPECPATH, "icon.ico")):
    datas += [(os.path.join(SPECPATH, "icon.ico"), ".")]

# The model (~1.5 GB) and CUDA libraries (~2 GB on disk) are NOT bundled by
# default — the installer downloads the ones the user picks, keeping the
# installer small. Set WHISPERTYPE_BUNDLE=1 to build a fully offline package.
BUNDLE = os.environ.get("WHISPERTYPE_BUNDLE") == "1"
BUNDLE_MODEL = os.environ.get("WHISPERTYPE_MODEL", "medium.en")

if BUNDLE:
    # Bundled Whisper model -> <bundle>/models/<name>
    model_src = os.path.join(SPECPATH, "models", BUNDLE_MODEL)
    if os.path.isdir(model_src):
        datas += collect_dir(
            model_src, os.path.join("models", BUNDLE_MODEL), skip=(".cache",)
        )

    # NVIDIA CUDA DLLs -> <bundle>/nvidia/<lib>/bin  (matches _nvidia_roots()).
    _nv = importlib.util.find_spec("nvidia")
    if _nv and _nv.submodule_search_locations:
        nv_root = _nv.submodule_search_locations[0]
        for lib in ("cublas", "cudnn", "cuda_nvrtc"):
            binp = os.path.join(nv_root, lib, "bin")
            if os.path.isdir(binp):
                datas += collect_dir(binp, os.path.join("nvidia", lib, "bin"))

a = Analysis(
    ["whispertype.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "torch"],  # tkinter IS needed (Settings window)
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperType",
    console=False,               # tray app, no console window
    icon="icon.ico",
    disable_windowed_traceback=False,
)

# Same code, console build. The installer runs this for `--fetch` so the user
# can see download progress; a windowed exe would show nothing.
exe_fetch = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperTypeFetch",
    console=True,
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    exe_fetch,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="WhisperType",
)
