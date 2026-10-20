#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]] && command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "setup_macos.sh requires an Apple silicon Mac running macOS 14 or later." >&2
    exit 1
fi

MACOS_VERSION="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VERSION%%.*}"
if [[ ! "$MACOS_MAJOR" =~ ^[0-9]+$ || "$MACOS_MAJOR" -lt 14 ]]; then
    echo "WhisperType requires macOS 14 or later; found $MACOS_VERSION." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"WhisperType requires Python 3.12; found {platform.python_version()}"
    )
if platform.machine() != "arm64":
    raise SystemExit(f"Python must be native arm64; found {platform.machine()}")
PY

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv .venv
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/python" -m pip install -r requirements.txt
"$ROOT_DIR/.venv/bin/python" -m pip install "pyinstaller>=6.22.0,<6.23"

echo
echo "Setup complete."
echo "Run from source:  .venv/bin/python whispertype.py"
echo "Build a DMG:      bash build_macos.sh"
