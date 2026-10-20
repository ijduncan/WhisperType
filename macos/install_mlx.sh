#!/usr/bin/env bash
set -euo pipefail

# Install the Apple-silicon GPU (MLX) runtime so the app still runs on macOS 14.
#
# MLX publishes separate wheels per macOS release (14, 15, 26) and pip picks
# the newest one the build machine supports, whose dylibs then refuse to load
# on older macOS. Pin the macOS 14 builds explicitly, then add mlx-whisper
# without its torch/numba/scipy dependencies (see requirements-mlx.txt).
#
# Usage: macos/install_mlx.sh /path/to/python
PYTHON_BIN="${1:?usage: install_mlx.sh /path/to/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_VERSION="$(sed -nE 's/^mlx==([0-9.]+).*/\1/p' "$ROOT_DIR/requirements.txt")"
WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$WHEEL_DIR"' EXIT

"$PYTHON_BIN" -m pip download \
    --only-binary=:all: --no-deps \
    --platform macosx_14_0_arm64 \
    --python-version "$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    --dest "$WHEEL_DIR" \
    "mlx==$MLX_VERSION" "mlx-metal==$MLX_VERSION"
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$WHEEL_DIR"/*.whl
"$PYTHON_BIN" -m pip install --no-deps -r "$ROOT_DIR/requirements-mlx.txt"
