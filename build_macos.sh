#!/usr/bin/env bash
set -euo pipefail

# Build a native Apple silicon, one-folder WhisperType.app and wrap it in a
# drag-to-install DMG. CTranslate2 and ONNX Runtime contain native code, so this
# build must run under an arm64 Python on an Apple silicon Mac.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -z "${PYTHON_BIN:-}" ]] && command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
ARCH="$(uname -m)"
RAW_VERSION="${WHISPERTYPE_VERSION:-dev}"
VERSION="${RAW_VERSION#v}"
BUILD_NUMBER="${WHISPERTYPE_BUILD_NUMBER:-1}"
ENTITLEMENTS_PATH="$ROOT_DIR/macos/entitlements.plist"
WORK_DIR="$ROOT_DIR/build/macos/$ARCH"
APP_DIST_DIR="$ROOT_DIR/dist/macos/$ARCH/app"
DMG_ROOT="$ROOT_DIR/dist/macos/$ARCH/dmg-root"
OUTPUT_DIR="$ROOT_DIR/dist/macos/$ARCH"
SUPPORTED_MACOS_VERSION="14.0"

if [[ "$ARCH" != "arm64" ]]; then
    echo "WhisperType's macOS build currently supports Apple silicon (arm64) only; found $ARCH." >&2
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_macos.sh must run on macOS." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"WhisperType requires Python 3.12; found {platform.python_version()}")
if platform.machine() != "arm64":
    raise SystemExit(f"Python must be native arm64; found {platform.machine()}")
PY

if [[ ! -f "$ENTITLEMENTS_PATH" ]]; then
    echo "Missing entitlements file: $ENTITLEMENTS_PATH" >&2
    exit 1
fi

if [[ ! "$BUILD_NUMBER" =~ ^[1-9][0-9]*$ ]]; then
    echo "WHISPERTYPE_BUILD_NUMBER must be a positive integer; found '$BUILD_NUMBER'." >&2
    exit 1
fi

if [[ "${MACOSX_DEPLOYMENT_TARGET:-$SUPPORTED_MACOS_VERSION}" != "$SUPPORTED_MACOS_VERSION" ]]; then
    echo "MACOSX_DEPLOYMENT_TARGET must be $SUPPORTED_MACOS_VERSION for supported release builds." >&2
    exit 1
fi

# Use a numeric bundle version even for branch/development builds.
if [[ "$VERSION" =~ ^[0-9]+([.][0-9]+){0,2}$ ]]; then
    BUNDLE_VERSION="$VERSION"
else
    BUNDLE_VERSION="0.0.0"
fi

safe_clean_dir() {
    local target="$1"
    case "$target" in
        "$ROOT_DIR"/build/*|"$ROOT_DIR"/dist/*)
            rm -rf -- "$target"
            ;;
        *)
            echo "Refusing to remove unexpected path: $target" >&2
            exit 1
            ;;
    esac
}

safe_clean_dir "$WORK_DIR"
safe_clean_dir "$APP_DIST_DIR"
safe_clean_dir "$DMG_ROOT"
mkdir -p "$WORK_DIR/spec" "$APP_DIST_DIR" "$DMG_ROOT" "$OUTPUT_DIR"

export MACOSX_DEPLOYMENT_TARGET="$SUPPORTED_MACOS_VERSION"
export PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR=1

echo "Building WhisperType.app for $ARCH (version $VERSION)..."
cd "$ROOT_DIR"
"$PYTHON_BIN" -m PyInstaller \
    --noconfirm \
    --clean \
    --onedir \
    --windowed \
    --noupx \
    --name WhisperType \
    --target-architecture "$ARCH" \
    --osx-bundle-identifier com.ijduncan.whispertype \
    --osx-entitlements-file "$ENTITLEMENTS_PATH" \
    --icon "$ROOT_DIR/icon.ico" \
    --add-data "$ROOT_DIR/icon.ico:." \
    --hidden-import pystray._darwin \
    --hidden-import pynput.keyboard._darwin \
    --hidden-import pynput.mouse._darwin \
    --collect-all ctranslate2 \
    --collect-all faster_whisper \
    --collect-all onnxruntime \
    --collect-all av \
    --collect-all _sounddevice_data \
    --exclude-module matplotlib \
    --exclude-module torch \
    --distpath "$APP_DIST_DIR" \
    --workpath "$WORK_DIR" \
    --specpath "$WORK_DIR/spec" \
    "$ROOT_DIR/whispertype.py"

APP_PATH="$APP_DIST_DIR/WhisperType.app"
PLIST_PATH="$APP_PATH/Contents/Info.plist"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/WhisperType"

if [[ ! -d "$APP_PATH" || ! -x "$EXECUTABLE_PATH" ]]; then
    echo "PyInstaller did not produce the expected app bundle: $APP_PATH" >&2
    exit 1
fi

plist_set_or_add() {
    local key="$1"
    local type="$2"
    local value="$3"
    if /usr/libexec/PlistBuddy -c "Print :$key" "$PLIST_PATH" >/dev/null 2>&1; then
        /usr/libexec/PlistBuddy -c "Set :$key $value" "$PLIST_PATH"
    else
        /usr/libexec/PlistBuddy -c "Add :$key $type $value" "$PLIST_PATH"
    fi
}

plist_set_or_add NSMicrophoneUsageDescription string \
    "WhisperType records audio only while you dictate, then transcribes it locally."
plist_set_or_add LSUIElement bool true
plist_set_or_add LSMultipleInstancesProhibited bool true
plist_set_or_add NSPrincipalClass string NSApplication
plist_set_or_add CFBundleDisplayName string WhisperType
plist_set_or_add LSApplicationCategoryType string public.app-category.utilities
plist_set_or_add LSMinimumSystemVersion string "$MACOSX_DEPLOYMENT_TARGET"
plist_set_or_add CFBundleShortVersionString string "$BUNDLE_VERSION"
plist_set_or_add CFBundleVersion string "$BUILD_NUMBER"
plutil -lint "$PLIST_PATH"

# Editing Info.plist invalidates PyInstaller's initial signature, so sign again.
# Ad-hoc ("-") is the default and suits CI smoke builds, but macOS keys an
# ad-hoc app's Accessibility/Input Monitoring grants to its exact cdhash, so
# every rebuild loses them. For local testing, a self-signed identity in a
# dedicated keychain keeps a stable designated requirement across rebuilds:
#   WHISPERTYPE_SIGN_KEYCHAIN (default ~/Library/Keychains/whispertype-signing.keychain-db)
#   WHISPERTYPE_SIGN_KEYCHAIN_PASSWORD_FILE (default ~/.config/whispertype/signing-keychain-password)
# WHISPERTYPE_SIGN_IDENTITY overrides both. Public releases should instead use
# a Developer ID Application identity and Apple notarization.
SIGN_IDENTITY="${WHISPERTYPE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${WHISPERTYPE_SIGN_KEYCHAIN:-$HOME/Library/Keychains/whispertype-signing.keychain-db}"
SIGN_KEYCHAIN_PASSWORD_FILE="${WHISPERTYPE_SIGN_KEYCHAIN_PASSWORD_FILE:-$HOME/.config/whispertype/signing-keychain-password}"
if [[ -z "$SIGN_IDENTITY" && -f "$SIGN_KEYCHAIN" && -f "$SIGN_KEYCHAIN_PASSWORD_FILE" ]]; then
    security unlock-keychain -p "$(cat "$SIGN_KEYCHAIN_PASSWORD_FILE")" "$SIGN_KEYCHAIN"
    # Self-signed identities are reported as untrusted, so match by name
    # without -v and sign by SHA-1 hash.
    SIGN_IDENTITY="$(security find-identity -p codesigning "$SIGN_KEYCHAIN" \
        | awk '/"WhisperType Local Signing"/ { print $2; exit }')"
fi
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
echo "Signing identity: $SIGN_IDENTITY"
codesign \
    --force \
    --deep \
    --all-architectures \
    --sign "$SIGN_IDENTITY" \
    --timestamp=none \
    --options=runtime \
    --entitlements "$ENTITLEMENTS_PATH" \
    "$APP_PATH"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
bash "$ROOT_DIR/macos/verify_bundle.sh" "$APP_PATH" "$SUPPORTED_MACOS_VERSION"

ditto "$APP_PATH" "$DMG_ROOT/WhisperType.app"
ln -s /Applications "$DMG_ROOT/Applications"

DMG_PATH="$OUTPUT_DIR/WhisperType-macOS-arm64.dmg"
rm -f -- "$DMG_PATH"
hdiutil create \
    -volname WhisperType \
    -srcfolder "$DMG_ROOT" \
    -format UDZO \
    -ov \
    "$DMG_PATH"

echo "Built: $DMG_PATH"
shasum -a 256 "$DMG_PATH"
