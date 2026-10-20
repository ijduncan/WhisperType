#!/usr/bin/env bash
set -euo pipefail

# Verify every native binary in a frozen macOS app. PyInstaller can collect
# optional helpers as data without enforcing their architecture, so checking
# only Contents/MacOS/WhisperType is not sufficient.

APP_PATH="${1:-}"
MAXIMUM_MACOS_VERSION="${2:-14.0}"
EXPECTED_ARCH="${WHISPERTYPE_ARCH:-arm64}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "verify_bundle.sh must run on macOS." >&2
    exit 1
fi

if [[ -z "$APP_PATH" || ! -d "$APP_PATH" ]]; then
    echo "Usage: verify_bundle.sh /path/to/WhisperType.app [maximum-macOS-version]" >&2
    exit 1
fi

version_exceeds() {
    local actual="$1"
    local maximum="$2"
    awk -v actual="$actual" -v maximum="$maximum" 'BEGIN {
        split(actual, a, ".")
        split(maximum, m, ".")
        for (i = 1; i <= 3; i++) {
            a[i] += 0
            m[i] += 0
            if (a[i] > m[i]) exit 0
            if (a[i] < m[i]) exit 1
        }
        exit 1
    }'
}

PLIST_PATH="$APP_PATH/Contents/Info.plist"
if [[ ! -f "$PLIST_PATH" ]]; then
    echo "Missing app Info.plist: $PLIST_PATH" >&2
    exit 1
fi

PLIST_MINIMUM="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$PLIST_PATH")"
if version_exceeds "$PLIST_MINIMUM" "$MAXIMUM_MACOS_VERSION"; then
    echo "Info.plist requires macOS $PLIST_MINIMUM, newer than supported $MAXIMUM_MACOS_VERSION." >&2
    exit 1
fi

BUNDLE_BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$PLIST_PATH")"
if [[ ! "$BUNDLE_BUILD" =~ ^[1-9][0-9]*$ ]]; then
    echo "CFBundleVersion must be a positive integer; found '$BUNDLE_BUILD'." >&2
    exit 1
fi

mach_o_count=0
failure_count=0

while IFS= read -r -d '' candidate; do
    description="$(file -b "$candidate")"
    if [[ "$description" != *"Mach-O"* ]]; then
        continue
    fi

    mach_o_count=$((mach_o_count + 1))
    if ! architectures="$(lipo -archs "$candidate" 2>&1)"; then
        echo "Could not inspect architectures: $candidate" >&2
        echo "  $architectures" >&2
        failure_count=$((failure_count + 1))
        continue
    fi
    if [[ " $architectures " != *" $EXPECTED_ARCH "* ]]; then
        echo "Missing $EXPECTED_ARCH slice: $candidate ($architectures)" >&2
        failure_count=$((failure_count + 1))
    fi

    if ! load_commands="$(LC_ALL=C otool -l "$candidate" 2>&1)"; then
        echo "Could not inspect deployment target: $candidate" >&2
        echo "  $load_commands" >&2
        failure_count=$((failure_count + 1))
        continue
    fi

    deployment_versions="$(awk '
        $1 == "cmd" { command = $2; next }
        command == "LC_BUILD_VERSION" && $1 == "minos" {
            print $2
            command = ""
            next
        }
        command == "LC_VERSION_MIN_MACOSX" && $1 == "version" {
            print $2
            command = ""
            next
        }
    ' <<< "$load_commands" | sort -u)"

    if [[ -z "$deployment_versions" ]]; then
        echo "No macOS deployment target found: $candidate" >&2
        failure_count=$((failure_count + 1))
        continue
    fi

    while IFS= read -r deployment_version; do
        if [[ ! "$deployment_version" =~ ^[0-9]+([.][0-9]+){0,2}$ ]]; then
            echo "Unparseable deployment target '$deployment_version': $candidate" >&2
            failure_count=$((failure_count + 1))
        elif version_exceeds "$deployment_version" "$MAXIMUM_MACOS_VERSION"; then
            echo "Requires macOS $deployment_version (maximum $MAXIMUM_MACOS_VERSION): $candidate" >&2
            failure_count=$((failure_count + 1))
        fi
    done <<< "$deployment_versions"
done < <(find "$APP_PATH" -type f -print0)

if (( mach_o_count == 0 )); then
    echo "No Mach-O files found in $APP_PATH" >&2
    exit 1
fi

if (( failure_count != 0 )); then
    echo "Bundle verification failed with $failure_count native compatibility error(s)." >&2
    exit 1
fi

echo "Verified $mach_o_count Mach-O files: $EXPECTED_ARCH, macOS <= $MAXIMUM_MACOS_VERSION."
