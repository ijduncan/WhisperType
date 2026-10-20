"""Small platform adapter for paths, hotkeys, permissions, and paste events.

The transcription core is intentionally platform-neutral.  Windows keeps the
well-tested ``keyboard`` backend, while macOS uses ``pynput`` and the native
Accessibility permission prompt.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

_MODIFIERS = {"ctrl", "option", "alt", "shift", "cmd", "windows"}
_SPECIAL_KEYS = {
    "space", "tab", "enter", "esc", "backspace", "delete", "insert",
    "home", "end", "page_up", "page_down", "up", "down", "left", "right",
}


def default_hotkey(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name == "darwin":
        # Command+Space belongs to Spotlight.  Adding Shift avoids that system
        # shortcut while remaining comfortable for repeated dictation.
        return "cmd+shift+space"
    return "ctrl+alt+windows+space"


def default_model(platform_name: str | None = None) -> str:
    """Use a smaller first download on CPU-only Macs."""
    return "small.en" if (platform_name or sys.platform) == "darwin" else "medium.en"


def default_acceleration(platform_name: str | None = None) -> tuple[str, str]:
    if (platform_name or sys.platform) == "darwin":
        return "cpu", "int8"
    return "cuda", "float16"


def user_data_dir(
    app_name: str = "WhisperType",
    *,
    platform_name: str | None = None,
    env: dict[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> str:
    """Return the writable per-user data directory for the current platform."""
    platform_name = platform_name or sys.platform
    env = os.environ if env is None else env

    override = env.get("WHISPERTYPE_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))

    if platform_name == "darwin":
        base = PurePosixPath(home) if home is not None else Path.home()
        return str(base / "Library" / "Application Support" / app_name)
    if platform_name == "win32":
        base = env.get("LOCALAPPDATA") or env.get("APPDATA")
        if base:
            return os.path.join(base, app_name)

    base = env.get("XDG_DATA_HOME")
    if not base:
        root = Path(home) if home is not None else Path.home()
        base = str(root / ".local" / "share")
    return os.path.join(base, app_name)


def normalize_hotkey(value: str, platform_name: str | None = None) -> str:
    """Normalize a user-entered shortcut to WhisperType's portable syntax."""
    platform_name = platform_name or sys.platform
    raw_parts = [p.strip().lower() for p in re.split(r"\s*\+\s*", value.strip())]
    if not raw_parts or any(not p for p in raw_parts):
        raise ValueError("hotkey cannot be empty")

    common = {
        "control": "ctrl",
        "ctl": "ctrl",
        "alternate": "alt",
        "opt": "option",
        "command": "cmd",
        "win": "windows",
        "super": "windows",
        "meta": "windows",
        "return": "enter",
        "escape": "esc",
        "pgup": "page_up",
        "pgdn": "page_down",
    }
    parts = [common.get(p, p) for p in raw_parts]

    if platform_name == "darwin":
        aliases = {"alt": "option", "windows": "cmd"}
        modifier_order = ("ctrl", "option", "cmd", "shift")
    else:
        aliases = {"option": "alt", "cmd": "windows"}
        modifier_order = ("ctrl", "alt", "shift", "windows")
    parts = [aliases.get(p, p) for p in parts]

    if len(set(parts)) != len(parts):
        raise ValueError("hotkey contains a duplicate key")

    keys = [p for p in parts if p not in _MODIFIERS]
    modifiers = [p for p in modifier_order if p in parts]
    if len(keys) != 1:
        raise ValueError("hotkey must contain exactly one non-modifier key")
    key = keys[0]
    valid_key = (
        len(key) == 1
        or key in _SPECIAL_KEYS
        or bool(re.fullmatch(r"f(?:[1-9]|1[0-9]|2[0-4])", key))
    )
    if not valid_key:
        raise ValueError(f"unsupported hotkey key: {key}")
    return "+".join([*modifiers, key])


def to_pynput_hotkey(value: str) -> str:
    """Translate portable macOS shortcut syntax to pynput's parser syntax."""
    parts = normalize_hotkey(value, "darwin").split("+")
    converted = []
    for part in parts:
        if part == "option":
            converted.append("<alt>")
        elif part in {"ctrl", "shift", "cmd"} or part in _SPECIAL_KEYS or part.startswith("f"):
            converted.append(f"<{part}>")
        else:
            converted.append(part)
    return "+".join(converted)


def _mac_key_name(key) -> str | None:
    """Convert a pynput key object to portable shortcut syntax."""
    name = getattr(key, "name", None)
    if name:
        aliases = {
            "ctrl_l": "ctrl", "ctrl_r": "ctrl", "ctrl": "ctrl",
            "alt_l": "option", "alt_r": "option", "alt": "option",
            "shift_l": "shift", "shift_r": "shift", "shift": "shift",
            "cmd_l": "cmd", "cmd_r": "cmd", "cmd": "cmd",
            "return": "enter", "escape": "esc",
        }
        return aliases.get(name, name)
    char = getattr(key, "char", None)
    return char.lower() if isinstance(char, str) and len(char) == 1 else None


_pynput_layout_pinned = False


def pin_pynput_keyboard_layout() -> None:
    """Read the keyboard layout once on the main thread and reuse it in pynput.

    pynput's listeners call ``keycode_context()`` (TISCopyCurrentKeyboardInputSource)
    on their own thread; macOS 14+ asserts that Text Input Sources are only
    touched on the main thread and kills the process. Call this from the main
    thread before any listener starts. Layout switches made while the app runs
    are not picked up until it restarts.
    """
    global _pynput_layout_pinned
    if not IS_MACOS or _pynput_layout_pinned:
        return
    if threading.current_thread() is not threading.main_thread():
        print("[keyboard] layout must be pinned on the main thread; skipped")
        return
    import contextlib

    import pynput._util.darwin as util_darwin
    import pynput.keyboard._darwin as keyboard_darwin

    with util_darwin.keycode_context() as context:
        cached = context

    @contextlib.contextmanager
    def cached_keycode_context():
        yield cached

    util_darwin.keycode_context = cached_keycode_context
    keyboard_darwin.keycode_context = cached_keycode_context
    _pynput_layout_pinned = True


class KeyboardBackend:
    """Register/capture a global shortcut and synthesize the platform paste."""

    def __init__(self, platform_name: str | None = None):
        self.platform_name = platform_name or sys.platform
        if self.platform_name == "darwin":
            pin_pynput_keyboard_layout()

    def validate(self, hotkey: str) -> str:
        if self.platform_name == "win32":
            # The Windows keyboard package supports multi-step sequences and
            # a wider set of named/media keys. Preserve that established
            # syntax instead of narrowing existing user configurations to the
            # portable macOS subset.
            normalized = hotkey.strip().lower()
            if not normalized:
                raise ValueError("hotkey cannot be empty")
            import keyboard

            keyboard.parse_hotkey(normalized)
            return normalized

        normalized = normalize_hotkey(hotkey, self.platform_name)
        if self.platform_name == "darwin":
            from pynput.keyboard import HotKey

            HotKey.parse(to_pynput_hotkey(normalized))
        else:
            raise RuntimeError(f"unsupported platform: {self.platform_name}")
        return normalized

    def register(self, hotkey: str, callback):
        normalized = self.validate(hotkey)
        if self.platform_name == "darwin":
            from pynput.keyboard import GlobalHotKeys

            listener = GlobalHotKeys({to_pynput_hotkey(normalized): callback})
            listener.start()
            listener.wait()
            return listener
        import keyboard

        return keyboard.add_hotkey(normalized, callback)

    def unregister(self, handle) -> None:
        if handle is None:
            return
        try:
            if self.platform_name == "darwin":
                handle.stop()
            else:
                import keyboard

                keyboard.remove_hotkey(handle)
        except Exception:
            pass

    def capture(self) -> str | None:
        if self.platform_name == "win32":
            import keyboard

            return self.validate(keyboard.read_hotkey(suppress=False))
        if self.platform_name != "darwin":
            raise RuntimeError(f"unsupported platform: {self.platform_name}")

        if not self.request_input_monitoring(prompt=True):
            print("[hotkey] Input Monitoring permission required to record a shortcut")
            return None

        from pynput import keyboard
        import pynput._util.darwin as util_darwin
        from Quartz import (
            CGEventGetFlags,
            kCGEventFlagMaskAlternate,
            kCGEventFlagMaskCommand,
            kCGEventFlagMaskControl,
            kCGEventFlagMaskShift,
            kCGEventKeyDown,
        )

        modifier_masks = (
            ("ctrl", kCGEventFlagMaskControl),
            ("option", kCGEventFlagMaskAlternate),
            ("cmd", kCGEventFlagMaskCommand),
            ("shift", kCGEventFlagMaskShift),
        )
        result: list[str] = []
        done = threading.Event()
        key_down_flags = [0]

        class FlagRecordingListener(keyboard.Listener):
            # Record the key-down event's own modifier flags before pynput
            # dispatches it to on_press (darwin_intercept runs only after).
            def _handle_message(self, proxy, event_type, event, refcon, injected):
                if event_type == kCGEventKeyDown:
                    key_down_flags[0] = CGEventGetFlags(event)
                super()._handle_message(proxy, event_type, event, refcon, injected)

        def key_token(key) -> str | None:
            if getattr(key, "name", None):
                return _mac_key_name(key)
            # Modifier-only (flagsChanged) events arrive as KeyCodes with no
            # char; skip them. For real keys use the unmodified character for
            # the virtual key, since Option/Shift change ``char`` (Option+K
            # reports "˚").
            if getattr(key, "char", None) is None:
                return None
            vk = getattr(key, "vk", None)
            if vk is not None:
                with util_darwin.keycode_context() as context:
                    base = util_darwin.keycode_to_string(context, vk)
                if base and len(base) == 1 and base.isprintable():
                    return base.lower()
            return _mac_key_name(key)

        def on_press(key, *_injected):
            token = key_token(key)
            if not token or token in {"ctrl", "option", "shift", "cmd"}:
                return
            # Take the modifiers from the key-down event itself rather than
            # tracking their press/release events, which pynput can miss or
            # report as unnamed KeyCodes.
            flags = key_down_flags[0]
            held = [name for name, mask in modifier_masks if flags & mask]
            try:
                result.append(normalize_hotkey("+".join([*held, token]), "darwin"))
            except ValueError as exc:
                print(f"[hotkey] could not use {'+'.join([*held, token])!r}: {exc}")
            done.set()
            return False

        listener = FlagRecordingListener(on_press=on_press)
        listener.start()
        done.wait(timeout=30)
        listener.stop()
        return result[0] if result else None

    def request_accessibility(self, prompt: bool = True) -> bool:
        if self.platform_name != "darwin":
            return True
        try:
            from ApplicationServices import (
                AXIsProcessTrustedWithOptions,
                kAXTrustedCheckOptionPrompt,
            )

            options = {kAXTrustedCheckOptionPrompt: prompt}
            return bool(AXIsProcessTrustedWithOptions(options))
        except Exception as exc:
            print(f"[permissions] Could not check Accessibility access: {exc}")
            return False

    def request_input_monitoring(self, prompt: bool = True) -> bool:
        """Check (and optionally request) Input Monitoring.

        pynput's listeners use a listen-only event tap, which macOS feeds
        only to apps with Input Monitoring; Accessibility alone is not
        enough. Requesting it also adds WhisperType to that settings list.
        """
        if self.platform_name != "darwin":
            return True
        try:
            from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess

            if CGPreflightListenEventAccess():
                return True
            if prompt:
                CGRequestListenEventAccess()
            return False
        except Exception as exc:
            print(f"[permissions] Could not check Input Monitoring access: {exc}")
            return False

    def request_hotkey_access(self, prompt: bool = True) -> bool:
        """Accessibility and Input Monitoring, both needed to hear hotkeys."""
        accessibility = self.request_accessibility(prompt=prompt)
        input_monitoring = self.request_input_monitoring(prompt=prompt)
        return accessibility and input_monitoring

    def paste(self) -> bool:
        """Send the platform paste shortcut; return False if permission is absent."""
        if self.platform_name == "darwin":
            if not self.request_accessibility(prompt=True):
                return False
            # Post Cmd+V with CGEvents rather than pynput's Controller: pynput
            # looks up the keyboard layout through Text Input Sources, which
            # macOS 14+ only allows on the main thread and aborts the process
            # otherwise. CGEventPost is safe from the transcription worker.
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventPost,
                CGEventSetFlags,
                kCGEventFlagMaskCommand,
                kCGHIDEventTap,
            )

            v_keycode = 9  # kVK_ANSI_V
            for key_down in (True, False):
                event = CGEventCreateKeyboardEvent(None, v_keycode, key_down)
                CGEventSetFlags(event, kCGEventFlagMaskCommand)
                CGEventPost(kCGHIDEventTap, event)
            return True
        import keyboard

        keyboard.send("ctrl+v")
        return True


def open_accessibility_settings(pane: str = "accessibility") -> None:
    if not IS_MACOS:
        return
    subprocess.Popen(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?"
            + ("Privacy_Accessibility" if pane == "accessibility" else "Privacy_ListenEvent"),
        ],
        close_fds=True,
    )
