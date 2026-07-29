"""Minimal multicast start-to-goal straight sprint runner.

Kept:
  - ActorPrepare / ActorPrepareResult
  - Notify start / finish / abort
  - INS pose reception
  - VehicleControl transmission and VehicleFeedback draining

Removed:
  - reinforcement learning
  - camera, lidar and NPC perception
  - Predictor
  - ROS and global planning
  - map loading and obstacle handling

Control policy:
  - speed-controlled alignment towards the goal (2 s is warning only)
  - zero steering and wait for heading/yaw/steering feedback to settle
  - re-anchor the goal line and align again if the new bearing has drifted
  - repeat one logical acceleration pulse until the chassis samples it
  - hold a constant speed target with acceleration/steering locked to zero
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import re
import select
import struct
import sys
import threading
import time
from pathlib import Path

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


# The onsite tree may contain more than one generated ``chassis`` package.
# Mixing a protobuf package from one SDK build with libMulticastNetwork from
# another is especially dangerous: scalar fields can still appear to work
# while nested fields (notably steering_control) are decoded incorrectly.
# Select the SDK root with the same rule used by the known-good run.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ONSITE_ROOT = _SCRIPT_DIR.parent


def _is_complete_sdk_root(path):
    return (
        path.is_dir()
        and (path / "modules").is_dir()
        and (
            path / "chassis" / "proto" / "chassis_enums_pb2.py"
        ).is_file()
        and (
            path / "chassis" / "proto" / "chassis_messages_pb2.py"
        ).is_file()
        and (path / "main" / "proto" / "enums_pb2.py").is_file()
        and (path / "main" / "proto" / "messages_pb2.py").is_file()
    )


def _find_onsite_sdk_root():
    """Find one coherent SDK root, matching run.py before any pb2 import."""
    current = _SCRIPT_DIR
    while True:
        for candidate in (current, current / "e2e"):
            if _is_complete_sdk_root(candidate):
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Compatibility fallback for stripped deployments without ``modules``.
    # Never silently choose between multiple protobuf copies.
    candidates = set()
    for base in (_ONSITE_ROOT, Path.cwd().resolve()):
        if not base.is_dir():
            continue
        for pattern in (
            "chassis/proto/chassis_enums_pb2.py",
            "*/chassis/proto/chassis_enums_pb2.py",
            "*/*/chassis/proto/chassis_enums_pb2.py",
        ):
            for enum_file in base.glob(pattern):
                candidate = enum_file.parents[2]
                if (
                    candidate
                    / "chassis"
                    / "proto"
                    / "chassis_messages_pb2.py"
                ).is_file() and (
                    candidate / "main" / "proto" / "messages_pb2.py"
                ).is_file():
                    candidates.add(candidate.resolve())
    if len(candidates) == 1:
        return next(iter(candidates))
    if candidates:
        choices = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            "multiple onsite protobuf roots found; refusing to mix SDK "
            f"versions: {choices}"
        )
    raise ModuleNotFoundError(
        "cannot locate a coherent onsite SDK root containing "
        "chassis/proto and main/proto"
    )


_ONSITE_PROTO_ROOT = _find_onsite_sdk_root()
for _module_root in (_ONSITE_ROOT, _SCRIPT_DIR, _ONSITE_PROTO_ROOT):
    _module_root_text = str(_module_root)
    if _module_root_text in sys.path:
        sys.path.remove(_module_root_text)
    sys.path.insert(0, _module_root_text)

import libMulticastNetwork

from chassis.proto.chassis_enums_pb2 import (
    VEHICLE_CONTROL,
    VEHICLE_FEEDBACK,
)
from chassis.proto.chassis_messages_pb2 import (
    VehicleControl,
    VehicleFeedback,
)
from get_ip import get_ip_address
from main.proto.enums_pb2 import (
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
    MT_NOTIFY,
    NT_ABORT_TEST,
    NT_COLLIDE_ROLE,
    NT_DESTROY_ROLE,
    NT_FINISH_TEST,
    NT_START_TEST,
)
from main.proto.messages_pb2 import (
    ActorPrepare,
    ActorPrepareResult,
    Notify,
)


class _TeeTextIO:
    """Mirror single-process runtime text to the terminal and one archive."""

    def __init__(self, terminal, archive, lock):
        self.terminal = terminal
        self.archive = archive
        self.lock = lock
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.archive.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.archive.flush()

    def isatty(self):
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self):
        return self.terminal.fileno()


def _install_run_logging(args):
    """Archive the single communication/control process."""
    log_dir = Path(args.run_log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = log_dir / (
        f"run_hmxzw_{stamp}_{os.getpid()}.log"
    )
    archive = archive_path.open(
        "a", encoding="utf-8", buffering=1
    )
    lock = threading.RLock()
    sys.stdout = _TeeTextIO(sys.stdout, archive, lock)
    sys.stderr = _TeeTextIO(sys.stderr, archive, lock)
    latest_pointer = log_dir / "latest.txt"
    latest_pointer.write_text(
        str(archive_path) + "\n", encoding="utf-8"
    )
    print(
        "[run3][log] "
        f"archive={archive_path} latest={latest_pointer}",
        flush=True,
    )
    return archive


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def session_order_key(value):
    match = re.search(r"_(\d{14})_(\d+)$", str(value or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def is_newer_session(candidate, current):
    candidate_key = session_order_key(candidate)
    current_key = session_order_key(current)
    return (
        candidate_key is not None
        and current_key is not None
        and candidate_key > current_key
    )


class LinuxKeyboardStateReader:
    """Track held keys through X11 (remote) or Linux evdev (local)."""

    EV_KEY = 0x01
    KEY_W = 17
    KEY_S = 31
    KEY_A = 30
    KEY_D = 32
    KEY_SPACE = 57
    KEY_UP = 103
    KEY_LEFT = 105
    KEY_RIGHT = 106
    KEY_DOWN = 108
    _EVENT = struct.Struct("@llHHi")
    _X11_KEYSYMS = {
        KEY_W: 0x0077,
        KEY_S: 0x0073,
        KEY_A: 0x0061,
        KEY_D: 0x0064,
        KEY_SPACE: 0x0020,
        KEY_UP: 0xFF52,
        KEY_LEFT: 0xFF51,
        KEY_RIGHT: 0xFF53,
        KEY_DOWN: 0xFF54,
    }

    def __init__(self):
        self.backend = None
        self.fd = None
        self.device_path = None
        self.x11 = None
        self.x11_display = None
        self.x11_keycodes = {}
        self.pressed = set()
        self.launch_requested = False
        self.buffer = b""
        self.stdin_fd = None
        self.saved_terminal_attributes = None

    @staticmethod
    def _proc_keyboard_devices():
        proc_path = Path("/proc/bus/input/devices")
        try:
            text = proc_path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return []

        devices = []
        for block in text.split("\n\n"):
            handler_line = next(
                (
                    line
                    for line in block.splitlines()
                    if line.startswith("H: Handlers=")
                ),
                "",
            )
            handlers = handler_line.partition("=")[2].split()
            if "kbd" not in handlers:
                continue
            for handler in handlers:
                if re.fullmatch(r"event\d+", handler):
                    devices.append(Path("/dev/input") / handler)
        return devices

    @classmethod
    def _candidate_devices(cls, configured_device):
        if configured_device:
            return [Path(configured_device).expanduser()]

        candidates = []
        for pattern in (
            "/dev/input/by-id/*-event-kbd",
            "/dev/input/by-path/*-event-kbd",
        ):
            candidates.extend(sorted(Path("/").glob(pattern[1:])))
        candidates.extend(cls._proc_keyboard_devices())

        unique = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            try:
                key = str(candidate.resolve())
            except OSError:
                pass
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _open_x11(self):
        display_name = str(os.environ.get("DISPLAY") or "").strip()
        if not display_name:
            raise RuntimeError("DISPLAY is not set")

        library_name = ctypes.util.find_library("X11")
        if not library_name:
            raise RuntimeError("libX11 was not found")
        try:
            x11 = ctypes.CDLL(library_name)
        except OSError as exc:
            raise RuntimeError(f"cannot load {library_name}: {exc}") from exc

        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        x11.XKeysymToKeycode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        x11.XKeysymToKeycode.restype = ctypes.c_ubyte
        x11.XQueryKeymap.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
        ]
        x11.XQueryKeymap.restype = ctypes.c_int

        display = x11.XOpenDisplay(display_name.encode("utf-8"))
        if not display:
            raise RuntimeError(
                f"cannot open X11 display {display_name!r}"
            )

        try:
            keycodes = {
                control_code: int(
                    x11.XKeysymToKeycode(display, keysym)
                )
                for control_code, keysym in self._X11_KEYSYMS.items()
            }
            missing = [
                code for code, keycode in keycodes.items()
                if keycode <= 0
            ]
            if missing:
                raise RuntimeError(
                    f"X11 keycode lookup failed for controls={missing}"
                )
        except Exception:
            x11.XCloseDisplay(display)
            raise

        self.x11 = x11
        self.x11_display = display
        self.x11_keycodes = keycodes
        self.backend = "x11"
        self.device_path = f"X11(display={display_name})"

    def _open_evdev(self, configured_device=""):
        if sys.platform != "linux":
            raise RuntimeError(
                "manual simultaneous-key control requires Linux evdev"
            )

        candidates = self._candidate_devices(configured_device)
        if not candidates:
            raise RuntimeError(
                "no Linux keyboard event device found; pass "
                "--keyboard-device /dev/input/eventN"
            )

        errors = []
        for candidate in candidates:
            try:
                fd = os.open(
                    str(candidate),
                    os.O_RDONLY | os.O_NONBLOCK,
                )
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            self.fd = fd
            self.device_path = str(candidate)
            break

        if self.fd is None:
            detail = "; ".join(errors)
            raise RuntimeError(
                "cannot open a Linux keyboard event device. Add this "
                "user to the input group (then sign out/in), run with "
                "suitable device permission, or pass "
                "--keyboard-device /dev/input/eventN. "
                f"Attempts: {detail}"
            )
        self.backend = "evdev"

    def _suppress_terminal_input(self):
        # X11/evdev supplies the actual key state. cbreak is used only to
        # keep W/A/S/D and escape sequences out of the shell's pending input.
        if (
            termios is None
            or tty is None
            or not sys.stdin.isatty()
        ):
            return
        try:
            self.stdin_fd = sys.stdin.fileno()
            self.saved_terminal_attributes = termios.tcgetattr(
                self.stdin_fd
            )
            tty.setcbreak(self.stdin_fd)
        except (OSError, termios.error):
            self.stdin_fd = None
            self.saved_terminal_attributes = None

    def open(self, configured_device=""):
        if sys.platform != "linux":
            raise RuntimeError(
                "manual simultaneous-key control requires Linux"
            )

        errors = []
        if not configured_device:
            try:
                self._open_x11()
            except RuntimeError as exc:
                errors.append(f"X11: {exc}")

        if self.backend is None:
            try:
                self._open_evdev(configured_device)
            except RuntimeError as exc:
                errors.append(f"evdev: {exc}")
                detail = " | ".join(errors)
                raise RuntimeError(
                    "cannot open a held-key input backend. Remote desktop "
                    "control needs access to its X11 DISPLAY; local evdev "
                    "control needs keyboard-device read permission. "
                    f"Details: {detail}"
                ) from exc

        self._suppress_terminal_input()
        self.clear_state()
        return self.device_path

    def clear_state(self):
        self.pressed.clear()
        self.launch_requested = False
        self.buffer = b""

    def is_pressed(self, *key_codes):
        return any(code in self.pressed for code in key_codes)

    def _process_bytes(self, chunk):
        self.buffer += chunk
        complete_size = (
            len(self.buffer) // self._EVENT.size
        ) * self._EVENT.size
        complete = self.buffer[:complete_size]
        self.buffer = self.buffer[complete_size:]
        for offset in range(0, len(complete), self._EVENT.size):
            _, _, event_type, key_code, value = (
                self._EVENT.unpack_from(complete, offset)
            )
            if event_type != self.EV_KEY:
                continue
            if value in (1, 2):
                self.pressed.add(key_code)
            elif value == 0:
                self.pressed.discard(key_code)
            if key_code == self.KEY_SPACE and value == 1:
                self.launch_requested = True

    def _drain_terminal_input(self):
        if self.stdin_fd is not None:
            try:
                readable, _, _ = select.select(
                    [self.stdin_fd], [], [], 0.0
                )
                if readable:
                    os.read(self.stdin_fd, 256)
            except OSError:
                pass

    def _poll_x11(self):
        keymap = (ctypes.c_char * 32)()
        self.x11.XQueryKeymap(self.x11_display, keymap)
        keymap_bytes = bytes(keymap)
        new_pressed = {
            control_code
            for control_code, keycode in self.x11_keycodes.items()
            if (
                keymap_bytes[keycode // 8]
                & (1 << (keycode % 8))
            )
        }
        if (
            self.KEY_SPACE in new_pressed
            and self.KEY_SPACE not in self.pressed
        ):
            self.launch_requested = True
        self.pressed = new_pressed

    def _poll_evdev(self):
        while True:
            readable, _, _ = select.select([self.fd], [], [], 0.0)
            if not readable:
                break
            try:
                chunk = os.read(self.fd, self._EVENT.size * 64)
            except BlockingIOError:
                break
            if not chunk:
                break
            self._process_bytes(chunk)

    def poll(self):
        if self.backend is None:
            return
        self._drain_terminal_input()
        if self.backend == "x11":
            self._poll_x11()
        elif self.backend == "evdev":
            self._poll_evdev()

    def consume_launch_request(self):
        requested = self.launch_requested
        self.launch_requested = False
        return requested

    def close(self):
        if (
            self.stdin_fd is not None
            and self.saved_terminal_attributes is not None
        ):
            try:
                termios.tcsetattr(
                    self.stdin_fd,
                    termios.TCSADRAIN,
                    self.saved_terminal_attributes,
                )
            except (OSError, termios.error):
                pass
        self.stdin_fd = None
        self.saved_terminal_attributes = None
        if self.x11_display is not None:
            self.x11.XCloseDisplay(self.x11_display)
        self.x11 = None
        self.x11_display = None
        self.x11_keycodes = {}
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        self.backend = None
        self.device_path = None
        self.clear_state()


class _X11Color(ctypes.Structure):
    _fields_ = [
        ("pixel", ctypes.c_ulong),
        ("red", ctypes.c_ushort),
        ("green", ctypes.c_ushort),
        ("blue", ctypes.c_ushort),
        ("flags", ctypes.c_byte),
        ("pad", ctypes.c_byte),
    ]


class _X11ClientMessageData(ctypes.Union):
    _fields_ = [
        ("b", ctypes.c_char * 20),
        ("s", ctypes.c_short * 10),
        ("l", ctypes.c_long * 5),
    ]


class _X11ClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", _X11ClientMessageData),
    ]


class X11AlignmentWindow:
    """Small dependency-free top-down manual-alignment display."""

    EXPOSURE_MASK = 1 << 15
    STRUCTURE_NOTIFY_MASK = 1 << 17
    CLIENT_MESSAGE = 33
    DESTROY_NOTIFY = 17
    XColor = _X11Color
    XClientMessageEvent = _X11ClientMessageEvent

    def __init__(self, enabled=True, width=760, height=560):
        self.enabled = bool(enabled)
        self.width = max(520, int(width))
        self.height = max(400, int(height))
        self.display = None
        self.window = 0
        self.gc = None
        self.x11 = None
        self.wm_delete_window = 0
        self.last_draw_time = 0.0
        self.colors = {}

    def _configure_x11(self, x11):
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x11.XDefaultScreen.restype = ctypes.c_int
        x11.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XRootWindow.restype = ctypes.c_ulong
        x11.XBlackPixel.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XBlackPixel.restype = ctypes.c_ulong
        x11.XWhitePixel.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XWhitePixel.restype = ctypes.c_ulong
        x11.XDefaultColormap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        x11.XDefaultColormap.restype = ctypes.c_ulong
        x11.XCreateSimpleWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        x11.XCreateSimpleWindow.restype = ctypes.c_ulong
        x11.XStoreName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_char_p,
        ]
        x11.XSelectInput.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_long,
        ]
        x11.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XCreateGC.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        x11.XCreateGC.restype = ctypes.c_void_p
        x11.XFreeGC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        x11.XSetForeground.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        x11.XSetLineAttributes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        x11.XDrawLine.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        x11.XDrawArc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
        ]
        x11.XFillArc.argtypes = list(x11.XDrawArc.argtypes)
        x11.XDrawString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        x11.XClearWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        x11.XPending.argtypes = [ctypes.c_void_p]
        x11.XPending.restype = ctypes.c_int
        x11.XNextEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        x11.XInternAtom.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XSetWMProtocols.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_int,
        ]
        x11.XAllocNamedColor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_char_p,
            ctypes.POINTER(self.XColor),
            ctypes.POINTER(self.XColor),
        ]
        x11.XAllocNamedColor.restype = ctypes.c_int
        x11.XDestroyWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        x11.XFlush.argtypes = [ctypes.c_void_p]

    def _allocate_color(self, name, fallback):
        screen_color = self.XColor()
        exact_color = self.XColor()
        if self.x11.XAllocNamedColor(
            self.display,
            self.colormap,
            name.encode("ascii"),
            ctypes.byref(screen_color),
            ctypes.byref(exact_color),
        ):
            return int(screen_color.pixel)
        return int(fallback)

    def open(self):
        if not self.enabled:
            return False
        display_name = str(os.environ.get("DISPLAY") or "").strip()
        if not display_name:
            print(
                "[run3][alignment-window][WARN] DISPLAY is not set; "
                "visual aid disabled",
                flush=True,
            )
            self.enabled = False
            return False

        library_name = ctypes.util.find_library("X11")
        if not library_name:
            print(
                "[run3][alignment-window][WARN] libX11 not found; "
                "visual aid disabled",
                flush=True,
            )
            self.enabled = False
            return False
        try:
            self.x11 = ctypes.CDLL(library_name)
            self._configure_x11(self.x11)
            self.display = self.x11.XOpenDisplay(
                display_name.encode("utf-8")
            )
            if not self.display:
                raise RuntimeError(
                    f"cannot open X11 display {display_name!r}"
                )
            screen = self.x11.XDefaultScreen(self.display)
            root = self.x11.XRootWindow(self.display, screen)
            black = self.x11.XBlackPixel(self.display, screen)
            white = self.x11.XWhitePixel(self.display, screen)
            self.colormap = self.x11.XDefaultColormap(
                self.display, screen
            )
            self.window = self.x11.XCreateSimpleWindow(
                self.display,
                root,
                80,
                80,
                self.width,
                self.height,
                1,
                black,
                white,
            )
            if not self.window:
                raise RuntimeError("XCreateSimpleWindow failed")
            self.x11.XStoreName(
                self.display,
                self.window,
                b"Run3 Manual Alignment",
            )
            self.x11.XSelectInput(
                self.display,
                self.window,
                self.EXPOSURE_MASK | self.STRUCTURE_NOTIFY_MASK,
            )
            self.wm_delete_window = self.x11.XInternAtom(
                self.display, b"WM_DELETE_WINDOW", 0
            )
            wm_protocols = (ctypes.c_ulong * 1)(
                self.wm_delete_window
            )
            self.x11.XSetWMProtocols(
                self.display, self.window, wm_protocols, 1
            )
            self.gc = self.x11.XCreateGC(
                self.display, self.window, 0, None
            )
            if not self.gc:
                raise RuntimeError("XCreateGC failed")
            self.colors = {
                "text": self._allocate_color("#111827", black),
                "muted": self._allocate_color("#6B7280", black),
                "line": self._allocate_color("#2563EB", black),
                "ego": self._allocate_color("#16A34A", black),
                "heading": self._allocate_color("#DC2626", black),
                "goal": self._allocate_color("#7C3AED", black),
            }
            self.x11.XMapRaised(self.display, self.window)
            self.x11.XFlush(self.display)
            print(
                "[run3][alignment-window] opened "
                f"display={display_name} size={self.width}x{self.height}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                "[run3][alignment-window][WARN] open failed; "
                f"visual aid disabled: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self.close()
            self.enabled = False
            return False

    def _set_color(self, name):
        self.x11.XSetForeground(
            self.display, self.gc, self.colors[name]
        )

    def _line(self, x1, y1, x2, y2, color, width=1):
        self._set_color(color)
        self.x11.XSetLineAttributes(
            self.display, self.gc, width, 0, 1, 1
        )
        self.x11.XDrawLine(
            self.display,
            self.window,
            self.gc,
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    def _text(self, x, y, text, color="text"):
        payload = str(text).encode("ascii", errors="replace")
        self._set_color(color)
        self.x11.XDrawString(
            self.display,
            self.window,
            self.gc,
            int(x),
            int(y),
            payload,
            len(payload),
        )

    def _circle(self, x, y, radius, color, filled=False):
        self._set_color(color)
        draw = self.x11.XFillArc if filled else self.x11.XDrawArc
        draw(
            self.display,
            self.window,
            self.gc,
            int(round(x - radius)),
            int(round(y - radius)),
            int(round(2 * radius)),
            int(round(2 * radius)),
            0,
            360 * 64,
        )

    def _process_events(self):
        event_buffer = ctypes.create_string_buffer(192)
        while self.x11.XPending(self.display) > 0:
            self.x11.XNextEvent(
                self.display, ctypes.byref(event_buffer)
            )
            event_type = ctypes.c_int.from_buffer(
                event_buffer
            ).value
            if event_type == self.CLIENT_MESSAGE:
                client_event = ctypes.cast(
                    ctypes.byref(event_buffer),
                    ctypes.POINTER(self.XClientMessageEvent),
                ).contents
                if (
                    int(client_event.data.l[0])
                    == int(self.wm_delete_window)
                ):
                    print(
                        "[run3][alignment-window] closed by user",
                        flush=True,
                    )
                    self.enabled = False
                    return False
            elif event_type == self.DESTROY_NOTIFY:
                self.window = 0
                self.enabled = False
                return False
        return True

    def _world_to_screen(
        self,
        points,
        view_heading=None,
        view_origin=None,
    ):
        plot_left = 45.0
        plot_right = float(self.width - 45)
        plot_top = 125.0
        plot_bottom = float(self.height - 45)

        transformed_points = list(points)
        if view_heading is not None:
            if view_origin is None:
                view_origin = points[-1]
            rotation = math.pi / 2.0 - float(view_heading)
            rotation_cos = math.cos(rotation)
            rotation_sin = math.sin(rotation)
            transformed_points = []
            for point in points:
                dx = point[0] - view_origin[0]
                dy = point[1] - view_origin[1]
                transformed_points.append(
                    (
                        rotation_cos * dx - rotation_sin * dy,
                        rotation_sin * dx + rotation_cos * dy,
                    )
                )

        xs = [point[0] for point in transformed_points]
        ys = [point[1] for point in transformed_points]
        center_x = 0.5 * (min(xs) + max(xs))
        center_y = 0.5 * (min(ys) + max(ys))
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        overall_span = max(span_x, span_y, 5.0)
        span_x = max(span_x * 1.25, overall_span * 0.30)
        span_y = max(span_y * 1.25, overall_span * 0.30)
        scale = min(
            (plot_right - plot_left) / span_x,
            (plot_bottom - plot_top) / span_y,
        )
        screen_center_x = 0.5 * (plot_left + plot_right)
        screen_center_y = 0.5 * (plot_top + plot_bottom)

        def transform(point):
            return (
                screen_center_x + (point[0] - center_x) * scale,
                screen_center_y - (point[1] - center_y) * scale,
            )

        return [transform(point) for point in transformed_points]

    def _draw_waiting(self):
        self.x11.XClearWindow(self.display, self.window)
        self._text(25, 35, "Run3 Manual Alignment")
        self._text(
            25,
            65,
            "Waiting for PREPARE and the first current INS sample...",
            "muted",
        )
        self._text(
            25,
            95,
            "Blue: start-goal line   Red: ego heading",
            "muted",
        )
        self.x11.XFlush(self.display)

    def _draw(self, controller, ego, started):
        self.x11.XClearWindow(self.display, self.window)
        if (
            controller.start_xy is None
            or controller.goal_xy is None
        ):
            self._draw_waiting()
            return

        start = tuple(controller.start_xy)
        goal = tuple(controller.goal_xy)
        ego_available = ego is not None
        ego_point = (
            (float(ego["x"]), float(ego["y"]))
            if ego_available
            else start
        )
        ego_heading = (
            float(ego["heading"]) if ego_available else None
        )
        start_screen, goal_screen, ego_screen = (
            self._world_to_screen(
                [start, goal, ego_point],
                view_heading=ego_heading,
                view_origin=ego_point,
            )
        )

        target_heading = math.atan2(
            goal[1] - start[1], goal[0] - start[0]
        )

        self._text(22, 28, "Run3 Manual Alignment")
        if ego_available:
            heading_error = wrap_angle(
                target_heading - ego_heading
            )
            self._text(
                22,
                52,
                f"state={controller.state} "
                f"started={int(bool(started))}  "
                "view=EGO-UP  "
                f"speed={float(ego['speed']):.2f}m/s  "
                f"target={controller.manual_speed:.2f}m/s  "
                f"steer={controller.manual_steering_deg:.1f}deg",
            )
            error_color = (
                "ego"
                if abs(math.degrees(heading_error)) <= 3.0
                else "heading"
            )
            self._text(
                22,
                76,
                f"ego heading={math.degrees(ego_heading):.2f}deg  "
                f"goal line={math.degrees(target_heading):.2f}deg  "
                f"error={math.degrees(heading_error):+.2f}deg",
                error_color,
            )
        else:
            self._text(
                22,
                52,
                f"state={controller.state} "
                f"started={int(bool(started))}  waiting for INS...",
                "muted",
            )
            self._text(
                22,
                76,
                f"goal line={math.degrees(target_heading):.2f}deg  "
                "ego heading=unavailable",
                "line",
            )
        self._text(
            22,
            100,
            "Hold W/UP + LEFT/RIGHT to align. "
            "Release W to cruise. SPACE to sprint.",
            "muted",
        )

        self._line(
            start_screen[0],
            start_screen[1],
            goal_screen[0],
            goal_screen[1],
            "line",
            4,
        )
        self._circle(
            start_screen[0], start_screen[1], 7, "line", True
        )
        self._circle(
            goal_screen[0], goal_screen[1], 10, "goal", False
        )
        self._circle(
            goal_screen[0], goal_screen[1], 4, "goal", True
        )
        self._text(
            start_screen[0] + 10,
            start_screen[1] - 10,
            "START",
            "line",
        )
        self._text(
            goal_screen[0] + 10,
            goal_screen[1] - 10,
            "GOAL",
            "goal",
        )

        if ego_available:
            self._circle(
                ego_screen[0], ego_screen[1], 9, "ego", True
            )
            arrow_length = 72.0
            arrow_x = ego_screen[0]
            arrow_y = ego_screen[1] - arrow_length
            self._line(
                ego_screen[0],
                ego_screen[1],
                arrow_x,
                arrow_y,
                "heading",
                4,
            )
            self._line(
                arrow_x,
                arrow_y,
                arrow_x - 12.0,
                arrow_y + 18.0,
                "heading",
                3,
            )
            self._line(
                arrow_x,
                arrow_y,
                arrow_x + 12.0,
                arrow_y + 18.0,
                "heading",
                3,
            )
            self._text(
                arrow_x + 10,
                arrow_y,
                "FRONT",
                "heading",
            )
            self._text(
                ego_screen[0] + 12,
                ego_screen[1] + 20,
                "EGO",
                "ego",
            )
        self.x11.XFlush(self.display)

    def update(self, controller, ego, started):
        if not self.enabled or self.display is None:
            return
        if not self._process_events():
            self.close()
            return
        now = time.monotonic()
        if now - self.last_draw_time < 0.05:
            return
        self.last_draw_time = now
        self._draw(controller, ego, started)

    def close(self):
        if self.display is None:
            return
        if self.gc is not None:
            self.x11.XFreeGC(self.display, self.gc)
            self.gc = None
        if self.window:
            self.x11.XDestroyWindow(self.display, self.window)
            self.window = 0
        self.x11.XFlush(self.display)
        self.x11.XCloseDisplay(self.display)
        self.display = None
        self.x11 = None


class StraightSprintController:
    def __init__(self, args):
        self.manual_start = not bool(args.auto_align)
        self.manual_max_speed = max(
            0.0, float(args.manual_max_speed)
        )
        self.manual_acceleration = max(
            0.0, float(args.manual_acceleration)
        )
        self.manual_brake_deceleration = max(
            0.0, float(args.manual_brake_deceleration)
        )
        manual_steer_command_deg = abs(
            float(args.manual_steer_command_deg)
        )
        self.align_speed = max(0.0, float(args.align_speed))
        self.align_acceleration = float(args.align_acceleration)
        self.align_speed_kp = max(
            0.0, float(args.align_speed_kp)
        )
        self.align_max_deceleration = abs(
            float(args.align_max_deceleration)
        )
        self.align_tolerance_deg = max(
            0.0, float(args.align_tolerance_deg)
        )
        self.sprint_heading_tolerance_deg = max(
            self.align_tolerance_deg,
            float(args.sprint_heading_tolerance_deg),
        )
        self.sprint_max_lateral_miss = max(
            0.0, float(args.sprint_max_lateral_miss)
        )
        self.align_confirm_frames = max(
            1, int(args.align_confirm_frames)
        )
        self.align_min_duration = max(
            0.0, float(args.align_min_duration)
        )
        self.align_max_duration = max(
            self.align_min_duration,
            float(args.align_max_duration),
        )
        self.settle_speed = max(0.0, float(args.settle_speed))
        self.settle_duration = max(
            0.0, float(args.settle_duration)
        )
        self.settle_confirm_frames = max(
            1, int(args.settle_confirm_frames)
        )
        self.settle_yaw_rate_deg = max(
            0.0, float(args.settle_yaw_rate_deg)
        )
        self.settle_steering_deg = max(
            0.0, float(args.settle_steering_deg)
        )
        self.align_reentry_error_deg = max(
            self.align_tolerance_deg,
            float(args.align_reentry_error_deg),
        )
        self.steer_kp = float(args.steer_kp)
        self.align_yaw_damping = max(
            0.0, float(args.align_yaw_damping)
        )
        self.steer_sign = float(args.steer_sign)
        self.steer_limit_deg = abs(float(args.steer_limit_deg))
        self.manual_steer_command_deg = min(
            self.steer_limit_deg, manual_steer_command_deg
        )
        self.steer_min_deg = min(
            self.steer_limit_deg,
            abs(float(args.steer_min_deg)),
        )
        self.sprint_acceleration = float(args.sprint_acceleration)
        self.sprint_speed = max(0.0, float(args.sprint_speed))
        self.sprint_pulse_min_duration = max(
            0.01, float(args.sprint_pulse_min_duration)
        )
        self.sprint_pulse_max_duration = max(
            self.sprint_pulse_min_duration,
            float(args.sprint_pulse_max_duration),
        )
        self.sprint_pulse_speed_delta = max(
            0.0, float(args.sprint_pulse_speed_delta)
        )
        self.line_sample_step = max(0.1, float(args.line_sample_step))

        self.start_xy = None
        self.goal_xy = None
        self.line_heading = None
        self.line_length = 0.0
        self.line_points = []
        self.state = "IDLE"
        self.confirm_count = 0
        self.last_confirm_sequence = None
        self.align_started_time = None
        self.align_timeout_warned = False
        self.settle_started_time = None
        self.settle_confirm_count = 0
        self.sprint_pulse_started_time = None
        self.sprint_pulse_entry_speed = 0.0
        self.sprint_pulse_complete = False
        self.manual_speed = 0.0
        self.manual_steering_deg = 0.0
        self.manual_launch_requested = False
        self.auto_sprint_override_requested = False
        self.manual_last_input_time = None
        self.manual_last_held_state = None
        self.manual_last_input_log_time = 0.0
        self.last_log_time = 0.0

    def reset(self):
        self.start_xy = None
        self.goal_xy = None
        self.line_heading = None
        self.line_length = 0.0
        self.line_points = []
        self.state = "IDLE"
        self.confirm_count = 0
        self.last_confirm_sequence = None
        self.align_started_time = None
        self.align_timeout_warned = False
        self.settle_started_time = None
        self.settle_confirm_count = 0
        self.sprint_pulse_started_time = None
        self.sprint_pulse_entry_speed = 0.0
        self.sprint_pulse_complete = False
        self.manual_speed = 0.0
        self.manual_steering_deg = 0.0
        self.manual_launch_requested = False
        self.auto_sprint_override_requested = False
        self.manual_last_input_time = None
        self.manual_last_held_state = None
        self.manual_last_input_log_time = 0.0
        self.last_log_time = 0.0

    def configure(self, init_state, target_state):
        self.reset()
        try:
            start_x = float(init_state["x"])
            start_y = float(init_state["y"])
            goal_x = float(target_state["x"])
            goal_y = float(target_state["y"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[run3][prepare][ERROR] invalid start/goal: {exc}")
            self.state = "INVALID"
            return False

        coordinates = (start_x, start_y, goal_x, goal_y)
        if not all(math.isfinite(value) for value in coordinates):
            print(
                "[run3][prepare][ERROR] non-finite start/goal "
                f"values={coordinates}"
            )
            self.state = "INVALID"
            return False

        dx = goal_x - start_x
        dy = goal_y - start_y
        line_length = math.hypot(dx, dy)
        if line_length < 0.5:
            print(
                "[run3][prepare][ERROR] start/goal distance too small "
                f"distance={line_length:.3f}m"
            )
            self.state = "INVALID"
            return False

        self.goal_xy = (goal_x, goal_y)
        self._set_tracking_line(start_x, start_y)
        self.state = "MANUAL" if self.manual_start else "ALIGN"
        print(
            "[run3][straight-line] ready "
            f"start=({start_x:.3f},{start_y:.3f}) "
            f"goal=({goal_x:.3f},{goal_y:.3f}) "
            f"length={line_length:.3f}m "
            f"heading={math.degrees(self.line_heading):.3f}deg "
            f"points={len(self.line_points)} "
            f"mode={'MANUAL' if self.manual_start else 'AUTO'}",
            flush=True,
        )
        return True

    def update_manual_inputs(
        self,
        forward,
        brake,
        left,
        right,
        launch_requested=False,
        ego_speed=None,
    ):
        if self.state != "MANUAL":
            return False

        now = time.monotonic()
        if self.manual_last_input_time is None:
            input_dt = 0.0
        else:
            input_dt = min(
                0.1,
                max(0.0, now - self.manual_last_input_time),
            )
        self.manual_last_input_time = now

        if launch_requested:
            self.manual_steering_deg = 0.0
            self.manual_launch_requested = True
            print(
                "[run3][manual] SPACE launch requested; "
                f"hold speed target={self.manual_speed:.2f}m/s, "
                "centre steering, then sprint",
                flush=True,
            )
            return True

        # W selects the manual cruise target. Releasing W intentionally
        # leaves that target unchanged, avoiding a target-speed step to
        # zero. S has priority and ramps the target down gently.
        if brake:
            brake_reference = self.manual_speed
            try:
                candidate_speed = float(ego_speed)
                if math.isfinite(candidate_speed):
                    brake_reference = min(
                        brake_reference,
                        max(0.0, candidate_speed),
                    )
            except (TypeError, ValueError):
                pass
            new_speed = max(
                0.0,
                brake_reference
                - self.manual_brake_deceleration * input_dt,
            )
        elif forward:
            new_speed = self.manual_max_speed
        else:
            new_speed = self.manual_speed

        if bool(left) == bool(right):
            new_steering = 0.0
        elif left:
            new_steering = self.manual_steer_command_deg
        else:
            new_steering = -self.manual_steer_command_deg

        changed = (
            new_speed != self.manual_speed
            or new_steering != self.manual_steering_deg
        )
        held_state = (
            bool(forward),
            bool(brake),
            bool(left),
            bool(right),
        )
        held_state_changed = held_state != self.manual_last_held_state
        self.manual_last_held_state = held_state
        self.manual_speed = new_speed
        self.manual_steering_deg = new_steering
        log_due = (
            held_state_changed
            or now - self.manual_last_input_log_time >= 0.25
            or (brake and self.manual_speed <= 0.0)
        )
        if (changed or held_state_changed) and log_due:
            self.manual_last_input_log_time = now
            print(
                "[run3][manual] held "
                f"forward={int(bool(forward))} "
                f"brake={int(bool(brake))} "
                f"left={int(bool(left))} "
                f"right={int(bool(right))} "
                f"speed={self.manual_speed:.2f}m/s "
                f"steer={self.manual_steering_deg:.1f}deg",
                flush=True,
            )
        return changed or held_state_changed

    def request_auto_sprint_override(self):
        if self.manual_start or self.state not in ("ALIGN", "SETTLE"):
            return False
        if self.auto_sprint_override_requested:
            return True
        self.auto_sprint_override_requested = True
        print(
            "[run3][AUTO] SPACE override requested; "
            "zero steering and sprint on current heading",
            flush=True,
        )
        return True

    def _set_tracking_line(self, start_x, start_y):
        goal_x, goal_y = self.goal_xy
        dx = goal_x - start_x
        dy = goal_y - start_y
        line_length = math.hypot(dx, dy)
        self.start_xy = (float(start_x), float(start_y))
        self.line_heading = math.atan2(dy, dx)
        self.line_length = line_length
        segment_count = max(
            1, int(math.ceil(line_length / self.line_sample_step))
        )
        self.line_points = [
            (
                start_x + dx * index / segment_count,
                start_y + dy * index / segment_count,
            )
            for index in range(segment_count + 1)
        ]

    def _set_manual_sprint_line(
        self, start_x, start_y, straight_heading
    ):
        self.start_xy = (float(start_x), float(start_y))
        self.line_heading = wrap_angle(float(straight_heading))
        goal_dx = self.goal_xy[0] - start_x
        goal_dy = self.goal_xy[1] - start_y
        self.line_length = math.hypot(goal_dx, goal_dy)
        self.line_points = []

    def _speed_acceleration(
        self,
        target_speed,
        ego_speed,
        max_acceleration=None,
    ):
        command = self.align_speed_kp * (
            float(target_speed) - float(ego_speed)
        )
        acceleration_limit = (
            self.align_acceleration
            if max_acceleration is None
            else max(0.0, float(max_acceleration))
        )
        return max(
            -self.align_max_deceleration,
            min(acceleration_limit, command),
        )

    def _alignment_steering(
        self, heading_error_deg, ego_yaw_rate_deg
    ):
        if abs(heading_error_deg) <= self.align_tolerance_deg:
            return 0.0
        predicted_error_deg = (
            heading_error_deg
            - self.align_yaw_damping * ego_yaw_rate_deg
        )
        raw_steering = (
            self.steer_sign
            * self.steer_kp
            * predicted_error_deg
        )
        if abs(raw_steering) < 1e-9:
            return 0.0
        steering_magnitude = max(
            self.steer_min_deg, abs(raw_steering)
        )
        return math.copysign(
            min(self.steer_limit_deg, steering_magnitude),
            raw_steering,
        )

    def command(
        self,
        ego_x,
        ego_y,
        ego_heading,
        ego_speed,
        ego_yaw_rate,
        steering_feedback,
        ins_sequence,
    ):
        if (
            self.state
            not in (
                "MANUAL",
                "MANUAL_CENTER",
                "ALIGN",
                "SETTLE",
                "SPRINT",
            )
            or self.goal_xy is None
            or self.line_heading is None
        ):
            return 0.0, 0.0, 0.0

        now = time.monotonic()
        if self.align_started_time is None:
            self.align_started_time = now
        align_elapsed = now - self.align_started_time

        goal_dx = self.goal_xy[0] - ego_x
        goal_dy = self.goal_xy[1] - ego_y
        distance_to_goal = math.hypot(goal_dx, goal_dy)
        goal_heading = math.atan2(goal_dy, goal_dx)

        tx = math.cos(self.line_heading)
        ty = math.sin(self.line_heading)
        relative_x = ego_x - self.start_xy[0]
        relative_y = ego_y - self.start_xy[1]
        along_track = relative_x * tx + relative_y * ty
        cross_track = relative_y * tx - relative_x * ty
        remaining_along = self.line_length - along_track

        # Align to the immutable start-to-goal axis. A moving pure-pursuit
        # bearing can run away when the vehicle starts with a large heading
        # error: while the car travels beside the line, the goal bearing
        # rotates faster than the chassis can turn. Once alignment ends,
        # steering is permanently zero for this episode.
        guidance_heading = self.line_heading
        heading_error = wrap_angle(guidance_heading - ego_heading)
        heading_error_deg = math.degrees(heading_error)
        ego_yaw_rate_deg = math.degrees(float(ego_yaw_rate))
        steering_feedback_valid = (
            steering_feedback is not None
            and math.isfinite(float(steering_feedback))
        )
        steering_feedback_deg = (
            float(steering_feedback)
            if steering_feedback_valid
            else float("nan")
        )

        fresh_ins = (
            ins_sequence is not None
            and ins_sequence != self.last_confirm_sequence
        )
        if fresh_ins:
            self.last_confirm_sequence = ins_sequence

        if (
            self.auto_sprint_override_requested
            and self.state in ("ALIGN", "SETTLE")
        ):
            previous_state = self.state
            self.auto_sprint_override_requested = False
            self._set_manual_sprint_line(
                ego_x, ego_y, ego_heading
            )
            tx = math.cos(self.line_heading)
            ty = math.sin(self.line_heading)
            along_track = 0.0
            cross_track = 0.0
            remaining_along = self.line_length
            guidance_heading = self.line_heading
            heading_error = 0.0
            heading_error_deg = 0.0
            self.state = "SPRINT"
            self.sprint_pulse_started_time = now
            self.sprint_pulse_entry_speed = float(ego_speed)
            self.sprint_pulse_complete = False
            print(
                f"[run3] {previous_state} -> SPRINT by SPACE override "
                f"straight_heading="
                f"{math.degrees(self.line_heading):.3f}deg "
                f"speed={ego_speed:.3f}m/s "
                "command_steer=0 without automatic settle wait",
                flush=True,
            )

        if self.state == "ALIGN":
            if fresh_ins:
                if abs(heading_error_deg) <= self.align_tolerance_deg:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 0

            ready_for_settle = (
                align_elapsed >= self.align_min_duration
                and self.confirm_count >= self.align_confirm_frames
            )
            if (
                align_elapsed >= self.align_max_duration
                and not self.align_timeout_warned
            ):
                self.align_timeout_warned = True
                print(
                    "[run3][ALIGN][WARN] alignment window exceeded; "
                    "continue aligning instead of unsafe sprint "
                    f"elapsed={align_elapsed:.3f}s "
                    f"heading_error={heading_error_deg:.3f}deg "
                    f"confirm={self.confirm_count}/"
                    f"{self.align_confirm_frames}",
                    flush=True,
                )
            if ready_for_settle:
                self.state = "SETTLE"
                self.settle_started_time = now
                self.settle_confirm_count = 0
                print(
                    "[run3] ALIGN -> SETTLE "
                    f"elapsed={align_elapsed:.3f}s "
                    f"heading_error={heading_error_deg:.3f}deg "
                    f"speed={ego_speed:.3f}m/s; "
                    "command steer=0 and wait for yaw/steering decay",
                    flush=True,
                )

        if self.state == "MANUAL" and self.manual_launch_requested:
            self.manual_launch_requested = False
            self.state = "MANUAL_CENTER"
            self.settle_started_time = now
            self.settle_confirm_count = 0
            print(
                "[run3] MANUAL -> MANUAL_CENTER "
                f"ego_heading={math.degrees(ego_heading):.3f}deg "
                f"speed={ego_speed:.3f}m/s "
                f"commanded_steer={self.manual_steering_deg:.3f}deg",
                flush=True,
            )

        settle_elapsed = (
            0.0
            if self.settle_started_time is None
            else now - self.settle_started_time
        )
        if self.state == "MANUAL_CENTER":
            yaw_ok = (
                abs(ego_yaw_rate_deg)
                <= self.settle_yaw_rate_deg
            )
            steering_ok = (
                not steering_feedback_valid
                or abs(steering_feedback_deg)
                <= self.settle_steering_deg
            )
            stable_now = (
                settle_elapsed >= self.settle_duration
                and yaw_ok
                and steering_ok
            )
            if fresh_ins:
                if stable_now:
                    self.settle_confirm_count += 1
                else:
                    self.settle_confirm_count = 0
            if (
                self.settle_confirm_count
                >= self.settle_confirm_frames
            ):
                self._set_manual_sprint_line(
                    ego_x, ego_y, ego_heading
                )
                tx = math.cos(self.line_heading)
                ty = math.sin(self.line_heading)
                along_track = 0.0
                cross_track = 0.0
                remaining_along = self.line_length
                guidance_heading = self.line_heading
                heading_error = 0.0
                heading_error_deg = 0.0
                self.state = "SPRINT"
                self.sprint_pulse_started_time = now
                self.sprint_pulse_entry_speed = float(ego_speed)
                self.sprint_pulse_complete = False
                print(
                    "[run3] MANUAL_CENTER -> SPRINT "
                    f"straight_heading="
                    f"{math.degrees(self.line_heading):.3f}deg "
                    f"yaw_rate={ego_yaw_rate_deg:.3f}deg/s "
                    f"steer_feedback={steering_feedback_deg:.3f}deg "
                    f"stable={self.settle_confirm_count}/"
                    f"{self.settle_confirm_frames}",
                    flush=True,
                )

        if self.state == "SETTLE":
            if (
                fresh_ins
                and abs(heading_error_deg)
                > self.align_reentry_error_deg
            ):
                self.state = "ALIGN"
                self.confirm_count = 0
                self.settle_confirm_count = 0
                self.settle_started_time = None
                print(
                    "[run3] SETTLE -> ALIGN "
                    f"heading drifted to {heading_error_deg:.3f}deg "
                    f"(limit={self.align_reentry_error_deg:.3f}deg)",
                    flush=True,
                )
            else:
                heading_ok = (
                    abs(heading_error_deg)
                    <= self.align_tolerance_deg
                )
                yaw_ok = (
                    abs(ego_yaw_rate_deg)
                    <= self.settle_yaw_rate_deg
                )
                steering_ok = (
                    not steering_feedback_valid
                    or abs(steering_feedback_deg)
                    <= self.settle_steering_deg
                )
                stable_now = (
                    settle_elapsed >= self.settle_duration
                    and heading_ok
                    and yaw_ok
                    and steering_ok
                )
                if fresh_ins:
                    if stable_now:
                        self.settle_confirm_count += 1
                    else:
                        self.settle_confirm_count = 0
                if (
                    self.settle_confirm_count
                    >= self.settle_confirm_frames
                ):
                    self._set_tracking_line(ego_x, ego_y)
                    tx = math.cos(self.line_heading)
                    ty = math.sin(self.line_heading)
                    along_track = 0.0
                    cross_track = 0.0
                    remaining_along = self.line_length
                    guidance_heading = self.line_heading
                    heading_error = wrap_angle(
                        guidance_heading - ego_heading
                    )
                    heading_error_deg = math.degrees(
                        heading_error
                    )
                    predicted_lateral_miss = abs(
                        math.sin(heading_error) * self.line_length
                    )
                    sprint_heading_ok = (
                        abs(heading_error_deg)
                        <= self.sprint_heading_tolerance_deg
                    )
                    sprint_miss_ok = (
                        predicted_lateral_miss
                        <= self.sprint_max_lateral_miss
                    )
                    if not (sprint_heading_ok and sprint_miss_ok):
                        self.state = "ALIGN"
                        self.confirm_count = 0
                        self.align_started_time = now
                        self.align_timeout_warned = False
                        self.settle_started_time = None
                        self.settle_confirm_count = 0
                        print(
                            "[run3] SETTLE -> ALIGN after re-anchor "
                            f"heading_error={heading_error_deg:.3f}deg "
                            f"(sprint_limit="
                            f"{self.sprint_heading_tolerance_deg:.3f}deg) "
                            f"predicted_miss="
                            f"{predicted_lateral_miss:.3f}m "
                            f"(limit="
                            f"{self.sprint_max_lateral_miss:.3f}m) "
                            f"anchor=({ego_x:.3f},{ego_y:.3f}) "
                            f"remaining_line={self.line_length:.3f}m",
                            flush=True,
                        )
                    else:
                        self.state = "SPRINT"
                        self.sprint_pulse_started_time = now
                        self.sprint_pulse_entry_speed = float(ego_speed)
                        self.sprint_pulse_complete = False
                        print(
                            "[run3] SETTLE -> SPRINT "
                            f"heading_error={heading_error_deg:.3f}deg "
                            f"predicted_miss="
                            f"{predicted_lateral_miss:.3f}m "
                            f"yaw_rate={ego_yaw_rate_deg:.3f}deg/s "
                            f"steer_feedback="
                            f"{steering_feedback_deg:.3f}deg "
                            f"stable={self.settle_confirm_count}/"
                            f"{self.settle_confirm_frames} "
                            f"anchor=({ego_x:.3f},{ego_y:.3f}) "
                            f"remaining_line={self.line_length:.3f}m",
                            flush=True,
                        )

        if self.state == "SPRINT":
            pulse_elapsed = (
                0.0
                if self.sprint_pulse_started_time is None
                else now - self.sprint_pulse_started_time
            )
            pulse_speed_delta = (
                float(ego_speed) - self.sprint_pulse_entry_speed
            )
            pulse_acknowledged = (
                pulse_elapsed >= self.sprint_pulse_min_duration
                and pulse_speed_delta >= self.sprint_pulse_speed_delta
            )
            pulse_timed_out = (
                pulse_elapsed >= self.sprint_pulse_max_duration
            )
            if (
                not self.sprint_pulse_complete
                and (pulse_acknowledged or pulse_timed_out)
            ):
                self.sprint_pulse_complete = True
                print(
                    "[run3] SPRINT pulse complete -> HOLD "
                    f"reason="
                    f"{'speed_ack' if pulse_acknowledged else 'timeout'} "
                    f"elapsed={pulse_elapsed:.3f}s "
                    f"speed_delta={pulse_speed_delta:.3f}m/s "
                    f"speed={self.sprint_speed:.3f}m/s "
                    "acc=0 steer=0",
                    flush=True,
                )
            acceleration = (
                0.0
                if self.sprint_pulse_complete
                else self.sprint_acceleration
            )
            target_speed = self.sprint_speed
            guidance_heading = self.line_heading
            heading_error = wrap_angle(
                guidance_heading - ego_heading
            )
            heading_error_deg = math.degrees(heading_error)
            steering = 0.0
        elif self.state == "SETTLE":
            acceleration = self._speed_acceleration(
                self.settle_speed, ego_speed
            )
            target_speed = self.settle_speed
            steering = 0.0
        elif self.state == "MANUAL_CENTER":
            acceleration = self._speed_acceleration(
                self.manual_speed,
                ego_speed,
                self.manual_acceleration,
            )
            target_speed = self.manual_speed
            steering = 0.0
        elif self.state == "MANUAL":
            acceleration = self._speed_acceleration(
                self.manual_speed,
                ego_speed,
                self.manual_acceleration,
            )
            target_speed = self.manual_speed
            steering = self.manual_steering_deg
        else:
            # Keep closing the speed error on every control frame. A lost
            # first frame can no longer leave the vehicle stopped forever.
            acceleration = self._speed_acceleration(
                self.align_speed, ego_speed
            )
            target_speed = self.align_speed
            steering = self._alignment_steering(
                heading_error_deg, ego_yaw_rate_deg
            )

        if now - self.last_log_time >= 0.25:
            self.last_log_time = now
            print(
                "[run3][control] "
                f"state={self.state} "
                f"ego=({ego_x:.3f},{ego_y:.3f}) "
                f"ego_heading={math.degrees(ego_heading):.3f}deg "
                f"line_heading={math.degrees(self.line_heading):.3f}deg "
                f"guidance_heading="
                f"{math.degrees(guidance_heading):.3f}deg "
                f"heading_error={heading_error_deg:.3f}deg "
                f"cross_track={cross_track:.3f}m "
                f"remaining_along={remaining_along:.3f}m "
                f"align_elapsed={align_elapsed:.3f}s "
                f"settle_elapsed={settle_elapsed:.3f}s "
                f"yaw_rate={ego_yaw_rate_deg:.3f}deg/s "
                f"steer_feedback={steering_feedback_deg:.3f}deg "
                f"stable={self.settle_confirm_count}/"
                f"{self.settle_confirm_frames} "
                f"pulse_active="
                f"{int(self.state == 'SPRINT' and not self.sprint_pulse_complete)} "
                f"ego_speed={ego_speed:.3f}m/s "
                f"goal_distance={distance_to_goal:.3f}m "
                f"acc={acceleration:.1f} speed={target_speed:.1f} "
                f"steer={steering:.3f}",
                flush=True,
            )

        return acceleration, target_speed, steering


class Run3:
    def __init__(self, args):
        self.args = args
        self.actor_id = str(args.actor_id)
        self.role_id = self.actor_id
        self.session_id = ""
        self.map_name = ""
        self.prepared = False
        self.started = False
        self.prepare_result_sent = False
        self.ego = None
        self.first_current_ins_ready = False
        self.last_ins_sequence = None
        self.last_heading = None
        self.last_heading_monotonic = None
        self.filtered_yaw_rate = 0.0
        self.last_ins_warning_time = 0.0
        self.start_gate_xy = None
        self.start_gate_tolerance = max(
            0.0, float(args.ins_start_gate_tolerance)
        )
        self.prepare_prime_duration = max(
            0.0, float(args.prepare_prime_duration)
        )

        self.controller = StraightSprintController(args)
        self.keyboard = LinuxKeyboardStateReader()
        self.keyboard_available = False
        self.alignment_window = X11AlignmentWindow(
            enabled=(
                self.controller.manual_start
                and not bool(args.no_alignment_window)
            ),
            width=args.alignment_window_width,
            height=args.alignment_window_height,
        )
        self.latest_feedback = None
        self.control_period = 1.0 / max(1.0, float(args.control_hz))
        self.last_control_time = 0.0
        self.prepare_channel = None
        self.notify_channel = None
        self.ins_channel = None
        self.control_channel = None

    def create_channels(self):
        chassis_module = sys.modules.get(VehicleControl.__module__)
        steering_field = VehicleControl.DESCRIPTOR.fields_by_name.get(
            "steering_control"
        )
        target_field = None
        if steering_field is not None:
            target_field = steering_field.message_type.fields_by_name.get(
                "target_steering_wheel_angle"
            )
        steering_fields = (
            "missing"
            if steering_field is None
            else ",".join(
                f"{field.name}#{field.number}:type{field.type}"
                for field in steering_field.message_type.fields
            )
        )
        print(
            "[run3][sdk] "
            f"root={_ONSITE_PROTO_ROOT} "
            f"pb2={getattr(chassis_module, '__file__', 'unknown')} "
            f"network={getattr(libMulticastNetwork, '__file__', 'unknown')} "
            f"vehicle_control={VehicleControl.DESCRIPTOR.full_name} "
            f"steering_field_no="
            f"{getattr(steering_field, 'number', 'missing')} "
            f"target_angle_field_no="
            f"{getattr(target_field, 'number', 'missing')} "
            f"steering_fields={steering_fields}",
            flush=True,
        )
        param = libMulticastNetwork.CreateChannelsParam()
        param.config_center_addr = self.args.config_center
        param.local_ip = get_ip_address(self.args.net_interface)
        param.net_interface_name = self.args.net_interface
        param.field_id = self.args.field_id
        param.log_level = 1
        param.client_name = self.actor_id
        param.recv_self_msg = False

        print(f"local_ip {param.local_ip}", flush=True)
        channels = libMulticastNetwork.ChannelPtrVector()
        result = libMulticastNetwork.create_channels(param, channels)
        if result:
            raise RuntimeError(f"create channels failed, ret={result}")

        channel_map = {}
        for channel in channels:
            print(
                "message channel name: {}, id: {}".format(
                    channel.name(), channel.id()
                ),
                flush=True,
            )
            channel_map[channel.name()] = channel

        required = ("prepare", "notify", "ins", "vehiclecontrol")
        missing = [name for name in required if name not in channel_map]
        if missing:
            raise RuntimeError(
                f"required multicast channels missing: {missing}; "
                f"available={sorted(channel_map)}"
            )

        self.prepare_channel = channel_map["prepare"]
        self.notify_channel = channel_map["notify"]
        self.ins_channel = channel_map["ins"]
        self.control_channel = channel_map["vehiclecontrol"]
        print(
            "[run3] communication ready: "
            "prepare/notify/ins/vehiclecontrol",
            flush=True,
        )

    def reset_episode_state(self, keep_session=True):
        current_session = self.session_id
        self.prepared = False
        self.started = False
        self.prepare_result_sent = False
        self.map_name = ""
        self.role_id = self.actor_id
        self.ego = None
        self.first_current_ins_ready = False
        self.last_ins_sequence = None
        self.last_heading = None
        self.last_heading_monotonic = None
        self.filtered_yaw_rate = 0.0
        self.latest_feedback = None
        self.start_gate_xy = None
        self.controller.reset()
        if not keep_session:
            self.session_id = ""
        else:
            self.session_id = current_session

    def send_prepare_result(self, accepted):
        result = ActorPrepareResult()
        result.session_id = self.session_id
        result.actor_id = self.actor_id
        result.result = bool(accepted)
        payload = result.SerializeToString()
        ret = self.prepare_channel.put(
            MT_ACTOR_PREPARE_RESULT, len(payload), payload
        )
        if ret != 0:
            print(
                f"[run3][prepare][ERROR] send result failed ret={ret}",
                flush=True,
            )
            return False
        self.prepare_result_sent = bool(accepted)
        print(
            "[run3][prepare] result sent "
            f"session={self.session_id} accepted={int(bool(accepted))}",
            flush=True,
        )
        return True

    def prime_ins_subscription(self):
        deadline = time.monotonic() + self.prepare_prime_duration
        poll_count = 0
        last_sequence = None
        while True:
            ins = self.ins_channel.get_ins()
            poll_count += 1
            try:
                last_sequence = int(ins.sequence_num)
            except Exception:
                last_sequence = None

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.005, remaining))

        print(
            "[run3][prepare] INS subscription primed "
            f"duration={self.prepare_prime_duration:.3f}s "
            f"polls={poll_count} last_seq={last_sequence}",
            flush=True,
        )

    def handle_prepare(self, prepare_message):
        incoming_session = str(
            prepare_message.session_id or ""
        ).strip()
        if not incoming_session:
            print("[run3][prepare][WARN] empty session ignored")
            return

        if incoming_session == self.session_id:
            if self.prepare_result_sent:
                print(
                    "[run3][prepare] duplicate ignored "
                    f"session={incoming_session}",
                    flush=True,
                )
            return

        if self.session_id and not is_newer_session(
            incoming_session, self.session_id
        ):
            print(
                "[run3][prepare][WARN] stale/unordered prepare ignored "
                f"received={incoming_session} current={self.session_id}",
                flush=True,
            )
            return

        if self.session_id:
            print(
                "[run3][prepare] supersede session "
                f"old={self.session_id} new={incoming_session}",
                flush=True,
            )

        self.reset_episode_state(keep_session=False)
        self.session_id = incoming_session
        try:
            brief_data = json.loads(
                prepare_message.archive_info.brief_data
            )
            testees = brief_data.get("testees") or []
            if not testees:
                raise ValueError("brief_data has no testees")
            testee = testees[0]
            init_state = testee["init_state"]
            target_state = testee["target_state"]
            self.role_id = str(
                testee.get("role_id") or self.actor_id
            )
            self.map_name = str(
                brief_data.get("zjl_odv_file")
                or brief_data.get("map_name")
                or brief_data.get("map_id")
                or ""
            )
            self.start_gate_xy = (
                float(init_state["x"]),
                float(init_state["y"]),
            )
            line_ready = self.controller.configure(
                init_state, target_state
            )
        except Exception as exc:
            print(
                "[run3][prepare][ERROR] brief_data rejected "
                f"session={self.session_id} error={type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )
            self.send_prepare_result(False)
            return

        self.prepared = bool(line_ready)
        self.started = False
        print(
            "[run3][prepare] accepted "
            f"session={self.session_id} role={self.role_id} "
            f"map={self.map_name}",
            flush=True,
        )
        if self.prepared:
            neutral_sent = self.send_control(0.0, 0.0, 0.0)
            if neutral_sent:
                print(
                    "[run3][prepare] neutral control sent before "
                    "PrepareResult acc=0 speed=0 steer=0",
                    flush=True,
                )
            else:
                print(
                    "[run3][prepare][ERROR] neutral control before "
                    "PrepareResult failed; reject prepare",
                    flush=True,
                )
                self.prepared = False
        if self.prepared:
            self.prime_ins_subscription()
        self.send_prepare_result(self.prepared)

    def poll_prepare(self):
        for _ in range(16):
            ret, message = self.prepare_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != MT_ACTOR_PREPARE:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            prepare_message = ActorPrepare()
            prepare_message.ParseFromString(payload)
            self.handle_prepare(prepare_message)

    def handle_notify(self, notify):
        notify_session = str(notify.session_id or "").strip()
        session_sensitive = notify.type in (
            NT_START_TEST,
            NT_FINISH_TEST,
            NT_ABORT_TEST,
        )
        if session_sensitive and self.session_id:
            if not notify_session:
                print(
                    "[run3][notify][WARN] empty-session event ignored "
                    f"type={notify.type} current={self.session_id}",
                    flush=True,
                )
                return
            if notify_session != self.session_id:
                print(
                    "[run3][notify][WARN] stale session ignored "
                    f"type={notify.type} received={notify_session} "
                    f"current={self.session_id}",
                    flush=True,
                )
                return

        if notify.type == NT_START_TEST:
            if not self.prepared or not self.prepare_result_sent:
                print(
                    "[run3][notify][WARN] start ignored before prepare "
                    f"session={notify_session}",
                    flush=True,
                )
                return
            # Synchronize the current remote/local key state and discard
            # any SPACE edge received before START. Do not clear held keys:
            # X11 reports them continuously, while evdev reports edges.
            self.keyboard.poll()
            self.keyboard.consume_launch_request()
            self.started = True
            startup_speed = (
                float(self.ego["speed"])
                if self.ego is not None
                else 0.0
            )
            neutral_sent = self.send_control(
                acceleration=0.0,
                speed=startup_speed,
                steering=0.0,
            )
            print(
                "[run3][notify] START "
                f"session={self.session_id} role={self.role_id} "
                f"neutral_control={int(neutral_sent)} "
                f"startup_speed={startup_speed:.3f}m/s",
                flush=True,
            )
            return

        if notify.type in (NT_FINISH_TEST, NT_ABORT_TEST):
            event_name = (
                "FINISH" if notify.type == NT_FINISH_TEST else "ABORT"
            )
            print(
                f"[run3][notify] {event_name} "
                f"session={self.session_id} "
                f"controller_state={self.controller.state}",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            self.reset_episode_state(keep_session=True)
            return

        if notify.type == NT_DESTROY_ROLE:
            print("[run3][notify] DESTROY_ROLE", flush=True)
            self.send_control(0.0, 0.0, 0.0)
            self.reset_episode_state(keep_session=True)
            return

        if notify.type == NT_COLLIDE_ROLE:
            print(
                "[run3][notify] COLLIDE_ROLE; wait for finish/abort",
                flush=True,
            )

    def poll_notify(self):
        for _ in range(32):
            ret, message = self.notify_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != MT_NOTIFY:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            notify = Notify()
            notify.ParseFromString(payload)
            self.handle_notify(notify)

    def poll_ins(self):
        ins = self.ins_channel.get_ins()
        try:
            sequence = int(ins.sequence_num)
        except Exception:
            return
        if sequence <= 1 or sequence > 1_000_000:
            return
        if sequence == self.last_ins_sequence:
            return

        try:
            x = float(ins.position.x)
            y = float(ins.position.y)
            heading = float(ins.heading)
            vx = float(ins.linear_velocity.x)
            vy = float(ins.linear_velocity.y)
        except Exception as exc:
            self.warn_ins(f"missing/invalid fields: {exc}")
            return
        if not all(math.isfinite(value) for value in (
            x, y, heading, vx, vy
        )):
            self.warn_ins(
                f"non-finite sample seq={sequence} "
                f"x={x} y={y} heading={heading}"
            )
            return

        if (
            self.prepared
            and not self.first_current_ins_ready
            and self.start_gate_xy is not None
        ):
            start_distance = math.hypot(
                x - self.start_gate_xy[0],
                y - self.start_gate_xy[1],
            )
            if start_distance > self.start_gate_tolerance:
                self.warn_ins(
                    "stale pre-start sample ignored "
                    f"seq={sequence} pos=({x:.3f},{y:.3f}) "
                    f"start=({self.start_gate_xy[0]:.3f},"
                    f"{self.start_gate_xy[1]:.3f}) "
                    f"distance={start_distance:.3f}m "
                    f"threshold={self.start_gate_tolerance:.3f}m"
                )
                return
            self.first_current_ins_ready = True
            print(
                "[run3][ins] first current-episode sample accepted "
                f"seq={sequence} distance_to_start={start_distance:.3f}m",
                flush=True,
            )

        sample_time = time.monotonic()
        if (
            self.last_heading is None
            or self.last_heading_monotonic is None
        ):
            yaw_rate = 0.0
        else:
            sample_dt = max(
                1e-3,
                sample_time - self.last_heading_monotonic,
            )
            raw_yaw_rate = wrap_angle(
                heading - self.last_heading
            ) / sample_dt
            # A small first-order filter suppresses INS quantization while
            # retaining the residual turn rate needed by SETTLE.
            filter_alpha = min(1.0, sample_dt / 0.10)
            self.filtered_yaw_rate += filter_alpha * (
                raw_yaw_rate - self.filtered_yaw_rate
            )
            yaw_rate = self.filtered_yaw_rate
        self.last_heading = heading
        self.last_heading_monotonic = sample_time
        self.last_ins_sequence = sequence
        self.ego = {
            "x": x,
            "y": y,
            "heading": heading,
            "speed": math.hypot(vx, vy),
            "yaw_rate": yaw_rate,
            "sequence": sequence,
        }

    def warn_ins(self, text):
        now = time.monotonic()
        if now - self.last_ins_warning_time < 1.0:
            return
        self.last_ins_warning_time = now
        print(f"[run3][ins][WARN] {text}", flush=True)

    def poll_feedback(self):
        for _ in range(8):
            ret, message = self.control_channel.get()
            if message is None:
                return
            if ret < 0 or message.type() != VEHICLE_FEEDBACK:
                continue
            payload = libMulticastNetwork.getMessageData(message)
            feedback = VehicleFeedback()
            feedback.ParseFromString(payload)
            self.latest_feedback = feedback

    def send_control(self, acceleration, speed, steering):
        command = VehicleControl()
        command.acceleration = float(acceleration)
        command.speed = float(speed)
        command.steering_control.target_steering_wheel_angle = float(
            steering
        )
        payload = command.SerializeToString()
        ret = self.control_channel.put(
            VEHICLE_CONTROL, len(payload), payload
        )
        if ret != 0:
            print(
                f"[run3][control][ERROR] send failed ret={ret}",
                flush=True,
            )
            return False
        return True

    def send_active_control(self):
        now = time.monotonic()
        if now - self.last_control_time < self.control_period:
            return
        self.last_control_time = now

        if not self.started:
            return
        if not self.first_current_ins_ready or self.ego is None:
            self.send_control(0.0, 0.0, 0.0)
            return

        steering_feedback = None
        if self.latest_feedback is not None:
            try:
                candidate = float(
                    self.latest_feedback.steering_feedback
                    .steering_wheel_angle
                )
                if math.isfinite(candidate):
                    steering_feedback = candidate
            except Exception:
                steering_feedback = None

        acceleration, speed, steering = self.controller.command(
            self.ego["x"],
            self.ego["y"],
            self.ego["heading"],
            self.ego["speed"],
            self.ego["yaw_rate"],
            steering_feedback,
            self.ego["sequence"],
        )
        self.send_control(acceleration, speed, steering)

    def poll_keyboard(self):
        if not self.keyboard_available:
            return
        self.keyboard.poll()
        launch_requested = self.keyboard.consume_launch_request()
        if not self.started:
            return
        if not self.controller.manual_start:
            if launch_requested:
                self.controller.request_auto_sprint_override()
            return
        self.controller.update_manual_inputs(
            forward=self.keyboard.is_pressed(
                self.keyboard.KEY_W,
                self.keyboard.KEY_UP,
            ),
            brake=self.keyboard.is_pressed(
                self.keyboard.KEY_S,
                self.keyboard.KEY_DOWN,
            ),
            left=self.keyboard.is_pressed(
                self.keyboard.KEY_A,
                self.keyboard.KEY_LEFT,
            ),
            right=self.keyboard.is_pressed(
                self.keyboard.KEY_D,
                self.keyboard.KEY_RIGHT,
            ),
            launch_requested=launch_requested,
            ego_speed=(
                self.ego["speed"]
                if self.ego is not None
                else None
            ),
        )

    def run(self):
        self.create_channels()
        keyboard_device = None
        try:
            keyboard_device = self.keyboard.open(
                self.args.keyboard_device
            )
            self.keyboard_available = True
        except RuntimeError as exc:
            if self.controller.manual_start:
                raise
            print(
                "[run3][AUTO][WARN] SPACE sprint override disabled; "
                f"automatic flow continues unchanged: {exc}",
                flush=True,
            )

        if self.controller.manual_start:
            self.alignment_window.open()
            print(
                "[run3][manual] controls ready: "
                f"device={keyboard_device} "
                f"hold W/UP={self.controller.manual_max_speed:.1f}m/s, "
                f"manual_acc_limit="
                f"{self.controller.manual_acceleration:.1f}m/s2, "
                "release W/UP=hold target, "
                "hold S/DOWN=slow target reduction at "
                f"{self.controller.manual_brake_deceleration:.1f}m/s2, "
                "hold A/LEFT or D/RIGHT="
                f"{self.controller.manual_steer_command_deg:.1f}deg, "
                "simultaneous keys enabled, "
                "SPACE hold speed/centre then sprint",
                flush=True,
            )
        elif self.keyboard_available:
            print(
                "[run3][AUTO] SPACE override ready: "
                f"device={keyboard_device}; "
                "during ALIGN/SETTLE press SPACE to command steer=0 "
                "and enter SPRINT immediately",
                flush=True,
            )
        print(
            "[run3] minimal straight sprint started "
            f"start_mode="
            f"{'MANUAL_WASD' if self.controller.manual_start else 'AUTO'} "
            f"align_speed={self.controller.align_speed:.2f}m/s "
            f"align_acc={self.controller.align_acceleration:.2f}m/s2 "
            f"control_hz={1.0 / self.control_period:.1f} "
            f"align_speed_kp={self.controller.align_speed_kp:.2f} "
            f"align_tolerance={self.controller.align_tolerance_deg:.2f}deg "
            f"sprint_tolerance="
            f"{self.controller.sprint_heading_tolerance_deg:.2f}deg/"
            f"{self.controller.sprint_max_lateral_miss:.2f}m "
            f"confirm_frames={self.controller.align_confirm_frames} "
            f"align_min/warn="
            f"[{self.controller.align_min_duration:.1f},"
            f"{self.controller.align_max_duration:.1f}]s "
            f"steer_kp={self.controller.steer_kp:.2f} "
            f"yaw_damping="
            f"{self.controller.align_yaw_damping:.2f}s "
            f"steer_range="
            f"[{self.controller.steer_min_deg:.1f},"
            f"{self.controller.steer_limit_deg:.1f}]deg "
            "steering_fields=TARGET_ONLY "
            f"settle={self.controller.settle_duration:.1f}s/"
            f"{self.controller.settle_confirm_frames}frames "
            f"yaw_limit={self.controller.settle_yaw_rate_deg:.2f}deg/s "
            f"steer_feedback_limit="
            f"{self.controller.settle_steering_deg:.2f}deg "
            "sprint_steer=LOCKED_ZERO "
            "sprint_acc_mode=REPEATED_LOGICAL_PULSE "
            f"sprint_pulse={self.controller.sprint_pulse_min_duration:.2f}-"
            f"{self.controller.sprint_pulse_max_duration:.2f}s/"
            f"{self.controller.sprint_pulse_speed_delta:.2f}m/s "
            f"sprint_acc={self.controller.sprint_acceleration:.1f} "
            f"sprint_speed={self.controller.sprint_speed:.1f}",
            flush=True,
        )
        try:
            while True:
                self.poll_prepare()
                self.poll_notify()
                self.poll_ins()
                self.poll_feedback()
                self.poll_keyboard()
                self.send_active_control()
                self.alignment_window.update(
                    self.controller, self.ego, self.started
                )
                time.sleep(min(0.002, 0.25 * self.control_period))
        finally:
            self.alignment_window.close()
            self.keyboard.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal start-to-goal straight sprint runner"
    )
    parser.add_argument(
        "--config_center",
        default="47.110.233.70:52009",
    )
    parser.add_argument(
        "--field_id",
        default="field-zd-test1-22-0331134113-888",
    )
    parser.add_argument(
        "--net_interface",
        default="usb0",
    )
    parser.add_argument("--actor-id", default="apollo_testee")
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="use automatic heading alignment instead of manual WASD",
    )
    parser.add_argument(
        "--keyboard-device",
        default="",
        help=(
            "force a Linux evdev keyboard path; by default remote-capable "
            "X11 input is tried first, then evdev is auto-detected "
            "(example: /dev/input/event3)"
        ),
    )
    parser.add_argument(
        "--no-alignment-window",
        action="store_true",
        help="disable the X11 manual-alignment visual aid",
    )
    parser.add_argument(
        "--alignment-window-width",
        type=int,
        default=760,
        help="manual-alignment window width (default: 760)",
    )
    parser.add_argument(
        "--alignment-window-height",
        type=int,
        default=560,
        help="manual-alignment window height (default: 560)",
    )
    parser.add_argument(
        "--manual-max-speed",
        type=float,
        default=20.0,
        help="maximum manual driving speed (default: 20.0 m/s)",
    )
    parser.add_argument(
        "--manual-acceleration",
        type=float,
        default=3.0,
        help=(
            "maximum acceleration in MANUAL/MANUAL_CENTER "
            "(default: 3.0 m/s^2)"
        ),
    )
    parser.add_argument(
        "--manual-brake-deceleration",
        type=float,
        default=1.0,
        help=(
            "target-speed reduction rate while S/DOWN is held "
            "(default: 1.0 m/s^2)"
        ),
    )
    parser.add_argument(
        "--manual-steer-command-deg",
        "--manual-steer-step-deg",
        dest="manual_steer_command_deg",
        type=float,
        default=24.0,
        help=(
            "steering-wheel angle while A/LEFT or D/RIGHT is held "
            "(default: 24 deg)"
        ),
    )
    parser.add_argument(
        "--manual-speed-step",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--align-speed",
        type=float,
        default=3.0,
        help="heading-alignment target (default: 3.0 m/s)",
    )
    parser.add_argument(
        "--align-acceleration",
        type=float,
        default=10.0,
        help="maximum heading-alignment acceleration (default: 10 m/s^2)",
    )
    parser.add_argument(
        "--align-pulse-acceleration",
        type=float,
        default=7200.0,
        help=(
            "deprecated compatibility option; ignored because ALIGN now "
            "uses closed-loop acceleration"
        ),
    )
    parser.add_argument(
        "--align-speed-kp",
        type=float,
        default=2.0,
        help="alignment speed proportional gain (default: 2.0 1/s)",
    )
    parser.add_argument(
        "--align-max-deceleration",
        type=float,
        default=4.0,
        help="maximum alignment/settling deceleration (default: 4 m/s^2)",
    )
    parser.add_argument(
        "--align-tolerance-deg",
        type=float,
        default=3.0,
        help="heading tolerance before settling (default: 3.0 deg)",
    )
    parser.add_argument(
        "--sprint-heading-tolerance-deg",
        type=float,
        default=8.0,
        help=(
            "maximum re-anchored heading error allowed before sprint "
            "(default: 8.0 deg)"
        ),
    )
    parser.add_argument(
        "--sprint-max-lateral-miss",
        type=float,
        default=3.5,
        help=(
            "maximum predicted lateral miss at the goal when sprint steering "
            "is locked to zero (default: 3.5 m)"
        ),
    )
    parser.add_argument(
        "--align-confirm-frames",
        type=int,
        default=2,
        help="fresh in-tolerance INS frames before settling (default: 2)",
    )
    parser.add_argument(
        "--align-min-duration",
        type=float,
        default=1.0,
        help="minimum moving-alignment time (default: 1.0 s)",
    )
    parser.add_argument(
        "--align-max-duration",
        type=float,
        default=2.0,
        help=(
            "alignment warning time; never forces an unsafe sprint "
            "(default: 2.0 s)"
        ),
    )
    parser.add_argument(
        "--settle-speed",
        type=float,
        default=3.0,
        help="speed target while steering/yaw settle (default: 3 m/s)",
    )
    parser.add_argument(
        "--settle-duration",
        type=float,
        default=0.1,
        help="minimum steer-zero settling duration (default: 0.1 s)",
    )
    parser.add_argument(
        "--settle-confirm-frames",
        type=int,
        default=2,
        help="stable fresh INS frames before sprint (default: 2)",
    )
    parser.add_argument(
        "--settle-yaw-rate-deg",
        type=float,
        default=1.0,
        help="maximum absolute yaw rate before sprint (default: 1.0 deg/s)",
    )
    parser.add_argument(
        "--settle-steering-deg",
        type=float,
        default=2.0,
        help="maximum steering feedback before sprint (default: 2 deg)",
    )
    parser.add_argument(
        "--align-reentry-error-deg",
        type=float,
        default=3.0,
        help="return SETTLE to ALIGN above this error (default: 3 deg)",
    )
    parser.add_argument(
        "--steer-kp",
        type=float,
        default=3.0,
        help="steering-wheel degrees per heading-error degree (default: 3.0)",
    )
    parser.add_argument(
        "--align-yaw-damping",
        type=float,
        default=0.4,
        help=(
            "predictive yaw-rate damping horizon used to avoid alignment "
            "overshoot (default: 0.4 s)"
        ),
    )
    parser.add_argument("--steer-sign", type=float, default=1.0)
    parser.add_argument(
        "--steer-min-deg",
        type=float,
        default=0.0,
        help="minimum wheel command outside tolerance (default: 0 deg)",
    )
    parser.add_argument(
        "--steer-limit-deg",
        type=float,
        default=42.0,
        help="alignment steering-wheel limit (default: 42 deg)",
    )
    parser.add_argument(
        "--steering-rate-deg",
        type=float,
        default=720.0,
        help=(
            "deprecated compatibility option; ignored because only the "
            "target steering-wheel angle is transmitted"
        ),
    )
    parser.add_argument(
        "--directive-steering-ratio",
        type=float,
        default=6.3,
        help=(
            "deprecated compatibility option; ignored because the directive "
            "wheel field is not transmitted"
        ),
    )
    parser.add_argument(
        "--no-steering-compat-fields",
        action="store_true",
        help=(
            "deprecated no-op; only target_steering_wheel_angle is always "
            "populated"
        ),
    )
    parser.add_argument(
        "--sprint-acceleration", type=float, default=10000.0
    )
    parser.add_argument(
        "--sprint-speed", type=float, default=10000.0
    )
    parser.add_argument(
        "--sprint-pulse-min-duration",
        type=float,
        default=0.08,
        help=(
            "minimum duration for repeating the logical acceleration pulse "
            "(default: 0.08 s)"
        ),
    )
    parser.add_argument(
        "--sprint-pulse-max-duration",
        type=float,
        default=0.20,
        help=(
            "maximum duration for repeating the logical acceleration pulse "
            "(default: 0.20 s)"
        ),
    )
    parser.add_argument(
        "--sprint-pulse-speed-delta",
        type=float,
        default=0.10,
        help=(
            "measured speed increase acknowledging the sprint pulse "
            "(default: 0.10 m/s)"
        ),
    )
    parser.add_argument(
        "--line-sample-step", type=float, default=1.0
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=100.0 / 3.0,
        help=(
            "VehicleControl publication rate matching the proven final "
            "publisher (default: 33.333 Hz)"
        ),
    )
    parser.add_argument(
        "--ins-start-gate-tolerance",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--prepare-prime-duration",
        type=float,
        default=0.10,
        help=(
            "poll and prime the INS subscription for this long before "
            "ActorPrepareResult (default: 0.10 s)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="accepted for launcher compatibility; control logs are always on",
    )
    parser.add_argument(
        "--run-log-dir",
        default=str(_SCRIPT_DIR / "debug_logs" / "run_hmxzw"),
        help="single-process runtime log archive directory",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="disable the top-level run_hmxzw terminal log archive",
    )
    return parser.parse_args()


def _log_source_identity():
    source_path = Path(__file__).resolve()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(
        f"[run3][source] path={source_path} sha256={source_hash}",
        flush=True,
    )


def main():
    run_log_archive = None
    try:
        args = parse_args()
        if not args.no_run_log:
            run_log_archive = _install_run_logging(args)
        _log_source_identity()
        Run3(args).run()
    except KeyboardInterrupt:
        print("[run3] stopped by user", flush=True)
    except Exception as exc:
        print(
            f"[run3][FATAL] {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        if run_log_archive is not None:
            run_log_archive.flush()


if __name__ == "__main__":
    main()
