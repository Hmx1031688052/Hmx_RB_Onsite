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
  - apply one INS-frame acceleration pulse
  - hold a constant speed target with acceleration/steering locked to zero
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None


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


TASK_TIMEOUT_RESTART_EXIT_CODE = 75
SIMULATOR_STALL_RESTART_EXIT_CODE = 76
SUPERVISOR_LOCK_PATH = Path("/tmp/run_hmxzw-supervisor.lock")


class _TeeTextIO:
    """Mirror supervisor/runtime text to the terminal and one archive."""

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
    """Archive the supervisor plus every restarted runtime child."""
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


def _start_runtime_child(child_argv):
    """Start a child and relay its output through the supervisor tee."""
    process = subprocess.Popen(
        child_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )

    def relay_output():
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            process.stdout.close()

    output_thread = threading.Thread(
        target=relay_output,
        name=f"run3-runtime-output-{process.pid}",
        daemon=True,
    )
    output_thread.start()
    process.run3_output_thread = output_thread
    return process


def acquire_supervisor_lock():
    """Allow only one process to manage this runtime and DriverSim."""
    if fcntl is None:
        return None
    lock_file = SUPERVISOR_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown"
        lock_file.close()
        print(
            "[run3][supervisor][FATAL] another supervisor is running "
            f"pid={owner}",
            flush=True,
        )
        return False
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    print(
        "[run3][supervisor] singleton lock acquired "
        f"pid={os.getpid()}",
        flush=True,
    )
    return lock_file


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


class StraightSprintController:
    def __init__(self, args):
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
        self.steer_min_deg = min(
            self.steer_limit_deg,
            abs(float(args.steer_min_deg)),
        )
        self.sprint_acceleration = float(args.sprint_acceleration)
        self.sprint_speed = max(0.0, float(args.sprint_speed))
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
        self.sprint_pulse_sequence = None
        self.sprint_pulse_complete = False
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
        self.sprint_pulse_sequence = None
        self.sprint_pulse_complete = False
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
        self.state = "ALIGN"
        print(
            "[run3][straight-line] ready "
            f"start=({start_x:.3f},{start_y:.3f}) "
            f"goal=({goal_x:.3f},{goal_y:.3f}) "
            f"length={line_length:.3f}m "
            f"heading={math.degrees(self.line_heading):.3f}deg "
            f"points={len(self.line_points)}",
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

    def _speed_acceleration(self, target_speed, ego_speed):
        command = self.align_speed_kp * (
            float(target_speed) - float(ego_speed)
        )
        return max(
            -self.align_max_deceleration,
            min(self.align_acceleration, command),
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
            self.state not in ("ALIGN", "SETTLE", "SPRINT")
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

        settle_elapsed = (
            0.0
            if self.settle_started_time is None
            else now - self.settle_started_time
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
                        self.sprint_pulse_sequence = ins_sequence
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
            if (
                not self.sprint_pulse_complete
                and fresh_ins
                and self.sprint_pulse_sequence is not None
                and ins_sequence != self.sprint_pulse_sequence
            ):
                self.sprint_pulse_complete = True
                print(
                    "[run3] SPRINT pulse complete -> HOLD "
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
        self.test_started_monotonic = None
        self.last_ins_monotonic = None

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

        self.controller = StraightSprintController(args)
        self.latest_feedback = None
        self.control_period = 1.0 / max(1.0, float(args.control_hz))
        self.last_control_time = 0.0
        self.first_control_before_prepare_sent = False

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
        self.test_started_monotonic = None
        self.last_ins_monotonic = None
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
        self.first_control_before_prepare_sent = False
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
            # The official run.py publishes one neutral VehicleControl frame
            # before acknowledging Prepare. This establishes control
            # ownership/mode in DriverSim before the scenario starts.
            self.first_control_before_prepare_sent = self.send_control(
                0.0, 0.0, 0.0
            )
            if not self.first_control_before_prepare_sent:
                print(
                    "[run3][prepare][ERROR] neutral control before "
                    "PrepareResult failed; reject prepare",
                    flush=True,
                )
                self.prepared = False
            else:
                print(
                    "[run3][prepare] first neutral control sent before "
                    "PrepareResult acc=0 speed=0 steer=0",
                    flush=True,
                )
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
            self.started = True
            self.test_started_monotonic = time.monotonic()
            self.last_ins_monotonic = None
            startup_speed = (
                float(self.ego["speed"])
                if self.ego is not None
                else 0.0
            )
            startup_control_sent = self.send_control(
                acceleration=0.0,
                speed=startup_speed,
                steering=0.0,
            )
            print(
                "[run3][notify] START "
                f"session={self.session_id} role={self.role_id} "
                f"timeout={self.args.task_timeout:.1f}s "
                f"neutral_control={int(startup_control_sent)} "
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
        self.last_ins_monotonic = sample_time
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

    def enforce_task_timeout(self):
        if (
            not self.started
            or self.test_started_monotonic is None
        ):
            return
        elapsed = time.monotonic() - self.test_started_monotonic
        waiting_for_first_ins = self.last_ins_monotonic is None
        last_ins_time = (
            self.test_started_monotonic
            if waiting_for_first_ins
            else self.last_ins_monotonic
        )
        ins_silence = time.monotonic() - last_ins_time
        active_ins_timeout = (
            self.args.first_ins_timeout
            if waiting_for_first_ins
            else self.args.ins_stall_timeout
        )
        if (
            active_ins_timeout > 0.0
            and ins_silence >= active_ins_timeout
        ):
            print(
                "[run3][WATCHDOG] "
                f"{'first INS timeout' if waiting_for_first_ins else 'INS stalled'} "
                f"for {ins_silence:.3f}s "
                f"(limit={active_ins_timeout:.1f}s); "
                f"session={self.session_id}; "
                "restart DriverSim and runtime",
                flush=True,
            )
            self.send_control(0.0, 0.0, 0.0)
            os._exit(SIMULATOR_STALL_RESTART_EXIT_CODE)

        if self.args.task_timeout <= 0.0:
            return
        if elapsed < self.args.task_timeout:
            return

        print(
            "[run3][WATCHDOG] task did not finish within "
            f"{self.args.task_timeout:.1f}s; "
            f"session={self.session_id} elapsed={elapsed:.3f}s; "
            "terminate runtime child for a clean reconnect",
            flush=True,
        )
        self.send_control(0.0, 0.0, 0.0)
        # A process-level exit is intentional. It guarantees that native
        # multicast channels and sockets are released before the supervisor
        # starts a fresh runtime and re-registers the testee.
        os._exit(TASK_TIMEOUT_RESTART_EXIT_CODE)

    def run(self):
        self.create_channels()
        print(
            "[run3] minimal straight sprint started "
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
            "sprint_acc_mode=ONE_INS_PULSE_THEN_ZERO "
            f"sprint_acc={self.controller.sprint_acceleration:.1f} "
            f"sprint_speed={self.controller.sprint_speed:.1f}",
            flush=True,
        )
        while True:
            self.poll_prepare()
            self.poll_notify()
            self.poll_ins()
            self.poll_feedback()
            self.send_active_control()
            self.enforce_task_timeout()
            time.sleep(min(0.002, 0.25 * self.control_period))


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
        default=0.5,
        help="heading tolerance before settling (default: 0.5 deg)",
    )
    parser.add_argument(
        "--sprint-heading-tolerance-deg",
        type=float,
        default=2.0,
        help=(
            "maximum re-anchored heading error allowed before sprint "
            "(default: 2.0 deg)"
        ),
    )
    parser.add_argument(
        "--sprint-max-lateral-miss",
        type=float,
        default=0.5,
        help=(
            "maximum predicted lateral miss at the goal when sprint steering "
            "is locked to zero (default: 0.5 m)"
        ),
    )
    parser.add_argument(
        "--align-confirm-frames",
        type=int,
        default=5,
        help="fresh in-tolerance INS frames before settling (default: 5)",
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
        default=0.4,
        help="minimum steer-zero settling duration (default: 0.4 s)",
    )
    parser.add_argument(
        "--settle-confirm-frames",
        type=int,
        default=5,
        help="stable fresh INS frames before sprint (default: 5)",
    )
    parser.add_argument(
        "--settle-yaw-rate-deg",
        type=float,
        default=0.5,
        help="maximum absolute yaw rate before sprint (default: 0.5 deg/s)",
    )
    parser.add_argument(
        "--settle-steering-deg",
        type=float,
        default=1.0,
        help="maximum steering feedback before sprint (default: 1 deg)",
    )
    parser.add_argument(
        "--align-reentry-error-deg",
        type=float,
        default=1.0,
        help="return SETTLE to ALIGN above this error (default: 1 deg)",
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
        "--debug",
        action="store_true",
        help="accepted for launcher compatibility; control logs are always on",
    )
    parser.add_argument(
        "--run-log-dir",
        default=str(_SCRIPT_DIR / "debug_logs" / "run_hmxzw"),
        help="supervisor/runtime log archive directory",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="disable the top-level run_hmxzw terminal log archive",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=20.0,
        help=(
            "restart the complete runtime if a started task has not ended "
            "within this many seconds; <=0 disables (default: 20)"
        ),
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=3.0,
        help=(
            "delay allowing the daemon to observe the disconnect before "
            "reconnecting (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--simulator-dir",
        default=(
            "/media/pc/FanXiang2T/Onsite_FirstWithForth/"
            "LinuxNoEditor416"
        ),
        help="directory containing the managed DriverSim start.sh",
    )
    parser.add_argument(
        "--simulator-ready-delay",
        type=float,
        default=2.0,
        help=(
            "extra settling delay after DriverSim PID appears "
            "(default: 2)"
        ),
    )
    parser.add_argument(
        "--simulator-start-timeout",
        type=float,
        default=30.0,
        help=(
            "maximum wait for a real DriverSim PID after start.sh "
            "(default: 30)"
        ),
    )
    parser.add_argument(
        "--max-simulator-start-failures",
        type=int,
        default=3,
        help=(
            "stop after this many consecutive startup failures; "
            "<=0 retries forever (default: 3)"
        ),
    )
    parser.add_argument(
        "--simulator-log-dir",
        default="",
        help=(
            "DriverSim launch/crash archive directory; default is "
            "Hmx_RB_Onsite/debug_logs/simulator"
        ),
    )
    parser.add_argument(
        "--first-ins-timeout",
        type=float,
        default=15.0,
        help=(
            "allow this long for the first post-START INS while loading "
            "the scene; <=0 disables (default: 15)"
        ),
    )
    parser.add_argument(
        "--ins-stall-timeout",
        type=float,
        default=4.0,
        help=(
            "after START, restart DriverSim if no fresh INS arrives for this "
            "many seconds; <=0 disables (default: 4)"
        ),
    )
    parser.add_argument(
        "--no-manage-simulator",
        action="store_true",
        help="leave LinuxNoEditor external and restart only this runtime",
    )
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _terminate_managed_process(process, label):
    if process is None or process.poll() is not None:
        return
    print(
        f"[run3][supervisor] terminate {label} pid={process.pid}",
        flush=True,
    )
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        print(
            f"[run3][supervisor] kill unresponsive {label} "
            f"pid={process.pid}",
            flush=True,
        )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _path_is_within(path, root):
    try:
        return os.path.commonpath(
            (str(Path(path).resolve()), str(Path(root).resolve()))
        ) == str(Path(root).resolve())
    except (OSError, ValueError):
        return False


def _find_driver_sim_pids(simulator_dir):
    """Return actual DriverSim PIDs, ignoring the short-lived start.sh."""
    if os.name != "posix":
        return []

    simulator_root = Path(simulator_dir).expanduser().resolve()
    driver_pids = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            exe_path = Path(os.readlink(proc_dir / "exe"))
        except (ValueError, OSError):
            continue
        if (
            exe_path.name.lower().startswith("driversim")
            and _path_is_within(exe_path, simulator_root)
        ):
            driver_pids.append(pid)
    return sorted(driver_pids)


def _kill_simulator_residuals(simulator_dir, reason):
    """SIGKILL detached DriverSim/crash-window processes under one build."""
    if os.name != "posix":
        return

    simulator_root = Path(simulator_dir).expanduser().resolve()
    own_pid = os.getpid()
    residual_pids = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid == own_pid:
            continue

        try:
            exe_path = Path(os.readlink(proc_dir / "exe"))
        except OSError:
            exe_path = None
        try:
            cwd_path = Path(os.readlink(proc_dir / "cwd"))
        except OSError:
            cwd_path = None
        try:
            command_parts = (
                (proc_dir / "cmdline")
                .read_bytes()
                .decode(errors="replace")
                .split("\0")
            )
        except OSError:
            command_parts = []

        executable_name = (
            "" if exe_path is None else exe_path.name.lower()
        )
        managed_binary = (
            exe_path is not None
            and _path_is_within(exe_path, simulator_root)
            and (
                executable_name.startswith("driversim")
                or "crashreport" in executable_name
            )
        )
        managed_launcher = (
            cwd_path is not None
            and _path_is_within(cwd_path, simulator_root)
            and any(
                Path(part).name == "start.sh"
                for part in command_parts
                if part
            )
        )
        if managed_binary or managed_launcher:
            residual_pids.append(pid)

    for pid in sorted(set(residual_pids), reverse=True):
        print(
            "[run3][supervisor] SIGKILL simulator residual "
            f"pid={pid} reason={reason}",
            flush=True,
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if residual_pids:
        time.sleep(0.5)


def _simulator_log_root(args):
    configured = str(args.simulator_log_dir or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_SCRIPT_DIR / "debug_logs" / "simulator").resolve()


def _diagnostic_stamp():
    milliseconds = int(time.time() * 1000) % 1000
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{milliseconds:03d}"


def _copy_recent_simulator_artifacts(source_dir, target_dir, limit=8):
    if not source_dir.is_dir():
        return []
    candidates = []
    for path in source_dir.rglob("*"):
        try:
            if path.is_file():
                candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    copied = []
    for _, source in sorted(candidates, reverse=True)[:limit]:
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(relative))
        except OSError as exc:
            copied.append(f"ERROR {relative}: {exc}")
    return copied


def _capture_diagnostic_command(target, command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=8.0,
            check=False,
        )
        output = result.stdout or ""
        if len(output) > 200_000:
            output = output[-200_000:]
        target.write_text(
            f"command={command!r}\nreturncode={result.returncode}\n"
            f"{output}",
            encoding="utf-8",
        )
    except Exception as exc:
        target.write_text(
            f"command={command!r}\n"
            f"capture_error={type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )


def _write_simulator_crash_summary(event_dir, launch_log_path):
    rules = (
        (
            "UE4 checked-cast fatal（场景对象类型不匹配）",
            re.compile(
                r"Cast of .* to .* failed|CastChecked",
                re.IGNORECASE,
            ),
        ),
        (
            "GPU/Vulkan 设备丢失或显卡驱动异常",
            re.compile(
                r"VK_ERROR_DEVICE_LOST|GPU.?Crash|NVRM: Xid|"
                r"device lost",
                re.IGNORECASE,
            ),
        ),
        (
            "内存或显存不足",
            re.compile(
                r"out of memory|oom-kill|killed process|"
                r"memory allocation failed",
                re.IGNORECASE,
            ),
        ),
        (
            "动态库缺失或加载失败",
            re.compile(
                r"error while loading shared libraries|"
                r"cannot open shared object file|undefined symbol",
                re.IGNORECASE,
            ),
        ),
        (
            "云端登录或 channels 配置失败",
            re.compile(
                r"登录失败|http request failed|channels info is empty",
                re.IGNORECASE,
            ),
        ),
        (
            "原生崩溃（段错误/异常终止）",
            re.compile(
                r"segmentation fault|signal 11|core dumped|"
                r"fatal error",
                re.IGNORECASE,
            ),
        ),
    )

    sources = []
    if launch_log_path:
        sources.append(Path(launch_log_path))
    for path in event_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in {".log", ".txt", ".xml", ".json"}
        ):
            sources.append(path)

    findings = []
    seen = set()
    for source in sources:
        try:
            data = source.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        if len(data) > 500_000:
            data = data[-500_000:]
        for line in data.splitlines():
            for cause, pattern in rules:
                if not pattern.search(line):
                    continue
                key = (cause, line.strip())
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    (cause, str(source), line.strip()[:1000])
                )
                break
            if len(findings) >= 30:
                break
        if len(findings) >= 30:
            break

    summary_lines = []
    if findings:
        summary_lines.append(f"初步判断：{findings[0][0]}")
        summary_lines.append("")
        summary_lines.append("匹配证据：")
        for cause, source, line in findings:
            summary_lines.append(f"- [{cause}] {source}")
            summary_lines.append(f"  {line}")
    else:
        summary_lines.extend(
            (
                "初步判断：未匹配到常见崩溃特征。",
                "",
                "请重点检查 Saved/Logs、Saved/Crashes、"
                "coredumpctl.txt 和 kernel-warnings.txt。",
            )
        )
    (event_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def _archive_simulator_diagnostics(args, reason, launcher=None):
    """Copy existing UE logs/crash data before killing or restarting it."""
    try:
        log_root = _simulator_log_root(args)
        event_dir = (
            log_root
            / "events"
            / f"{_diagnostic_stamp()}_{reason}"
        )
        suffix = 1
        while event_dir.exists():
            event_dir = event_dir.with_name(
                f"{event_dir.name}_{suffix}"
            )
            suffix += 1
        event_dir.mkdir(parents=True, exist_ok=False)

        simulator_root = Path(
            args.simulator_dir
        ).expanduser().resolve()
        saved_root = simulator_root / "DriverSim" / "Saved"
        copied_logs = _copy_recent_simulator_artifacts(
            saved_root / "Logs",
            event_dir / "Saved" / "Logs",
        )

        copied_crash = []
        crashes_root = saved_root / "Crashes"
        if crashes_root.is_dir():
            crash_dirs = []
            for crash_dir in crashes_root.iterdir():
                try:
                    if crash_dir.is_dir():
                        crash_dirs.append(
                            (crash_dir.stat().st_mtime, crash_dir)
                        )
                except OSError:
                    continue
            if crash_dirs:
                newest_crash = max(
                    crash_dirs, key=lambda item: item[0]
                )[1]
                target = (
                    event_dir
                    / "Saved"
                    / "Crashes"
                    / newest_crash.name
                )
                try:
                    shutil.copytree(newest_crash, target)
                    copied_crash.append(newest_crash.name)
                except OSError as exc:
                    copied_crash.append(
                        f"ERROR {newest_crash.name}: {exc}"
                    )

        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "simulator_dir": str(simulator_root),
            "driver_pids": _find_driver_sim_pids(
                args.simulator_dir
            ),
            "launcher_pid": (
                None if launcher is None else launcher.pid
            ),
            "launcher_code": (
                None if launcher is None else launcher.poll()
            ),
            "launch_log": (
                None
                if launcher is None
                else getattr(launcher, "run3_log_path", None)
            ),
            "copied_logs": copied_logs,
            "copied_crash": copied_crash,
        }
        (event_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if os.name == "posix":
            _capture_diagnostic_command(
                event_dir / "nvidia-smi.txt",
                ["nvidia-smi"],
            )
            _capture_diagnostic_command(
                event_dir / "kernel-warnings.txt",
                ["dmesg", "--ctime", "--level=err,warn"],
            )
            _capture_diagnostic_command(
                event_dir / "coredumpctl.txt",
                [
                    "coredumpctl",
                    "--no-pager",
                    "info",
                    "DriverSim",
                ],
            )
        _write_simulator_crash_summary(
            event_dir,
            metadata["launch_log"],
        )
        print(
            "[run3][supervisor] simulator diagnostics archived "
            f"reason={reason} path={event_dir}",
            flush=True,
        )
        return event_dir
    except Exception as exc:
        print(
            "[run3][supervisor][WARN] diagnostic archive failed "
            f"reason={reason} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _start_managed_simulator(args):
    simulator_dir = Path(args.simulator_dir).expanduser().resolve()
    start_script = simulator_dir / "start.sh"
    if not start_script.is_file():
        raise FileNotFoundError(
            f"DriverSim start script not found: {start_script}"
        )
    if start_script.stat().st_size == 0:
        raise RuntimeError(
            f"DriverSim start script is empty: {start_script}; "
            "refuse to enter the restart loop"
        )
    launch_log_dir = _simulator_log_root(args) / "launches"
    launch_log_dir.mkdir(parents=True, exist_ok=True)
    launch_log_path = (
        launch_log_dir
        / f"driversim_{_diagnostic_stamp()}_{os.getpid()}.log"
    )
    print(
        "[run3][supervisor] start DriverSim "
        f"cwd={simulator_dir} command='bash start.sh' "
        f"log={launch_log_path}",
        flush=True,
    )
    with launch_log_path.open("ab", buffering=0) as launch_log:
        process = subprocess.Popen(
            ["bash", "start.sh"],
            cwd=str(simulator_dir),
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    process.run3_log_path = str(launch_log_path)
    (
        _simulator_log_root(args) / "latest_launch_log.txt"
    ).write_text(str(launch_log_path), encoding="utf-8")
    return process


def _wait_for_driver_sim(args, launcher):
    timeout = max(0.1, float(args.simulator_start_timeout))
    deadline = time.monotonic() + timeout
    launcher_exit_seen = None
    while time.monotonic() < deadline:
        driver_pids = _find_driver_sim_pids(args.simulator_dir)
        if driver_pids:
            settle_delay = max(
                0.0, float(args.simulator_ready_delay)
            )
            if settle_delay:
                time.sleep(settle_delay)
            return _find_driver_sim_pids(args.simulator_dir)

        launcher_code = launcher.poll()
        if launcher_code is not None:
            if launcher_exit_seen is None:
                launcher_exit_seen = time.monotonic()
            elif time.monotonic() - launcher_exit_seen >= 1.0:
                break
        time.sleep(0.2)
    return []


def supervise_runtime(args):
    child_argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--runtime-child",
    ]
    restart_count = 0
    simulator_start_failures = 0
    simulator = None
    runtime = None
    while True:
        try:
            if not args.no_manage_simulator:
                driver_pids = _find_driver_sim_pids(args.simulator_dir)
                if driver_pids:
                    simulator = None
                    print(
                        "[run3][supervisor] attach existing DriverSim "
                        f"pids={driver_pids}",
                        flush=True,
                    )
                else:
                    simulator = _start_managed_simulator(args)
                    driver_pids = _wait_for_driver_sim(args, simulator)

                if not driver_pids:
                    launcher_code = simulator.poll()
                    simulator_start_failures += 1
                    print(
                        "[run3][supervisor] DriverSim process not found "
                        "after startup wait "
                        f"launcher_code={launcher_code} "
                        f"failure={simulator_start_failures}",
                        flush=True,
                    )
                    _archive_simulator_diagnostics(
                        args,
                        reason="startup_process_missing",
                        launcher=simulator,
                    )
                    _terminate_managed_process(
                        simulator, "DriverSim launcher"
                    )
                    _kill_simulator_residuals(
                        args.simulator_dir,
                        reason="startup_process_missing",
                    )
                    simulator = None
                    failure_limit = int(
                        args.max_simulator_start_failures
                    )
                    if (
                        failure_limit > 0
                        and simulator_start_failures >= failure_limit
                    ):
                        print(
                            "[run3][supervisor][FATAL] stop after "
                            f"{simulator_start_failures} consecutive "
                            "DriverSim startup failures",
                            flush=True,
                        )
                        return 1
                    time.sleep(max(0.0, float(args.restart_delay)))
                    continue
                simulator_start_failures = 0
                launcher_code = (
                    None if simulator is None else simulator.poll()
                )
                print(
                    "[run3][supervisor] DriverSim process ready "
                    f"pids={driver_pids} launcher_code={launcher_code}",
                    flush=True,
                )

            print(
                "[run3][supervisor] start runtime "
                f"attempt={restart_count + 1} "
                f"task_timeout={args.task_timeout:.1f}s "
                f"first_ins_timeout={args.first_ins_timeout:.1f}s "
                f"ins_stall_timeout={args.ins_stall_timeout:.1f}s",
                flush=True,
            )
            runtime = _start_runtime_child(child_argv)
            restart_reason = None
            return_code = None
            while restart_reason is None:
                return_code = runtime.poll()
                if return_code is not None:
                    if return_code == TASK_TIMEOUT_RESTART_EXIT_CODE:
                        restart_reason = "task_timeout"
                    elif (
                        return_code
                        == SIMULATOR_STALL_RESTART_EXIT_CODE
                    ):
                        restart_reason = "ins_stall"
                    else:
                        break
                if (
                    not args.no_manage_simulator
                    and not _find_driver_sim_pids(args.simulator_dir)
                ):
                    restart_reason = "simulator_process_missing"
                if restart_reason is None and return_code is None:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print(
                "[run3][supervisor] interrupted; stop managed processes",
                flush=True,
            )
            _terminate_managed_process(runtime, "runtime")
            _terminate_managed_process(simulator, "DriverSim")
            if not args.no_manage_simulator:
                _kill_simulator_residuals(
                    args.simulator_dir,
                    reason="supervisor_interrupted",
                )
            return 130

        if restart_reason is None:
            _terminate_managed_process(simulator, "DriverSim")
            if not args.no_manage_simulator:
                _kill_simulator_residuals(
                    args.simulator_dir,
                    reason="runtime_exit",
                )
            print(
                "[run3][supervisor] runtime exited "
                f"code={return_code}; supervisor stops",
                flush=True,
            )
            return return_code

        if not args.no_manage_simulator:
            if restart_reason == "simulator_process_missing":
                # Give UE's crash handler a moment to finish CrashContext
                # and minidump files before they are copied and killed.
                time.sleep(1.0)
            _archive_simulator_diagnostics(
                args,
                reason=restart_reason,
                launcher=simulator,
            )
        _terminate_managed_process(runtime, "runtime")
        _terminate_managed_process(simulator, "DriverSim")
        if not args.no_manage_simulator:
            _kill_simulator_residuals(
                args.simulator_dir,
                reason=restart_reason,
            )
        runtime = None
        simulator = None
        restart_count += 1
        restart_delay = max(0.0, float(args.restart_delay))
        print(
            "[run3][supervisor] restart complete stack "
            f"reason={restart_reason} count={restart_count} "
            f"delay={restart_delay:.1f}s",
            flush=True,
        )
        if restart_delay > 0.0:
            time.sleep(restart_delay)


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
        if not args.runtime_child:
            if not args.no_run_log:
                run_log_archive = _install_run_logging(args)
        _log_source_identity()
        if not args.runtime_child:
            supervisor_lock = acquire_supervisor_lock()
            if supervisor_lock is False:
                raise SystemExit(2)
            raise SystemExit(supervise_runtime(args))
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
