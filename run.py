import argparse
import atexit
import base64
import json
import math
import os
import re
import sys
import time
import threading
from collections import deque


class _TeeStream:
    def __init__(self, original, files, lock):
        self.original = original
        self.files = files
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.original.write(text)
            for file_obj in self.files:
                file_obj.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.original.flush()
            for file_obj in self.files:
                file_obj.flush()

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()

    def __getattr__(self, name):
        return getattr(self.original, name)


class _TerminalLog:
    def __init__(self, output_dir):
        output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(
            output_dir, f"run_{timestamp}_{os.getpid()}.txt"
        )
        latest_path = os.path.join(output_dir, "latest.txt")
        self.archive_path = archive_path
        self.latest_path = latest_path
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._files = [
            open(archive_path, "w", encoding="utf-8", buffering=1),
            open(latest_path, "w", encoding="utf-8", buffering=1),
        ]
        self._lock = threading.RLock()
        self._closed = False
        sys.stdout = _TeeStream(self._original_stdout, self._files, self._lock)
        sys.stderr = _TeeStream(self._original_stderr, self._files, self._lock)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            for file_obj in self._files:
                try:
                    file_obj.close()
                except Exception:
                    pass


def _setup_terminal_log(output_dir):
    terminal_log = _TerminalLog(output_dir)
    atexit.register(terminal_log.close)
    print(f"[terminal-log] archive={terminal_log.archive_path}")
    print(f"[terminal-log] latest={terminal_log.latest_path}")
    return terminal_log


def _setup_project_paths():
    current = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    while True:
        candidates.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for base in candidates:
        if (
            os.path.isdir(os.path.join(base, "modules"))
            and os.path.isdir(os.path.join(base, "chassis"))
            and os.path.isdir(os.path.join(base, "main"))
        ):
            if base not in sys.path:
                sys.path.insert(0, base)
            return
        e2e_base = os.path.join(base, "e2e")
        if os.path.isdir(os.path.join(e2e_base, "modules")):
            if e2e_base not in sys.path:
                sys.path.insert(0, e2e_base)
            return


_setup_project_paths()


def _setup_global_route_path():
    sample_dir = os.path.abspath(os.path.dirname(__file__))
    current = sample_dir
    while True:
        candidate = os.path.join(current, "ros2_map", "src", "gloplan")
        if os.path.isfile(
            os.path.join(candidate, "gloplan", "global_route_planner.py")
        ):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise ImportError("cannot locate ros2_map/src/gloplan pure-Python planner")


_setup_global_route_path()

from opendrive_spiral_compat import install_spiral_support

install_spiral_support()

import cv2
import numpy as np

# Global routing is an in-process Python call.  These names stay defined only
# so old helper functions fail closed if called by an external integration;
# run.py never imports or initialises ROS2.
rclpy = None
Pose = None
PoseArray = None
Path = None
HistoryPolicy = None
QoSProfile = None
ReliabilityPolicy = None
String = None

import libMulticastNetwork

from chassis.proto.chassis_enums_pb2 import VEHICLE_FEEDBACK, VEHICLE_CONTROL
from chassis.proto.chassis_messages_pb2 import VehicleFeedback, VehicleControl
from main.proto.messages_pb2 import PubRole, Notify, ActorPrepare, ActorPrepareResult
from main.proto.enums_pb2 import (
    MT_PUBROLE,
    MT_NOTIFY,
    NT_ABORT_TEST,
    NT_START_TEST,
    NT_FINISH_TEST,
    NT_DESTROY_ROLE,
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
    NT_COLLIDE_ROLE
)
from get_ip import get_ip_address
from predictor import Predictor
from drive_trace_logger import DriveTraceLogger
from gt_perception import GroundTruthObstacleAdapter
from rule_based_planner import (
    PlannerConfig,
    RuleBasedPlanner,
    StableController,
    build_direct_sprint_route,
    session_startup_speed,
)
from roundabout_controller import RoundaboutController
from speed_limits import initial_state_speed_mps, resolve_expected_speed
from control_publisher import FinalSpeedLimiter
from npc_truth import (
    decode_npc_payload,
    ensure_npc_truth_timestamp,
)
from gloplan.global_route_planner import DirectGlobalRoutePlanner, RoutePlanningError

# The runtime compatibility patch changes parsed road geometry.  Invalidate
# routes cached by the legacy parser even when the external ros2_map package
# itself has not yet been updated.
DirectGlobalRoutePlanner.PLANNER_CACHE_VERSION = (
    "direct-global-route-v2-spiral"
)

try:
    from google.protobuf.json_format import MessageToDict
except ImportError:  # pragma: no cover - protobuf is provided by DriveSim
    MessageToDict = None

DEBUG_SYNC = os.environ.get("E2E_DEBUG_SYNC", "0") == "1"
DEBUG_DRIVE = os.environ.get("E2E_DEBUG_DRIVE", "0") == "1"
PERCEPTION_SOURCE = os.environ.get(
    "E2E_PERCEPTION_SOURCE", "gt"
).strip().lower()
NPC_TRUTH_ENABLED = (
    os.environ.get("E2E_NPC_TRUTH_ENABLED", "1") == "1"
)
GT_STARTUP_GRACE_SECONDS = max(
    0.0,
    float(os.environ.get("E2E_GT_STARTUP_GRACE_SECONDS", "0.50")),
)
EXPECTED_SPEED_CLI_MPS = None
USE_XODR_EXPECTED_SPEED = False
current_expected_speed = None
episode_initial_speed = None
ALGORITHM_POLICY_VERSION = (
    "2026-07-27-episode-speed-reset-v27"
)
MULTICAST_CLIENT_NAME = "apollo_testee"
CONTROL_LOOP_PERIOD = max(0.005, float(os.environ.get("RULE_CONTROL_PERIOD", "0.02")))
PREPARE_RESULT_REPEAT = max(
    1, int(os.environ.get("E2E_PREPARE_RESULT_REPEAT", "1"))
)
PREPARE_RESULT_INTERVAL = max(
    0.0, float(os.environ.get("E2E_PREPARE_RESULT_INTERVAL", "0.03"))
)
PREPARE_RESPONSE_DELAY = max(
    0.0, float(os.environ.get("E2E_PREPARE_RESPONSE_DELAY", "0.0"))
)
PREPARE_RESULT_RESEND_INTERVAL = max(
    0.0,
    float(os.environ.get("E2E_PREPARE_RESULT_RESEND_INTERVAL", "0.0")),
)
loop_count = 0
last_ins_sequence = None
last_ins_position = None
duplicate_ins_count = 0
last_duplicate_ins_warn_ts = 0.0
ins_monotonic_sequence_enabled = (
    os.environ.get("E2E_INS_MONOTONIC_SEQUENCE", "0") == "1"
)
nonmonotonic_ins_count = 0
last_nonmonotonic_ins_warn_ts = 0.0
invalid_ins_count = 0
last_invalid_ins_warn_ts = 0.0
map_loading = False
map_ready = False
prepare_received_ts = None
map_ready_ts = None
prepare_sent_ts = None
prepare_not_before_ts = None
first_ins_ready = False
first_ins_ready_ts = None
first_pointcloud_ready = False
first_pointcloud_ready_ts = None
first_control_sent = False
first_control_sent_ts = None
ros_plan_node = None
ros_plan_pub = None
ros_global_plan_sub = None
ros_episode_event_pub = None
local_abort_pending = False
local_abort_sent_ts = 0.0
plan_start_check_remaining = 0
plan_start_check_total = int(os.environ.get("E2E_PLAN_START_CHECK_FRAMES", "15"))
plan_start_warn_distance = float(os.environ.get("E2E_PLAN_START_WARN_DISTANCE", "20.0"))
plan_start_check_xy = None
plan_start_check_map = ""
global_plan_ready = False
global_plan_ready_ts = None
global_plan_ready_points = 0
global_plan_ready_frame_id = ""
global_plan_ready_start_dist = float("inf")
global_plan_wait_timeout = float(os.environ.get("E2E_GLOBAL_PLAN_WAIT_TIMEOUT", "20.0"))
global_plan_wait_start_tolerance = float(os.environ.get("E2E_GLOBAL_PLAN_WAIT_START_TOL", "20.0"))
global_plan_settle_delay = max(
    0.0,
    float(os.environ.get("E2E_GLOBAL_PLAN_SETTLE_DELAY", "1.0")),
)
global_plan_expected_frame_id = ""
global_plan_expected_start_xy = None
global_plan_min_stamp_ns = 0
global_plan_data = None
last_global_plan_hold_warn_ts = 0.0
ins_start_gate_enabled = os.environ.get("E2E_INS_START_GATE_ENABLED", "1") == "1"
ins_start_gate_tolerance = float(os.environ.get("E2E_INS_START_GATE_TOL", "50.0"))
ins_start_gate_xy = None
ins_start_gate_map = ""
ins_start_gate_reject_count = 0
last_ins_start_gate_warn_ts = 0.0
ins_receiver_stop_event = threading.Event()
final_control_enabled = (
    os.environ.get("E2E_FINAL_CONTROL_LIMITER", "0") == "1"
)
final_control_interval = max(
    0.005,
    float(os.environ.get("E2E_FINAL_CONTROL_INTERVAL", "0.03")),
)
final_control_safe_accel = max(
    0.0,
    float(os.environ.get("E2E_FINAL_CONTROL_SAFE_ACCEL", "2.0")),
)
final_control_one_shot_target = (
    os.environ.get("E2E_FINAL_CONTROL_ONE_SHOT_TARGET", "0") == "1"
)
final_control_one_shot_effective_interval = max(
    1e-6,
    float(
        os.environ.get(
            "E2E_FINAL_CONTROL_ONE_SHOT_EFFECTIVE_INTERVAL",
            str(final_control_interval),
        )
    ),
)
final_control_limiter = FinalSpeedLimiter(
    safe_accel=final_control_safe_accel,
    publish_interval=final_control_interval,
    one_shot_target=final_control_one_shot_target,
    one_shot_effective_interval=(
        final_control_one_shot_effective_interval
    ),
)
final_control_lock = threading.Lock()
final_control_stop_event = threading.Event()
final_control_active = False
final_control_desired = None
final_control_last_log_ts = 0.0
test_start_received_ts = None
channel_created_ts = None
first_prepare_startup_timeout = max(
    0.0, float(os.environ.get("E2E_FIRST_PREPARE_TIMEOUT", "0.0"))
)
first_ins_startup_timeout = max(
    0.0, float(os.environ.get("E2E_FIRST_INS_TIMEOUT", "0.0"))
)
last_ego_hold_warn_ts = 0.0
last_prepare_wait_warn_ts = 0.0
last_start_wait_warn_ts = 0.0
last_rule_plan = None
roundabout_controller = None
last_rule_plan_wall_time = 0.0
last_control_wall_time = None
last_drive_plan_debug_ts = 0.0
last_drive_control_debug_ts = 0.0
last_drive_feedback_debug_ts = 0.0
last_drive_ins_debug_ts = 0.0
last_drive_lidar_debug_ts = 0.0
last_npc_truth_debug_ts = 0.0
last_npc_web_debug_ts = 0.0
npc_truth_debug_history = {}
drive_trace_logger = None
last_drive_trace_error_ts = 0.0
npc_channel = None
npc_truth_recorder = None
npc_truth_frames = deque(maxlen=64)
gt_obstacle_adapter = GroundTruthObstacleAdapter(
    track_hold_seconds=float(
        os.environ.get("GT_TRACK_HOLD_SECONDS", "1.0")
    ),
    innovation_gate_m=float(
        os.environ.get("GT_TRACK_INNOVATION_GATE_M", "4.0")
    ),
    innovation_gate_speed=float(
        os.environ.get("GT_TRACK_INNOVATION_GATE_SPEED", "12.0")
    ),
)
last_gt_planning_timestamp = None


class _NpcTruthRecorder:
    """Lossless probe for the simulator-provided NPC multicast channel."""

    def __init__(self, output_dir):
        output_dir = os.path.abspath(
            os.path.expanduser(str(output_dir))
        )
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.archive_path = os.path.join(
            output_dir,
            f"npc_truth_{timestamp}_{os.getpid()}.jsonl",
        )
        self.latest_path = os.path.join(
            output_dir, "npc_truth_latest.jsonl"
        )
        self._files = [
            open(
                self.archive_path,
                "w",
                encoding="utf-8",
                buffering=1,
            ),
            open(
                self.latest_path,
                "w",
                encoding="utf-8",
                buffering=1,
            ),
        ]
        self.count = 0
        self.last_status_wall_time = 0.0
        self.closed = False
        descriptor_fields = []
        try:
            for field in PubRole.DESCRIPTOR.fields:
                descriptor_fields.append(
                    {
                        "name": field.name,
                        "number": field.number,
                        "type": field.type,
                        "label": field.label,
                        "message_type": (
                            field.message_type.full_name
                            if field.message_type is not None
                            else None
                        ),
                    }
                )
        except Exception:
            descriptor_fields = []
        self._write(
            {
                "record_type": "schema",
                "schema_version": 1,
                "wall_time": time.time(),
                "channel": "npc",
                "candidate_parser": "PubRole",
                "pubrole_fields": descriptor_fields,
                "description": (
                    "Raw NPC multicast messages plus best-effort "
                    "PubRole reflection decoding"
                ),
            }
        )
        print(f"[npc-truth] archive={self.archive_path}")
        print(f"[npc-truth] latest={self.latest_path}")
        print(
            "[npc-truth] PubRole fields="
            + (
                ",".join(
                    item["name"] for item in descriptor_fields
                )
                or "<none>"
            )
        )

    def _write(self, record):
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for file_obj in self._files:
            file_obj.write(line + "\n")

    def record(self, message_type, payload):
        if self.closed:
            return
        payload = bytes(payload)
        parsed = None
        parse_error = None
        npc_truth = None
        npc_truth_error = None
        try:
            npc_truth = decode_npc_payload(payload)
        except Exception as exc:
            npc_truth_error = (
                f"{type(exc).__name__}: {exc}"
            )
        try:
            candidate = PubRole()
            candidate.ParseFromString(payload)
            if MessageToDict is not None:
                parsed = MessageToDict(
                    candidate,
                    preserving_proto_field_name=True,
                )
            else:
                parsed = {
                    "protobuf_text": " ".join(
                        str(candidate).split()
                    )
                }
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        ego = getattr(globals().get("model"), "ego", None)
        ego_record = None
        if ego is not None:
            ego_record = {
                "x": getattr(ego, "x", None),
                "y": getattr(ego, "y", None),
                "theta": getattr(ego, "theta", None),
                "speed": getattr(ego, "speed", None),
                "length": getattr(ego, "length", None),
                "width": getattr(ego, "width", None),
                "ins_sequence": globals().get(
                    "last_ins_sequence"
                ),
            }
        current_model = globals().get("model")
        lidar_timestamp = (
            getattr(
                current_model,
                "latest_obstacle_measurement_stamp_s",
                None,
            )
            if current_model is not None
            else None
        )
        self.count += 1
        self._write(
            {
                "record_type": "npc_message",
                "schema_version": 1,
                "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "session_id": globals().get("session_id", ""),
                "map_name": globals().get("map_name"),
                "message_index": self.count,
                "message_type": int(message_type),
                "payload_size": len(payload),
                "lidar_timestamp": lidar_timestamp,
                "ego": ego_record,
                "pubrole": parsed,
                "parse_error": parse_error,
                "npc_truth": npc_truth,
                "npc_truth_error": npc_truth_error,
                # Preserve the source bytes so a different protobuf can be
                # applied later without rerunning the scenario.
                "raw_base64": base64.b64encode(payload).decode(
                    "ascii"
                ),
            }
        )
        now = time.time()
        if (
            self.count <= 3
            or now - self.last_status_wall_time >= 2.0
        ):
            self.last_status_wall_time = now
            print(
                "[npc-truth] "
                f"messages={self.count} "
                f"type={int(message_type)} "
                f"bytes={len(payload)} "
                f"roles={npc_truth.get('role_count') if npc_truth else '-'} "
                f"npc_decoded={bool(npc_truth is not None)} "
                f"pubrole_decoded={bool(parsed)} "
                f"error={parse_error or '-'}"
            )
        return npc_truth

    def close(self):
        if self.closed:
            return
        self.closed = True
        for file_obj in self._files:
            try:
                file_obj.close()
            except Exception:
                pass


def _nearest_npc_truth(lidar_timestamp, max_delta_s=0.30):
    if not npc_truth_frames:
        return None, None
    try:
        lidar_timestamp = float(lidar_timestamp)
    except (TypeError, ValueError):
        return npc_truth_frames[-1], None
    nearest = min(
        npc_truth_frames,
        key=lambda truth: abs(
            float(truth["timestamp_s"]) - lidar_timestamp
        ),
    )
    delta_s = float(nearest["timestamp_s"]) - lidar_timestamp
    if abs(delta_s) > max(0.0, float(max_delta_s)):
        return None, delta_s
    return nearest, delta_s


def _print_npc_truth_debug(
    npc_truth,
    message_type,
    payload_size,
):
    """Print decoded NPC state without affecting the control data path."""
    global last_npc_truth_debug_ts
    if not DEBUG_DRIVE or not isinstance(npc_truth, dict):
        return
    now = time.monotonic()
    if now - last_npc_truth_debug_ts < 0.5:
        return
    last_npc_truth_debug_ts = now
    current_model = globals().get("model")
    ego = getattr(current_model, "ego", None)
    ego_speed = float(getattr(ego, "speed", 0.0) or 0.0)
    ego_lateral_speed = float(
        getattr(current_model, "ego_ros_vy", 0.0) or 0.0
    )
    lidar_timestamp = getattr(
        current_model,
        "latest_obstacle_measurement_stamp_s",
        None,
    )
    npc_timestamp = npc_truth.get("timestamp_s")
    try:
        timestamp_delta_ms = (
            float(npc_timestamp) - float(lidar_timestamp)
        ) * 1000.0
        timestamp_text = f"{timestamp_delta_ms:+.1f}ms"
    except (TypeError, ValueError):
        timestamp_text = "-"
    roles = npc_truth.get("roles", [])
    print(
        "[npc-truth][parsed] "
        f"type={int(message_type)} bytes={int(payload_size)} "
        f"decoder={npc_truth.get('decoder', '-')} "
        f"roles={len(roles)} npc_ts={npc_timestamp} "
        f"last_lidar_ts={lidar_timestamp} "
        f"npc_minus_lidar={timestamp_text} "
        f"ego_v=({ego_speed:.2f},{ego_lateral_speed:.2f})m/s"
    )
    for role in roles:
        position = role.get("position", {})
        velocity = role.get("vector_raw", {})
        dimensions = role.get("dimensions", {})
        try:
            x = float(position.get("x", 0.0))
            y = float(position.get("y", 0.0))
            z = float(position.get("z", 0.0))
            vx = float(velocity.get("x", 0.0))
            vy = float(velocity.get("y", 0.0))
            vz = float(velocity.get("z", 0.0))
            distance = math.hypot(x, y)
            npc_speed = math.hypot(vx, vy)
            relative_vx = vx - ego_speed
            relative_vy = vy - ego_lateral_speed
            relative_speed = math.hypot(
                relative_vx, relative_vy
            )
            closing_speed = (
                -(
                    x * relative_vx
                    + y * relative_vy
                )
                / max(distance, 1e-6)
            )
            previous = npc_truth_debug_history.get(
                str(role.get("role_name", "?"))
            )
            relative_fd_text = "-"
            if previous is not None and npc_timestamp is not None:
                dt = float(npc_timestamp) - previous[0]
                if 0.02 <= dt <= 2.0:
                    fd_vx = (x - previous[1]) / dt
                    fd_vy = (y - previous[2]) / dt
                    fd_closing = -(
                        x * fd_vx + y * fd_vy
                    ) / max(distance, 1e-6)
                    relative_fd_text = (
                        f"({fd_vx:+.2f},{fd_vy:+.2f})m/s "
                        f"closing_fd={fd_closing:+.2f}m/s"
                    )
            if npc_timestamp is not None:
                npc_truth_debug_history[
                    str(role.get("role_name", "?"))
                ] = (float(npc_timestamp), x, y)
            yaw_deg = math.degrees(float(role.get("yaw", 0.0)))
            length = float(dimensions.get("length", 0.0))
            width = float(dimensions.get("width", 0.0))
            height = float(dimensions.get("height", 0.0))
        except (TypeError, ValueError):
            print(
                "[npc-truth][role][WARN] malformed role="
                f"{role.get('role_name', '?')}"
            )
            continue
        print(
            "[npc-truth][role] "
            f"name={role.get('role_name', '?')} "
            f"model={role.get('model_name', '?')} "
            f"type={role.get('class_name', 'Unknown')} "
            f"rel_pos=({x:+.2f},{y:+.2f},{z:+.2f})m "
            f"distance={distance:.2f}m "
            f"npc_v_raw=({vx:+.2f},{vy:+.2f},{vz:+.2f})m/s "
            f"npc_speed={npc_speed:.2f}m/s "
            f"rel_v_est=({relative_vx:+.2f},"
            f"{relative_vy:+.2f})m/s "
            f"rel_speed_est={relative_speed:.2f}m/s "
            f"closing_est={closing_speed:+.2f}m/s "
            f"rel_v_fd={relative_fd_text} "
            f"yaw={yaw_deg:+.1f}deg "
            f"size=({length:.2f},{width:.2f},{height:.2f})m "
            f"size_valid={bool(role.get('dimensions_valid', False))}"
        )


def process_npc_truth(max_messages=128):
    """Decode NPC truth for visualization; optionally record raw frames."""
    if not NPC_TRUTH_ENABLED:
        return 0
    if npc_channel is None:
        return 0
    received = 0
    latest_decoded = None
    latest_message_type = None
    latest_payload_size = 0
    for _ in range(max(1, int(max_messages))):
        try:
            ret, msg = npc_channel.get()
            if msg is None:
                break
            if ret < 0:
                continue
            payload = libMulticastNetwork.getMessageData(msg)
            if npc_truth_recorder is not None:
                npc_truth = npc_truth_recorder.record(
                    msg.type(), payload
                )
            else:
                npc_truth = decode_npc_payload(payload)
            if npc_truth is not None:
                received_monotonic = time.monotonic()
                previous_truth = (
                    npc_truth_frames[-1]
                    if npc_truth_frames
                    else None
                )
                ensure_npc_truth_timestamp(
                    npc_truth,
                    previous_truth=previous_truth,
                    received_monotonic=received_monotonic,
                    wall_timestamp=time.time(),
                )
                npc_truth["_received_monotonic"] = received_monotonic
                current_ego = getattr(
                    globals().get("model"), "ego", None
                )
                if current_ego is not None:
                    try:
                        npc_truth["_ego_snapshot"] = {
                            "x": float(current_ego.x),
                            "y": float(current_ego.y),
                            "theta": float(current_ego.theta),
                        }
                    except (
                        TypeError,
                        ValueError,
                        AttributeError,
                    ):
                        pass
                npc_truth_frames.append(npc_truth)
                latest_decoded = npc_truth
                latest_message_type = msg.type()
                latest_payload_size = len(payload)
            received += 1
        except Exception as exc:
            print(f"[npc-truth][WARN] receive failed: {exc}")
            break
    if latest_decoded is not None:
        _print_npc_truth_debug(
            latest_decoded,
            latest_message_type,
            latest_payload_size,
        )
    return received


def _record_drive_trace(**kwargs):
    """Never allow diagnostics failure to interrupt vehicle control."""
    global last_drive_trace_error_ts
    if drive_trace_logger is None:
        return False
    try:
        return drive_trace_logger.record_cycle(**kwargs)
    except Exception as exc:
        now = time.monotonic()
        if now - last_drive_trace_error_ts >= 5.0:
            last_drive_trace_error_ts = now
            print(f"[drive-trace][WARN] record failed: {exc}")
        return False


def _sample_path_for_trace(path, max_points=200):
    if not isinstance(path, dict):
        return None
    xs = path.get("x")
    ys = path.get("y")
    try:
        count = min(len(xs), len(ys))
    except Exception:
        return None
    if count <= 0:
        return None
    sample_count = min(count, max(2, int(max_points)))
    if sample_count == count:
        indices = list(range(count))
    else:
        step = (count - 1) / float(sample_count - 1)
        indices = sorted(
            {min(count - 1, int(round(index * step)))
             for index in range(sample_count)}
        )
    result = {
        "original_count": count,
        "sample_count": len(indices),
        "frame_id": path.get("frame_id"),
    }
    for name in ("x", "y", "yaw", "kappa", "s", "speed_limit"):
        values = path.get(name)
        if values is None:
            continue
        try:
            result[name] = [float(values[index]) for index in indices]
        except Exception:
            pass
    return result


def _xodr_frame_id(map_id):
    map_id = str(map_id or "").strip()
    if not map_id:
        raise ValueError("map id can not be empty")
    if not map_id.lower().endswith(".xodr"):
        map_id = f"{map_id}.xodr"
    return map_id


def _normalize_frame_id(frame_id):
    frame_id = str(frame_id or "").strip().replace("\\", "/")
    frame_id = frame_id.rsplit("/", 1)[-1]
    if frame_id and not frame_id.lower().endswith(".xodr"):
        frame_id = f"{frame_id}.xodr"
    return frame_id.lower()


def _same_xodr_frame(left, right):
    return _normalize_frame_id(left) == _normalize_frame_id(right)


def _is_aitown_map(map_id):
    # print('_normalize_frame_id(map_id)',_normalize_frame_id(map_id))
    return _normalize_frame_id(map_id) == "aitownreconstructed_v0103_200518.xodr"

def _state_to_pose(state):
    pose = Pose()
    yaw = np.deg2rad(float(state.get("orientation_z", 0.0)))
    pose.position.x = float(state["x"])
    pose.position.y = float(state["y"])
    pose.position.z = float(state.get("z", 0.0))
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(yaw * 0.5)
    pose.orientation.w = math.cos(yaw * 0.5)
    return pose

def _valid_imu_xy(x, y):
    try:
        x = float(x)
        y = float(y)
    except Exception:
        return False

    if not math.isfinite(x) or not math.isfinite(y):
        return False

    # Map coordinates are not guaranteed to use a local +/-1000 m frame.
    # MT_21, for example, legitimately starts near (786, 1569). Stale
    # previous-session samples are already rejected by the init-state gate,
    # so this check should only reject obviously corrupt magnitudes.
    if abs(x) > 10000000.0 or abs(y) > 10000000.0:
        return False

    return True


def _ins_xy(ins):
    pos = getattr(ins, "position", None)
    if pos is None:
        return None
    try:
        return (float(pos.x), float(pos.y))
    except Exception:
        return None


def _warn_duplicate_ins(sequence, xy):
    global last_duplicate_ins_warn_ts

    now = time.time()
    if now - last_duplicate_ins_warn_ts < 1.0:
        return
    last_duplicate_ins_warn_ts = now

    if xy is None:
        pos_text = "unknown"
    else:
        pos_text = f"({xy[0]:.3f}, {xy[1]:.3f})"
    print(
        "[ins][WARN] duplicate INS frame ignored "
        f"seq={sequence} repeats={duplicate_ins_count} pos={pos_text}; "
        "sequence and position are unchanged, likely stale sample from previous cycle"
    )


def _warn_nonmonotonic_ins(sequence, xy):
    global last_nonmonotonic_ins_warn_ts

    now = time.time()
    if now - last_nonmonotonic_ins_warn_ts < 1.0:
        return
    last_nonmonotonic_ins_warn_ts = now
    pos_text = (
        "unknown"
        if xy is None
        else f"({xy[0]:.3f}, {xy[1]:.3f})"
    )
    print(
        "[ins][WARN] non-monotonic INS sample ignored "
        f"seq={sequence} accepted_high_water={last_ins_sequence} "
        f"count={nonmonotonic_ins_count} pos={pos_text}; "
        "the current episode pose is held"
    )


def should_reject_nonmonotonic_ins(ins_sequence, ins_xy):
    global nonmonotonic_ins_count

    if (
        not ins_monotonic_sequence_enabled
        or last_ins_sequence is None
        or int(ins_sequence) > int(last_ins_sequence)
    ):
        nonmonotonic_ins_count = 0
        return False

    nonmonotonic_ins_count += 1
    _warn_nonmonotonic_ins(ins_sequence, ins_xy)
    return True


def _ins_attr_float(ins, object_name, attr_name):
    obj = getattr(ins, object_name, None)
    if obj is None:
        return None
    try:
        return float(getattr(obj, attr_name))
    except Exception:
        return None


def _ins_sample_status(ins):
    xy = _ins_xy(ins)
    if xy is None:
        return False, "missing position", xy
    if not all(math.isfinite(value) for value in xy):
        return False, "non-finite position", xy
    if not _valid_imu_xy(xy[0], xy[1]):
        return (
            False,
            "position magnitude exceeds 10000000m",
            xy,
        )

    heading = getattr(ins, "heading", 0.0)
    try:
        heading = float(heading)
    except Exception:
        return False, "invalid heading", xy
    if not math.isfinite(heading):
        return False, "invalid heading", xy

    return True, "", xy


def _warn_invalid_ins(sequence, xy, reason):
    global last_invalid_ins_warn_ts

    now = time.time()
    if now - last_invalid_ins_warn_ts < 1.0:
        return
    last_invalid_ins_warn_ts = now

    if xy is None:
        pos_text = "unknown"
    else:
        pos_text = f"({xy[0]:.6g}, {xy[1]:.6g})"
    print(
        "[ins][WARN] invalid INS sample ignored "
        f"seq={sequence} count={invalid_ins_count} pos={pos_text} reason={reason}"
    )


def start_ins_start_gate(start_x, start_y, map_id):
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts

    ins_start_gate_xy = (float(start_x), float(start_y))
    ins_start_gate_map = str(map_id or "")
    ins_start_gate_reject_count = 0
    last_ins_start_gate_warn_ts = 0.0


def reset_ins_start_gate():
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts

    ins_start_gate_xy = None
    ins_start_gate_map = ""
    ins_start_gate_reject_count = 0
    last_ins_start_gate_warn_ts = 0.0


def _warn_stale_start_ins(sequence, xy, dist):
    global last_ins_start_gate_warn_ts

    now = time.time()
    if now - last_ins_start_gate_warn_ts < 1.0:
        return
    last_ins_start_gate_warn_ts = now

    start_text = "unknown"
    if ins_start_gate_xy is not None:
        start_text = f"({ins_start_gate_xy[0]:.3f}, {ins_start_gate_xy[1]:.3f})"
    pos_text = "unknown" if xy is None else f"({xy[0]:.3f}, {xy[1]:.3f})"
    print(
        "[ins][WARN] stale pre-start INS sample ignored "
        f"seq={sequence} count={ins_start_gate_reject_count} "
        f"pos={pos_text} start={start_text} dist={dist:.3f}m "
        f"threshold={ins_start_gate_tolerance:.3f}m map={ins_start_gate_map}; "
        "waiting for first INS near current episode init_state"
    )


def should_reject_pre_first_ins(ins_sequence, ins_xy):
    global ins_start_gate_reject_count

    if not ins_start_gate_enabled or first_ins_ready:
        return False
    if ins_start_gate_xy is None or ins_xy is None:
        return False

    start_x, start_y = ins_start_gate_xy
    dist = math.hypot(ins_xy[0] - start_x, ins_xy[1] - start_y)
    if dist <= ins_start_gate_tolerance:
        if ins_start_gate_reject_count > 0:
            print(
                "[ins] first current-episode INS accepted after stale samples "
                f"rejected={ins_start_gate_reject_count} seq={ins_sequence} "
                f"dist_to_start={dist:.3f}m"
            )
        return False

    ins_start_gate_reject_count += 1
    _warn_stale_start_ins(ins_sequence, ins_xy, dist)
    return True


def start_plan_start_imu_check(start_x, start_y, map_id):
    global plan_start_check_remaining
    global plan_start_check_xy
    global plan_start_check_map

    plan_start_check_remaining = max(0, int(plan_start_check_total))
    plan_start_check_xy = (float(start_x), float(start_y))
    plan_start_check_map = str(map_id or "")


def reset_global_plan_wait(map_id, start_x, start_y):
    global global_plan_ready
    global global_plan_ready_ts
    global global_plan_ready_points
    global global_plan_ready_frame_id
    global global_plan_ready_start_dist
    global global_plan_expected_frame_id
    global global_plan_expected_start_xy
    global global_plan_min_stamp_ns
    global global_plan_data
    global last_global_plan_hold_warn_ts

    global_plan_ready = False
    global_plan_ready_ts = None
    global_plan_ready_points = 0
    global_plan_ready_frame_id = ""
    global_plan_ready_start_dist = float("inf")
    global_plan_expected_frame_id = _xodr_frame_id(map_id)
    global_plan_expected_start_xy = (float(start_x), float(start_y))
    global_plan_min_stamp_ns = 0
    global_plan_data = None
    last_global_plan_hold_warn_ts = 0.0


def _on_global_plan_ready_msg(msg):
    global global_plan_ready
    global global_plan_ready_ts
    global global_plan_ready_points
    global global_plan_ready_frame_id
    global global_plan_ready_start_dist
    global global_plan_data

    poses = list(getattr(msg, "poses", []) or [])
    if len(poses) < 2 or global_plan_expected_start_xy is None:
        return

    frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    if global_plan_expected_frame_id and not _same_xodr_frame(frame_id, global_plan_expected_frame_id):
        return

    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    try:
        msg_stamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
    except Exception:
        msg_stamp_ns = 0
    if global_plan_min_stamp_ns > 0 and msg_stamp_ns > 0 and msg_stamp_ns < global_plan_min_stamp_ns:
        print(
            "[global-plan-ready] ignore stale /global_plan "
            f"frame={frame_id} stamp_ns={msg_stamp_ns} "
            f"min_stamp_ns={global_plan_min_stamp_ns}"
        )
        return

    first_pose = getattr(poses[0], "pose", None)
    first_pos = getattr(first_pose, "position", None)
    if first_pos is None:
        return

    try:
        first_x = float(first_pos.x)
        first_y = float(first_pos.y)
    except Exception:
        return
    if not math.isfinite(first_x) or not math.isfinite(first_y):
        return

    start_x, start_y = global_plan_expected_start_xy
    start_dist = math.hypot(first_x - start_x, first_y - start_y)
    if start_dist > global_plan_wait_start_tolerance:
        print(
            "[global-plan-ready] ignore /global_plan with mismatched start "
            f"frame={frame_id} points={len(poses)} "
            f"first=({first_x:.3f}, {first_y:.3f}) "
            f"expected=({start_x:.3f}, {start_y:.3f}) "
            f"dist={start_dist:.3f}"
        )
        return

    global_plan_ready = True
    global_plan_ready_ts = time.time()
    global_plan_ready_points = len(poses)
    global_plan_ready_frame_id = frame_id
    global_plan_ready_start_dist = start_dist
    path_x = []
    path_y = []
    path_speed_limit = []
    for pose_stamped in poses:
        position = getattr(getattr(pose_stamped, "pose", None), "position", None)
        if position is None:
            continue
        try:
            x = float(position.x)
            y = float(position.y)
            speed_limit = float(position.z)
        except Exception:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if path_x and math.hypot(x - path_x[-1], y - path_y[-1]) < 1e-4:
            if math.isfinite(speed_limit) and speed_limit > 0.0:
                path_speed_limit[-1] = speed_limit
            continue
        path_x.append(x)
        path_y.append(y)
        path_speed_limit.append(speed_limit if math.isfinite(speed_limit) else 0.0)
    if len(path_x) >= 2:
        global_plan_data = {
            "x": path_x,
            "y": path_y,
            "speed_limit": path_speed_limit,
            "frame_id": frame_id,
            "stamp": msg_stamp_ns if msg_stamp_ns > 0 else time.time(),
        }


def _ensure_global_plan_ready_subscription():
    global ros_plan_node
    global ros_global_plan_sub

    if rclpy is None or Path is None:
        print("[global-plan-ready] ROS2 nav_msgs/Path is not available; skip wait")
        return None

    node, _ = _ensure_global_plan_publisher()
    if node is None:
        return None

    if ros_global_plan_sub is None:
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ros_global_plan_sub = node.create_subscription(
            Path,
            "/global_plan",
            _on_global_plan_ready_msg,
            qos,
        )

    return node


def wait_for_global_plan_ready(timeout_sec=None):
    timeout = global_plan_wait_timeout if timeout_sec is None else float(timeout_sec)
    node = _ensure_global_plan_ready_subscription()
    if node is None:
        return True

    deadline = time.time() + max(0.0, timeout)
    while rclpy.ok() and time.time() < deadline:
        if global_plan_ready:
            print(
                "[global-plan-ready] received /global_plan "
                f"frame={global_plan_ready_frame_id} "
                f"points={global_plan_ready_points} "
                f"start_dist={global_plan_ready_start_dist:.3f} "
                f"wait={time.time() - (prepare_received_ts or time.time()):.3f}s"
            )
            return True
        rclpy.spin_once(node, timeout_sec=0.05)

    print(
        "[global-plan-ready][WARN] timeout waiting for /global_plan; "
        f"timeout={timeout:.1f}s frame={global_plan_expected_frame_id} "
        f"start={global_plan_expected_start_xy}"
    )
    return False


def hold_until_global_plan_ready():
    global last_global_plan_hold_warn_ts

    if global_plan_ready and global_plan_data is not None:
        return False

    now = time.time()
    if now - last_global_plan_hold_warn_ts > 1.0:
        last_global_plan_hold_warn_ts = now
        hold_speed = startup_hold_speed()
        print(
            "[direct-global-plan] no valid route after start; "
            f"preserve startup speed={hold_speed:.3f}m/s"
        )
    send_control_cmd(0.0, startup_hold_speed(), 0.0)
    return True


def startup_hold_speed():
    """Use current INS speed, or ActorPrepare's initial vehicle speed."""
    return session_startup_speed(
        first_ins_ready,
        getattr(globals().get("model"), "ego", None),
        episode_initial_speed,
    )


def hold_until_ego_ready():
    global last_ego_hold_warn_ts

    if first_ins_ready and model.ego is not None:
        return False

    now = time.time()
    start_age = (
        0.0
        if test_start_received_ts is None
        else max(0.0, now - test_start_received_ts)
    )
    if (
        first_ins_startup_timeout > 0.0
        and test_start_received_ts is not None
        and start_age >= first_ins_startup_timeout
    ):
        print(
            "[startup][RECONNECT] no valid INS arrived after "
            f"{start_age:.1f}s invalid_count={invalid_ins_count}; "
            "requesting one clean process-level channel reconnect",
            flush=True,
        )
        # This path runs only inside run_ct.py's supervised runtime child.
        # A normal SystemExit executes model.close(), whose HTTP visualizer
        # shutdown can deadlock after a failed native-channel startup and
        # prevent the parent from ever seeing the reconnect exit code.
        # Process-level disposal is intentional here: the OS releases every
        # native channel/socket before the supervisor creates a fresh child.
        os._exit(75)
    if now - last_ego_hold_warn_ts >= 1.0:
        last_ego_hold_warn_ts = now
        hold_speed = startup_hold_speed()
        print(
            "[startup][WAIT] waiting for first valid INS "
            f"start_age={start_age:.1f}s; "
            f"preserve_speed={hold_speed:.3f}m/s; "
            "single subscription remains active"
        )
    send_control_cmd(0.0, startup_hold_speed(), 0.0)
    return True


def reconnect_if_prepare_subscription_is_dead():
    if (
        first_prepare_startup_timeout <= 0.0
        or channel_created_ts is None
        or recv_prepare
    ):
        return
    channel_age = max(0.0, time.time() - channel_created_ts)
    if channel_age < first_prepare_startup_timeout:
        return
    print(
        "[startup][RECONNECT] ActorPrepare not received after "
        f"{channel_age:.1f}s from native channel creation; "
        "requesting one clean process-level channel reconnect",
        flush=True,
    )
    os._exit(75)



def check_plan_start_imu(ego, ins_sequence):
    global plan_start_check_remaining

    if plan_start_check_remaining <= 0 or plan_start_check_xy is None or ego is None:
        return

    try:
        ego_x = float(ego.x)
        ego_y = float(ego.y)
    except Exception:
        return
    if not _valid_imu_xy(ego_x, ego_y):
        return

    frame_index = max(1, int(plan_start_check_total) - plan_start_check_remaining + 1)
    plan_start_check_remaining -= 1

    start_x, start_y = plan_start_check_xy
    dist = math.hypot(ego_x - start_x, ego_y - start_y)
    if dist > plan_start_warn_distance:
        print(
            "[global-plan-request][WARN] imu/start mismatch "
            f"frame={frame_index}/{plan_start_check_total} "
            f"dist={dist:.3f}m threshold={plan_start_warn_distance:.3f}m "
            f"start=({start_x:.3f}, {start_y:.3f}) "
            f"imu=({ego_x:.3f}, {ego_y:.3f}) "
            f"map={plan_start_check_map} ins_seq={ins_sequence}"
        )


def _ensure_global_plan_publisher():
    global ros_plan_node
    global ros_plan_pub

    if rclpy is None:
        print("[global-plan-request] ROS2 python packages are not available; skip publish")
        return None, None

    if not rclpy.ok():
        rclpy.init(args=None)

    if ros_plan_node is None:
        bridge = getattr(globals().get("model", None), "ros_bridge", None)
        if bridge is not None and getattr(bridge, "enabled", False):
            ros_plan_node = bridge.node
        else:
            ros_plan_node = rclpy.create_node("wutfsd_global_plan_request_sender")

    if ros_plan_pub is None:
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ros_plan_pub = ros_plan_node.create_publisher(
            PoseArray,
            "/global_plan_request",
            qos,
        )

    return ros_plan_node, ros_plan_pub


def _ensure_episode_event_publisher():
    global ros_plan_node
    global ros_episode_event_pub

    if rclpy is None or String is None:
        print("[episode-event] ROS2 python packages are not available; skip publish")
        return None, None

    if not rclpy.ok():
        rclpy.init(args=None)

    if ros_plan_node is None:
        bridge = getattr(globals().get("model", None), "ros_bridge", None)
        if bridge is not None and getattr(bridge, "enabled", False):
            ros_plan_node = bridge.node
        else:
            ros_plan_node = rclpy.create_node("wutfsd_ros_event_sender")

    if ros_episode_event_pub is None:
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ros_episode_event_pub = ros_plan_node.create_publisher(
            String,
            "/wutfsd_episode_event",
            qos,
        )

    return ros_plan_node, ros_episode_event_pub


def publish_episode_event(event, repeat=3, wait_subscriber=0.5):
    try:
        node, publisher = _ensure_episode_event_publisher()
        if node is None or publisher is None:
            return False

        msg = String()
        msg.data = str(event)

        deadline = time.time() + max(0.0, float(wait_subscriber))
        while (
            time.time() < deadline and
            publisher.get_subscription_count() == 0 and
            rclpy.ok()
        ):
            rclpy.spin_once(node, timeout_sec=0.05)

        for _ in range(max(1, int(repeat))):
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)

        print(f"[episode-event] published /wutfsd_episode_event data={msg.data}")
        return True
    except Exception as exc:
        print(f"[episode-event] failed to publish /wutfsd_episode_event: {exc}")
        return False


def publish_global_plan_request_from_brief_data(
    brief_data,
    testee_index=0,
    repeat=1,
    wait_subscriber=2.0,
    wait_global_plan=False,
):
    global global_plan_min_stamp_ns

    try:
        node, publisher = _ensure_global_plan_publisher()
        if node is None or publisher is None:
            return False

        map_id = (
            brief_data.get("zjl_odv_file") or
            brief_data.get("map_name") or
            brief_data.get("map_id")
        )
        testees = brief_data.get("testees") or []
        if not testees:
            raise ValueError("brief_data must contain at least one testee")

        testee = testees[int(testee_index)]
        init_state = testee.get("init_state") or {}
        target_state = testee.get("target_state") or {}

        msg = PoseArray()
        msg.header.frame_id = _xodr_frame_id(map_id)
        msg.poses.append(_state_to_pose(init_state))
        msg.poses.append(_state_to_pose(target_state))
        reset_global_plan_wait(msg.header.frame_id, init_state["x"], init_state["y"])
        _ensure_global_plan_ready_subscription()

        deadline = time.time() + max(0.0, float(wait_subscriber))
        while (
            time.time() < deadline and
            publisher.get_subscription_count() == 0 and
            rclpy.ok()
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        try:
            global_plan_min_stamp_ns = int(node.get_clock().now().nanoseconds)
        except Exception:
            global_plan_min_stamp_ns = 0
        for _ in range(max(1, int(repeat))):
            msg.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        print(
            "[global-plan-request] published /global_plan_request "
            f"map={msg.header.frame_id} "
            f"start=({init_state['x']:.3f}, {init_state['y']:.3f}) "
            f"goal=({target_state['x']:.3f}, {target_state['y']:.3f})"
        )
        start_plan_start_imu_check(init_state["x"], init_state["y"], msg.header.frame_id)
        if wait_global_plan:
            return wait_for_global_plan_ready()
        return True
    except Exception as exc:
        print(f"[global-plan-request] failed to publish /global_plan_request: {exc}")
        return False
    #到这个部分为止，主要是发布全局规划请求，给ros2_map进行路径规划。


def build_global_plan_directly(brief_data, testee_index=0):
    """Synchronously parse XODR and build the route without ROS messages."""
    global global_plan_data
    global global_plan_ready
    global global_plan_ready_ts
    global global_plan_ready_points
    global global_plan_ready_frame_id
    global global_plan_ready_start_dist

    try:
        map_id = (
            brief_data.get("zjl_odv_file")
            or brief_data.get("map_name")
            or brief_data.get("map_id")
        )
        testees = brief_data.get("testees") or []
        if not testees:
            raise RoutePlanningError("brief_data contains no testee")
        testee = testees[int(testee_index)]
        start_state = testee.get("init_state") or {}
        goal_state = testee.get("target_state") or {}

        reset_global_plan_wait(map_id, start_state["x"], start_state["y"])
        if rule_planner.config.sprint_applies_to(map_id):
            route = build_direct_sprint_route(
                start_state=start_state,
                goal_state=goal_state,
                frame_id=_xodr_frame_id(map_id),
                speed_limit=rule_planner.config.sprint_speed,
            )
            print(
                "[direct-global-plan][SPRINT-DIRECT] "
                "skip XODR lane graph and A*; "
                f"map={_xodr_frame_id(map_id)} "
                f"seed_points={len(route['x'])}"
            )
        else:
            route = global_route_planner.plan(
                start_state=start_state,
                goal_state=goal_state,
                map_ref=map_id,
            )
        first_x = float(route["x"][0])
        first_y = float(route["y"][0])
        start_dist = math.hypot(
            first_x - float(start_state["x"]),
            first_y - float(start_state["y"]),
        )
        global_plan_data = route
        global_plan_ready = True
        global_plan_ready_ts = time.time()
        global_plan_ready_points = len(route["x"])
        global_plan_ready_frame_id = route["frame_id"]
        global_plan_ready_start_dist = start_dist
        print(
            "[direct-global-plan] ready "
            f"map={route['frame_id']} points={global_plan_ready_points} "
            f"start_dist={start_dist:.3f}m "
            f"cache_hit="
            f"{bool((route.get('_persistent_cache') or {}).get('hit', False))}"
        )
        route_x = np.asarray(route.get("x", []), dtype=float)
        route_y = np.asarray(route.get("y", []), dtype=float)
        route_distance = float(
            np.sum(np.hypot(np.diff(route_x), np.diff(route_y)))
        )
        expected_speed_mps = float(
            (current_expected_speed or {}).get(
                "speed_mps",
                rule_planner.config.expected_speed_mps or 0.0,
            )
        )
        expected_duration = (
            route_distance / expected_speed_mps
            if expected_speed_mps > 0.0
            else float("nan")
        )
        print(
            "[expected-duration] "
            f"route_distance={route_distance:.3f}m "
            f"expected_speed={expected_speed_mps:.3f}m/s "
            f"expected_speed_kmh={expected_speed_mps * 3.6:.1f}km/h "
            f"estimated_expected_duration={expected_duration:.3f}s "
            f"source={rule_planner.config.expected_speed_source} "
            "note=score_uses_actual_ego_distance"
        )
        route_cache = route.get("_persistent_cache") or {}
        if route.get("_sprint_direct_fallback"):
            print(
                "[global-plan-vis] skip XODR background render for "
                "sprint-direct route"
            )
        elif route_cache.get("hit") and getattr(
            global_route_planner, "opendrive", None
        ) is None:
            print(
                "[global-plan-vis] skip PNG rerender for persistent "
                f"cache hit; route_json={route_cache.get('route_path')}"
            )
        else:
            save_global_plan_png(route)
        start_plan_start_imu_check(
            start_state["x"], start_state["y"], route["frame_id"]
        )
        return True
    except Exception as exc:
        global_plan_data = None
        global_plan_ready = False
        global_plan_ready_points = 0
        print(f"[direct-global-plan][ERROR] {exc}")
        return False


def save_global_plan_png(route):
    """Save the current episode's direct global route without opening a window."""
    try:
        path_x = np.asarray(route.get("x", []), dtype=float)
        path_y = np.asarray(route.get("y", []), dtype=float)
        if path_x.size < 2 or path_x.size != path_y.size:
            raise ValueError("global route must contain at least two x/y points")

        finite = np.isfinite(path_x) & np.isfinite(path_y)
        path_xy = np.column_stack((path_x[finite], path_y[finite]))
        if path_xy.shape[0] < 2:
            raise ValueError("global route contains fewer than two finite points")

        opendrive = getattr(global_route_planner, "opendrive", None)
        if opendrive is None or not hasattr(opendrive, "drawPathOnMap"):
            raise RuntimeError("OpenDRIVE map renderer is unavailable")

        output_dir = os.environ.get(
            "E2E_GLOBAL_PLAN_PNG_DIR",
            os.path.join(os.path.dirname(__file__), "global_plan_vis"),
        )
        os.makedirs(output_dir, exist_ok=True)

        safe_session = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in str(globals().get("session_id", "") or "unknown")
        )
        map_stem = os.path.splitext(
            os.path.basename(str(route.get("frame_id", "map")))
        )[0]
        safe_map = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in map_stem
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        milliseconds = int(time.time() * 1000) % 1000
        output_path = os.path.abspath(
            os.path.join(
                output_dir,
                f"{timestamp}_{milliseconds:03d}_{safe_session}_{safe_map}.png",
            )
        )
        saved_path = opendrive.drawPathOnMap(
            path_xy,
            output_path=output_path,
            zoom_to_path=True,
            title=f"Global Path - {route.get('frame_id', '')}",
        )
        if not saved_path:
            raise RuntimeError("OpenDRIVE map renderer returned an empty path")
        print(f"[global-plan-vis] saved {saved_path}")
        return saved_path
    except Exception as exc:
        print(f"[global-plan-vis][WARN] failed to save PNG: {exc}")
        return None


def prepare(result=True):
    global prepare_sent_ts
    prepare_sent_ts = time.time()
    print(
        "send prepare result="
        f"{bool(result)} session_id={session_id} actor_id={actor_id}"
    )
    print(
        "[prepare-timing] prepare_result_sent "
        f"wall_time={prepare_sent_ts:.3f} "
        f"prepare_received={prepare_received_ts} "
        f"map_ready={map_ready_ts}"
    )
    send_prepare_result = ActorPrepareResult()
    send_prepare_result.session_id = session_id
    send_prepare_result.actor_id = actor_id
    send_prepare_result.result = bool(result)
    data = send_prepare_result.SerializeToString()
    length = len(data)
    successful_copies = 0
    for copy_index in range(PREPARE_RESULT_REPEAT):
        ret = prepare_channel.put(
            MT_ACTOR_PREPARE_RESULT, length, data
        )
        if ret == 0:
            successful_copies += 1
        else:
            print(
                "send prepare msg error "
                f"copy={copy_index + 1}/{PREPARE_RESULT_REPEAT}"
            )
        if (
            copy_index + 1 < PREPARE_RESULT_REPEAT
            and PREPARE_RESULT_INTERVAL > 0.0
        ):
            time.sleep(PREPARE_RESULT_INTERVAL)
    print(
        "[prepare-timing] prepare_result_copies "
        f"successful={successful_copies}/{PREPARE_RESULT_REPEAT} "
        f"interval={PREPARE_RESULT_INTERVAL:.3f}s"
    )


def publish_first_control_before_prepare():
    """Put one neutral control frame on the wire before PrepareResult."""
    global first_control_sent
    global first_control_sent_ts

    if first_control_sent:
        return True

    control_speed = startup_hold_speed()
    ret = _put_control_cmd(0.0, control_speed, 0.0)
    if ret != 0:
        print(
            "[prepare-timing][WAIT] first control publish failed; "
            "PrepareResult remains inhibited",
            flush=True,
        )
        return False

    first_control_sent = True
    first_control_sent_ts = time.time()
    print(
        "[prepare-timing] first_control_sent "
        f"wall_time={first_control_sent_ts:.3f} "
        "phase=before_prepare_result "
        f"acc=0.000m/s2 speed={control_speed:.3f}m/s steer=0.000deg",
        flush=True,
    )
    return True


def should_send_prepare_result(now=None):
    if prepare_sent_ts is None:
        return True
    if PREPARE_RESULT_RESEND_INTERVAL <= 0.0:
        return False
    if now is None:
        now = time.time()
    return (
        float(now) - prepare_sent_ts
        >= PREPARE_RESULT_RESEND_INTERVAL
    )


def _session_order_key(value):
    """Extract the platform timestamp/sequence suffix for session ordering."""
    match = re.search(r"_(\d{14})_(\d+)$", str(value or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def _is_newer_session(candidate, current):
    candidate_key = _session_order_key(candidate)
    current_key = _session_order_key(current)
    return (
        candidate_key is not None
        and current_key is not None
        and candidate_key > current_key
    )


def reset_episode_ego_state(reason):
    """Drop pose, speed, and feedback state that cannot cross sessions."""
    previous_speed = float(
        getattr(getattr(model, "ego", None), "speed", 0.0) or 0.0
    )
    model.ego = None
    model.last_ego = None
    model.ego_ros_speed = 0.0
    model.ego_ros_vy = 0.0
    model.ego_ros_yawrate = 0.0
    model.ego_ros_acc = 0.0
    model.ego_ros_ax = 0.0
    model.ego_ros_ay = 0.0
    model.ego_lon_acc = 0.0
    model.ego_lat_acc = 0.0
    model.ego_ros_heading_deg = None
    model.ego_ros_sequence = None
    model.last_ego_update_wall_time = None
    model.last_ego_update_sequence = None
    model.previous_ego_yaw = None
    model.previous_ego_yaw_wall_time = None
    model.last_ego_speed_pose = None
    model.last_ego_speed_pose_wall_time = 0.0
    model.vehicle_feedback.steering_wheel_angle = 0.0
    model.vehicle_feedback.accelerator_pedal_position = 0.0
    model.vehicle_feedback.brake_pedal_position = 0.0
    print(
        "[episode-reset] ego state cleared "
        f"reason={reason} previous_speed={previous_speed:.3f}m/s"
    )


def get_prepare():
    global recv_prepare
    global start_test
    global session_id
    global actor_id
    global role_id
    global map_name
    global map_loading
    global map_ready
    global prepare_received_ts
    global map_ready_ts
    global prepare_not_before_ts
    global prepare_sent_ts
    global first_ins_ready
    global first_ins_ready_ts
    global first_pointcloud_ready
    global first_pointcloud_ready_ts
    global first_control_sent
    global first_control_sent_ts
    global local_abort_pending
    global local_abort_sent_ts
    global plan_start_check_remaining
    global plan_start_check_xy
    global plan_start_check_map
    global last_ins_sequence
    global last_ins_position
    global duplicate_ins_count
    global last_duplicate_ins_warn_ts
    global nonmonotonic_ins_count
    global last_nonmonotonic_ins_warn_ts
    global invalid_ins_count
    global last_invalid_ins_warn_ts
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts
    global last_rule_plan
    global last_rule_plan_wall_time
    global last_control_wall_time
    global last_drive_plan_debug_ts
    global last_drive_control_debug_ts
    global last_drive_feedback_debug_ts
    global last_drive_ins_debug_ts
    global last_drive_lidar_debug_ts
    global last_gt_planning_timestamp
    global current_expected_speed
    global episode_initial_speed
    global test_start_received_ts

    ret, msg = prepare_channel.get()
    if msg is None:
        return
    msg_type = msg.type()
    if ret < 0 or msg_type != MT_ACTOR_PREPARE:
        return
    if ret >= 0 and msg_type == MT_ACTOR_PREPARE:
        data = libMulticastNetwork.getMessageData(msg)
        prepare_msg = ActorPrepare()
        prepare_msg.ParseFromString(data)
        incoming_session_id = str(prepare_msg.session_id or "").strip()
        current_session_id = str(session_id or "").strip()
        if not incoming_session_id:
            print(
                "[prepare][WARN] ignore ActorPrepare with empty session_id",
                flush=True,
            )
            return
        if incoming_session_id == current_session_id:
            # The daemon can retransmit Prepare while awaiting its result.
            # Never reset an already prepared or running episode.
            return
        if current_session_id and not _is_newer_session(
            incoming_session_id,
            current_session_id,
        ):
            print(
                "[prepare][WARN] ignore stale or unordered ActorPrepare "
                f"received={incoming_session_id} current={current_session_id}",
                flush=True,
            )
            return

        # ActorPrepare is the authoritative episode boundary. Do not let the
        # previous episode's START state leak into the new map.
        start_test = False
        prepare_received_ts = time.time()
        map_loading = True
        map_ready = False
        map_ready_ts = None
        prepare_not_before_ts = None
        prepare_sent_ts = None
        first_ins_ready = False
        first_ins_ready_ts = None
        first_pointcloud_ready = False
        first_pointcloud_ready_ts = None
        first_control_sent = False
        first_control_sent_ts = None
        test_start_received_ts = None
        recv_prepare = True
        session_id = incoming_session_id
        npc_truth_frames.clear()
        npc_truth_debug_history.clear()
        gt_obstacle_adapter.reset()
        last_gt_planning_timestamp = None
        current_visualizer = getattr(
            globals().get("model"),
            "perception_web_visualizer",
            None,
        )
        if current_visualizer is not None:
            current_visualizer.publish_ground_truth(None)
        print(
            "[prepare-timing] prepare_received "
            f"wall_time={prepare_received_ts:.3f} "
            f"session_id={session_id}"
        )
        local_abort_pending = False
        local_abort_sent_ts = 0.0
        plan_start_check_remaining = 0
        plan_start_check_xy = None
        plan_start_check_map = ""
        last_ins_sequence = None
        last_ins_position = None
        duplicate_ins_count = 0
        last_duplicate_ins_warn_ts = 0.0
        nonmonotonic_ins_count = 0
        last_nonmonotonic_ins_warn_ts = 0.0
        invalid_ins_count = 0
        last_invalid_ins_warn_ts = 0.0
        reset_ins_start_gate()
        reset_episode_ego_state("prepare")
        model.invalid_ego_sample_count = 0
        model.last_invalid_ego_sample_warn_wall_time = 0.0

        brief_data = json.loads(prepare_msg.archive_info.brief_data)
        testee_brief = brief_data["testees"][0]
        # The daemon registers this process as
        # ``client1: testee/apollo_testee``. ActorPrepareResult must use that
        # registered client identity; ``testee_D9A9`` is the scenario role,
        # not the multicast client actor id.
        actor_id = MULTICAST_CLIENT_NAME
        print(
            "[prepare-identity] "
            f"client_name={MULTICAST_CLIENT_NAME} "
            f"response_actor_id={actor_id} "
            f"brief_actor_id={testee_brief.get('actor_id')} "
            f"brief_role_id={testee_brief.get('role_id')}"
        )
        weather =  brief_data["environment"]['weather']
        model.weather = weather
        model.start = 0
        #print('change_map')
        model.change_map(brief_data["zjl_odv_file"])
        map_name = brief_data["zjl_odv_file"]
        if rule_planner.config.sprint_applies_to(map_name):
            # Sprint ignores map limits and topology.  Parsing a large AITOWN
            # XODR here only for diagnostic speed statistics delayed the
            # prepare handshake by roughly half a second.
            expected_xodr_path = None
            print(
                "[expected-speed][SPRINT-FAST] "
                "skip XODR speed scan during prepare"
            )
        else:
            try:
                expected_xodr_path = (
                    global_route_planner.resolve_map_path(map_name)
                )
            except Exception:
                expected_xodr_path = None
        current_expected_speed = resolve_expected_speed(
            brief_data,
            map_name,
            xodr_path=expected_xodr_path,
            command_line_mps=EXPECTED_SPEED_CLI_MPS,
            use_xodr=USE_XODR_EXPECTED_SPEED,
        )
        rule_planner.config.expected_speed_mps = float(
            current_expected_speed["speed_mps"]
        )
        rule_planner.config.expected_speed_source = str(
            current_expected_speed["source"]
        )
        prepare_speed_candidates = current_expected_speed.get(
            "prepare_candidates", []
        )
        candidate_text = (
            ";".join(
                "{}={}{}->{:.3f}m/s".format(
                    item["path"],
                    item["raw_value"],
                    item["unit"],
                    item["speed_mps"],
                )
                for item in prepare_speed_candidates
            )
            or "none"
        )
        xodr_speed = current_expected_speed.get("xodr", {})
        print(
            "[expected-speed] "
            f"value={current_expected_speed['speed_mps']:.3f}m/s "
            f"value_kmh={current_expected_speed['speed_mps'] * 3.6:.1f}km/h "
            f"source={current_expected_speed['source']} "
            f"prepare_candidates={candidate_text} "
            f"xodr_speed_count={int(xodr_speed.get('count', 0))} "
            f"xodr_speed_min="
            f"{float(xodr_speed.get('min_mps') if xodr_speed.get('min_mps') is not None else float('nan')):.3f}m/s "
            f"xodr_speed_median="
            f"{float(xodr_speed.get('median_mps') if xodr_speed.get('median_mps') is not None else float('nan')):.3f}m/s "
            f"xodr_speed_max="
            f"{float(xodr_speed.get('max_mps') if xodr_speed.get('max_mps') is not None else float('nan')):.3f}m/s"
        )
        if rule_planner.config.override_map_speed_limit:
            print(
                "[expected-speed][WARN] "
                "--override_map_speed_limit is active; the resolved "
                "expected speed will not cap cruise speed"
            )
        rule_planner.reset(map_name)
        stable_controller.reset()
        if roundabout_controller is not None:
            roundabout_controller.reset(map_name)
        last_rule_plan = None
        last_rule_plan_wall_time = 0.0
        last_control_wall_time = None
        last_drive_plan_debug_ts = 0.0
        last_drive_control_debug_ts = 0.0
        last_drive_feedback_debug_ts = 0.0
        last_drive_ins_debug_ts = 0.0
        last_drive_lidar_debug_ts = 0.0
        init_state = testee_brief["init_state"]
        target_state = testee_brief["target_state"]
        episode_initial_speed = initial_state_speed_mps(init_state)
        reset_final_control_publisher(episode_initial_speed)
        print(
            "[startup-speed] "
            f"initial="
            f"{episode_initial_speed if episode_initial_speed is not None else float('nan'):.3f}m/s "
            "source=ActorPrepare.init_state.speed"
        )
        #print(target_state)
        role_id = testee_brief["role_id"]
        if drive_trace_logger is not None:
            drive_trace_logger.start_session(
                session_id,
                map_name,
                metadata={
                    "algorithm_policy_version": (
                        ALGORITHM_POLICY_VERSION
                    ),
                    "role_id": role_id,
                    "init_state": init_state,
                    "target_state": target_state,
                    "weather": weather,
                    "expected_speed": dict(
                        current_expected_speed or {}
                    ),
                    "planner_config": {
                        "max_speed": rule_planner.config.max_speed,
                        "override_map_speed_limit": (
                            rule_planner.config.override_map_speed_limit
                        ),
                        "respect_path_speed_limit": (
                            rule_planner.config.respect_path_speed_limit
                        ),
                        "stop_at_goal": (
                            rule_planner.config.stop_at_goal
                        ),
                        "goal_decel": (
                            rule_planner.config.goal_decel
                        ),
                        "gt_track_hold_seconds": (
                            gt_obstacle_adapter.track_hold_seconds
                        ),
                        "gt_startup_grace_seconds": (
                            GT_STARTUP_GRACE_SECONDS
                        ),
                        "gt_track_innovation_gate_m": (
                            gt_obstacle_adapter.innovation_gate_m
                        ),
                        "gt_track_innovation_gate_speed": (
                            gt_obstacle_adapter
                            .innovation_gate_speed
                        ),
                        "strict_alignment_speed_guard": (
                            rule_planner.config
                            .strict_alignment_speed_guard
                        ),
                        "max_accel": rule_planner.config.max_accel,
                        "max_decel": rule_planner.config.max_decel,
                        "max_lon_jerk": (
                            rule_planner.config.max_lon_jerk
                        ),
                        "max_lat_speed": (
                            rule_planner.config.max_lat_speed
                        ),
                        "max_lat_accel": (
                            rule_planner.config.max_lat_accel
                        ),
                        "max_lat_jerk": (
                            rule_planner.config.max_lat_jerk
                        ),
                        "max_curvature": (
                            rule_planner.config.max_curvature
                        ),
                        "max_cartesian_lat_accel": (
                            rule_planner.config.max_lateral_accel
                        ),
                        "curve_speed_factor": (
                            rule_planner.config.curve_speed_factor
                        ),
                        "centerline_feedback_gain": (
                            rule_planner.config.centerline_feedback_gain
                        ),
                        "centerline_natural_frequency": (
                            rule_planner.config
                            .centerline_natural_frequency
                        ),
                        "centerline_damping_ratio": (
                            rule_planner.config
                            .centerline_damping_ratio
                        ),
                        "steering_ratio": (
                            rule_planner.config.steering_ratio
                        ),
                        "scenario_overrides_path": (
                            rule_planner.config
                            .scenario_overrides_path
                        ),
                        "centerline_safety_stop_enabled": (
                            rule_planner.config
                            .centerline_safety_stop_enabled
                        ),
                        "follow_time_headway": (
                            rule_planner.config.time_headway
                        ),
                        "minimum_gap": (
                            rule_planner.config.minimum_gap
                        ),
                        "avoidance_speed": (
                            rule_planner.config.static_avoidance_speed
                        ),
                        "avoidance_half_width": (
                            rule_planner.config.avoidance_half_width
                        ),
                        "minimum_bypass_shift": (
                            rule_planner.config.minimum_bypass_shift
                        ),
                        "static_side_clearance": (
                            rule_planner.config.static_side_clearance
                        ),
                        "lead_lateral_tolerance": (
                            rule_planner.config
                            .lead_lateral_tolerance
                        ),
                        "static_avoidance_min_hits": (
                            rule_planner.config
                            .static_avoidance_min_hits
                        ),
                        "static_avoidance_trigger_distance": (
                            rule_planner.config
                            .static_avoidance_trigger_distance
                        ),
                        "lateral_sample_step": (
                            rule_planner.config.lateral_sample_step
                        ),
                        "collision_margin": (
                            rule_planner.config.collision_margin
                        ),
                        "collision_check_dt": (
                            rule_planner.config.collision_check_dt
                        ),
                        "planning_horizons": (
                            rule_planner.config.horizons
                        ),
                        "non_yielding_replay_traffic": (
                            rule_planner.config
                            .non_yielding_replay_traffic
                        ),
                        "rear_follow_lateral_tolerance": (
                            rule_planner.config
                            .rear_follow_lateral_tolerance
                        ),
                        "rear_pressure_distance": (
                            rule_planner.config.rear_pressure_distance
                        ),
                        "rear_pressure_closing_speed": (
                            rule_planner.config
                            .rear_pressure_closing_speed
                        ),
                    },
                },
            )
            print(
                "[drive-trace] session reset "
                f"session_id={session_id} "
                f"latest={drive_trace_logger.latest_path}"
            )
        start_ins_start_gate(init_state["x"], init_state["y"], map_name)
        model.set_destination(     
            target_state["x"],
            target_state["y"],
            np.deg2rad(target_state["orientation_z"]),
        )
        # Parse OpenDRIVE and compute the route synchronously in this process.
        plan_request_sent = build_global_plan_directly(brief_data)
        if drive_trace_logger is not None and global_plan_data is not None:
            drive_trace_logger.record_event(
                "global_plan",
                metadata={
                    "path": _sample_path_for_trace(global_plan_data),
                },
            )
        map_loading = False
        map_ready = bool(plan_request_sent)
        if map_ready:
            map_ready_ts = time.time()
            prepare_not_before_ts = (
                map_ready_ts + PREPARE_RESPONSE_DELAY
            )
            print(
                "[prepare-timing] map_ready "
                f"wall_time={map_ready_ts:.3f} "
                f"elapsed={map_ready_ts - prepare_received_ts:.3f} "
                f"map={map_name} "
                f"global_plan_points={global_plan_ready_points} "
                "planner=direct-python "
                f"prepare_not_before={prepare_not_before_ts:.3f}"
            )
        else:
            print(
                "[prepare-timing][WARN] map not ready because direct route planning failed; "
                "send ActorPrepareResult(result=False)"
            )
            prepare(False)
            recv_prepare = False


def get_pointcloud_msg():
    global first_pointcloud_ready
    global first_pointcloud_ready_ts
    global last_rule_plan
    global last_rule_plan_wall_time
    global last_control_wall_time
    global last_drive_plan_debug_ts
    global last_drive_control_debug_ts
    global last_drive_lidar_debug_ts
    global last_npc_web_debug_ts
    global last_gt_planning_timestamp
    lidar_get_started = time.monotonic()
    debug_lidar_frame = (
        DEBUG_DRIVE
        and lidar_get_started - last_drive_lidar_debug_ts >= 0.5
    )
    if debug_lidar_frame:
        last_drive_lidar_debug_ts = lidar_get_started
        print("[drive-debug][lidar] calling get_pointcloud()")
    msg = pointcloud_channel.get_pointcloud()
    lidar_get_elapsed = time.monotonic() - lidar_get_started
    fresh_pointcloud = len(msg) > 0
    if fresh_pointcloud and not first_pointcloud_ready:
        first_pointcloud_ready = True
        first_pointcloud_ready_ts = time.time()
        print(
            "[prepare-timing] first_pointcloud_ready "
            f"wall_time={first_pointcloud_ready_ts:.3f}"
        )
    if debug_lidar_frame:
        point_count = 0
        for pointcloud in msg:
            try:
                point_count += len(pointcloud.points)
            except Exception:
                pass
        print(
            "[drive-debug][lidar] "
            f"messages={len(msg)} points={point_count} "
            f"get_time={lidar_get_elapsed:.3f}s "
            "starting_perception=true"
        )
    perception_started = time.monotonic()
    perception_visualizer = getattr(
        model, "perception_web_visualizer", None
    )
    truth = None
    truth_delta_s = None
    gt_ready = False
    fresh_gt = False
    gt_age = float("inf")
    if PERCEPTION_SOURCE == "none":
        obstacles = []
    elif PERCEPTION_SOURCE == "gt":
        if npc_truth_frames:
            candidate = npc_truth_frames[-1]
            gt_age = (
                time.monotonic()
                - float(
                    candidate.get(
                        "_received_monotonic",
                        time.monotonic(),
                    )
                )
            )
            max_gt_age = max(
                0.10,
                float(
                    os.environ.get(
                        "GT_PERCEPTION_MAX_AGE", "0.50"
                    )
                ),
            )
            if gt_age <= max_gt_age:
                truth = candidate
                gt_ready = True
                truth_timestamp = float(truth["timestamp_s"])
                fresh_gt = (
                    last_gt_planning_timestamp is None
                    or abs(
                        truth_timestamp
                        - last_gt_planning_timestamp
                    )
                    > 1e-6
                )
                if fresh_gt:
                    gt_obstacle_adapter.update(
                        truth,
                        truth.get("_ego_snapshot") or model.ego,
                    )
                    last_gt_planning_timestamp = truth_timestamp
                obstacles = gt_obstacle_adapter.predict_at(
                    truth_timestamp + gt_age
                )
            else:
                obstacles = []
        else:
            obstacles = []
    else:
        obstacles = model.perceive(msg, map_name)
    perception_elapsed = time.monotonic() - perception_started

    if npc_truth_frames and PERCEPTION_SOURCE != "gt":
        lidar_timestamp = getattr(
            model,
            "latest_obstacle_measurement_stamp_s",
            None,
        )
        truth, truth_delta_s = _nearest_npc_truth(
            lidar_timestamp,
            max_delta_s=float(
                os.environ.get("NPC_TRUTH_MAX_SYNC_DELTA", "0.30")
            ),
        )
    else:
        lidar_timestamp = None
    if perception_visualizer is not None:
        perception_visualizer.publish_ground_truth(
            truth,
            lidar_timestamp=lidar_timestamp,
            sync_delta_s=truth_delta_s,
        )
        if truth is not None:
            debug_now = time.monotonic()
            if (
                DEBUG_DRIVE
                and debug_now - last_npc_web_debug_ts >= 0.5
            ):
                last_npc_web_debug_ts = debug_now
                delta_text = (
                    f"{truth_delta_s * 1000.0:+.1f}ms"
                    if truth_delta_s is not None
                    else "-"
                )
                print(
                    "[npc-truth][web] "
                    f"buffer_frames={len(npc_truth_frames)} "
                    f"selected=True "
                    f"boxes={len(truth.get('roles', []))} "
                    f"npc_minus_lidar={delta_text} "
                    f"url=http://127.0.0.1:"
                    f"{perception_visualizer.port}"
                )
    if debug_lidar_frame:
        print(
            "[drive-debug][lidar] "
            f"perception_complete=true elapsed={perception_elapsed:.3f}s "
            f"source={PERCEPTION_SOURCE} "
            f"gt_ready={gt_ready} gt_age={gt_age:.3f}s "
            f"obstacles={len(obstacles or [])} "
            f"gt_predicted="
            f"{sum(bool(getattr(item, 'track_predicted', False)) for item in (obstacles or []))} "
            f"gt_jump_rejected="
            f"{sum(bool(getattr(item, 'innovation_rejected', False)) for item in (obstacles or []))}"
        )

    now = time.monotonic()
    fresh_perception = (
        fresh_gt
        if PERCEPTION_SOURCE == "gt"
        else fresh_pointcloud
    )
    replanned = False
    planning_elapsed = 0.0
    # Detector frames trigger immediate replanning. Between detector frames a
    # bounded 5 Hz refresh accounts for moving ego and cached obstacle motion.
    if (
        fresh_perception
        or last_rule_plan is None
        or now - last_rule_plan_wall_time >= 0.20
    ):
        planning_started = time.monotonic()
        if (
            roundabout_controller is not None
            and roundabout_controller.enabled
        ):
            last_rule_plan = roundabout_controller.plan(
                ego=model.ego,
                obstacles=obstacles,
                global_path=global_plan_data,
                map_name=map_name,
            )
            planning_debug_owner = roundabout_controller
        else:
            last_rule_plan = rule_planner.plan(
                ego=model.ego,
                obstacles=obstacles,
                global_path=global_plan_data,
                map_name=map_name,
                ego_lateral_speed=model.ego_ros_vy,
                now=now,
            )
            planning_debug_owner = rule_planner
        planning_elapsed = time.monotonic() - planning_started
        replanned = True
        last_rule_plan_wall_time = now
        planner_debug = getattr(
            planning_debug_owner, "last_debug", {}
        ) or {}
        projection_debug = planner_debug.get("projection") or {}
        perception_visualizer = getattr(
            model, "perception_web_visualizer", None
        )
        if perception_visualizer is not None:
            perception_visualizer.publish_paths(
                ego=model.ego,
                global_path=global_plan_data,
                local_trajectory=getattr(
                    last_rule_plan, "trajectory", None
                ),
                behavior=getattr(last_rule_plan, "behavior", ""),
                target_speed=getattr(last_rule_plan, "target_speed", 0.0),
                emergency=getattr(last_rule_plan, "emergency", False),
                current_s=projection_debug.get("s"),
                current_d=projection_debug.get("d"),
                map_name=map_name,
                override_active=planner_debug.get(
                    "manual_control_active", False
                ),
                override_name=planner_debug.get(
                    "manual_override_name"
                ),
                override_s_start=planner_debug.get("manual_s_start"),
                override_s_end=planner_debug.get("manual_s_end"),
            )
        if DEBUG_DRIVE and now - last_drive_plan_debug_ts >= 0.5:
            last_drive_plan_debug_ts = now
            projection = None
            try:
                projection = planning_debug_owner.reference.project(
                    float(model.ego.x), float(model.ego.y)
                )
            except Exception:
                pass
            path_distance = (
                float(projection["distance"])
                if projection is not None
                else float("nan")
            )
            trajectory = getattr(last_rule_plan, "trajectory", None)
            trajectory_points = (
                int(getattr(getattr(trajectory, "x", None), "size", 0))
                if trajectory is not None
                else 0
            )
            print(
                "[drive-debug][plan] "
                f"ego=({model.ego.x:.3f},{model.ego.y:.3f}) "
                f"ego_speed={model.ego.speed:.3f}m/s "
                f"path_distance={path_distance:.3f}m "
                f"obstacles={len(obstacles or [])} "
                f"behavior={last_rule_plan.behavior} "
                f"emergency={last_rule_plan.emergency} "
                f"target_speed={last_rule_plan.target_speed:.3f}m/s "
                f"trajectory_points={trajectory_points} "
                f"reason={last_rule_plan.reason or 'none'}"
            )
            speed_limit_debug = (
                planner_debug.get("speed_limits") or {}
            )
            hard_rejections = planner_debug.get("hard_rejections") or {}
            hard_text = ",".join(
                f"{name}:{count}"
                for name, count in sorted(hard_rejections.items())
            ) or "none"
            rear_non_blocking_text = ",".join(
                str(value)
                for value in planner_debug.get(
                    "rear_non_blocking_ids", []
                )
            ) or "none"
            rear_pressure_debug = (
                planner_debug.get("rear_pressure") or {}
            )
            print(
                "[drive-debug][plan-detail] "
                f"s={float(projection_debug.get('s', float('nan'))):.3f}m "
                f"d={float(projection_debug.get('d', float('nan'))):.3f}m "
                f"path_yaw={math.degrees(float(projection_debug.get('yaw', float('nan')))):.3f}deg "
                f"heading_error={math.degrees(float(planner_debug.get('heading_error', float('nan')))):.3f}deg "
                f"path_kappa={float(projection_debug.get('kappa', float('nan'))):.4f} "
                f"ego_vy={float(planner_debug.get('ego_lateral_speed', float('nan'))):.3f}m/s "
                f"speed_caps_kmh="
                f"scene:{3.6 * float(speed_limit_debug.get('scene_limit_mps', float('nan'))):.1f},"
                f"path:{3.6 * float(speed_limit_debug.get('path_limit_mps', float('nan'))):.1f},"
                f"curve:{3.6 * float(speed_limit_debug.get('curve_limit_mps', float('nan'))):.1f},"
                f"goal:{3.6 * float(speed_limit_debug.get('goal_limit_mps', float('nan'))):.1f} "
                f"curve_lat_budget={float(speed_limit_debug.get('curve_lateral_accel_budget_mps2', float('nan'))):.2f}m/s2 "
                f"map_curve_override={bool(speed_limit_debug.get('map_curve_override_active', False))} "
                f"map_override={bool(speed_limit_debug.get('override_map_speed_limit', False))} "
                f"respect_path_cap={bool(speed_limit_debug.get('respect_path_speed_limit', False))} "
                f"stop_at_goal={bool(speed_limit_debug.get('stop_at_goal', False))} "
                f"expected_speed_source={speed_limit_debug.get('expected_speed_source', 'unknown')} "
                f"manual_control={bool(planner_debug.get('manual_control_active', False))} "
                f"manual_rule={planner_debug.get('manual_override_name')} "
                f"manual_target_d={float(planner_debug.get('manual_target_d') if planner_debug.get('manual_target_d') is not None else float('nan')):.3f}m "
                f"manual_target_v={float(planner_debug.get('manual_target_speed_mps') if planner_debug.get('manual_target_speed_mps') is not None else float('nan')):.3f}m/s "
                f"collision_bypass={bool(planner_debug.get('manual_collision_bypass', False) or planner_debug.get('sprint_collision_bypass', False))} "
                f"sprint={bool(planner_debug.get('sprint_active', False))} "
                f"sprint_goal_distance={float(planner_debug.get('sprint_goal_distance') if planner_debug.get('sprint_goal_distance') is not None else float('nan')):.3f}m "
                f"roundabout_mode={planner_debug.get('mode', 'off')} "
                f"roundabout_lead={planner_debug.get('locked_lead_id')} "
                f"roundabout_gap={float((planner_debug.get('lead') or {}).get('gap', float('nan'))):.3f}m "
                f"roundabout_lead_d={float((planner_debug.get('lead') or {}).get('d', float('nan'))):.3f}m "
                f"roundabout_path_vehicles={int(planner_debug.get('path_vehicle_count', 0))} "
                f"roundabout_path_d_limit={float(planner_debug.get('path_d_threshold', float('nan'))):.3f}m "
                f"roundabout_curve_v={float(planner_debug.get('curve_speed_limit', float('nan'))):.3f}m/s "
                f"roundabout_curve_kappa={float(planner_debug.get('lookahead_max_curvature', float('nan'))):.4f} "
                f"remaining={float(planner_debug.get('remaining', float('nan'))):.3f}m "
                f"obstacles_raw={int(planner_debug.get('raw_obstacle_count', 0))} "
                f"obstacles_used={int(planner_debug.get('planning_obstacle_count', 0))} "
                f"generated={int(planner_debug.get('generated_candidates', 0))} "
                f"accepted={int(planner_debug.get('accepted_candidates', 0))} "
                f"collision_rejected={int(planner_debug.get('collision_rejected', 0))} "
                f"rear_non_blocking={rear_non_blocking_text} "
                f"rear_pressure_id={rear_pressure_debug.get('id')} "
                f"rear_pressure_distance="
                f"{float(rear_pressure_debug.get('distance', float('nan'))):.2f}m "
                f"rear_pressure_closing="
                f"{float(rear_pressure_debug.get('closing_speed', float('nan'))):.2f}m/s "
                f"selection={planner_debug.get('candidate_selection_mode', 'DIRECT')} "
                f"direct_clear={bool(planner_debug.get('direct_clear_path', False))} "
                f"static_avoidance={bool(planner_debug.get('static_avoidance', False))} "
                f"avoidance_side={int(planner_debug.get('avoidance_side', 0))} "
                f"avoidance_candidates={int(planner_debug.get('avoidance_candidates', 0))} "
                f"hard_rejected={hard_text} "
                f"selected_d={float(planner_debug.get('selected_target_d', float('nan'))):.3f}m "
                f"horizon={float(planner_debug.get('selected_horizon', float('nan'))):.2f}s "
                f"cost={float(planner_debug.get('selected_cost', float('nan'))):.3f} "
                f"clearance={float(planner_debug.get('selected_clearance', float('nan'))):.3f} "
                f"closest_id={planner_debug.get('selected_closest_obstacle_id')} "
                f"closest_t={float(planner_debug.get('selected_closest_time') if planner_debug.get('selected_closest_time') is not None else float('nan')):.2f}s "
                f"max_kappa={float(planner_debug.get('selected_max_curvature', float('nan'))):.4f}"
            )
            cost_components = planner_debug.get("selected_cost_components") or {}
            if cost_components:
                print(
                    "[drive-debug][cost] "
                    + " ".join(
                        f"{name}={float(value):.3f}"
                        for name, value in sorted(cost_components.items())
                    )
                )
            for candidate_debug in planner_debug.get("top_candidates", []):
                print(
                    "[drive-debug][candidate] "
                    f"rank={int(candidate_debug.get('rank', 0))} "
                    f"target_d={float(candidate_debug.get('target_d', float('nan'))):.3f}m "
                    f"horizon={float(candidate_debug.get('horizon', float('nan'))):.2f}s "
                    f"end_speed={float(candidate_debug.get('end_speed', float('nan'))):.3f}m/s "
                    f"cost={float(candidate_debug.get('cost', float('nan'))):.3f} "
                    f"clearance={float(candidate_debug.get('clearance', float('nan'))):.3f} "
                    f"max_kappa={float(candidate_debug.get('max_curvature', float('nan'))):.4f}"
                )
            lead_debug = planner_debug.get("lead")
            if lead_debug is not None:
                print(
                    "[drive-debug][lead] "
                    f"id={lead_debug.get('id')} "
                    f"gap={float(lead_debug.get('gap', float('nan'))):.3f}m "
                    f"s={float(lead_debug.get('s', float('nan'))):.3f}m "
                    f"d={float(lead_debug.get('d', float('nan'))):.3f}m "
                    f"speed={float(lead_debug.get('longitudinal_speed', float('nan'))):.3f}m/s"
                )
            for obstacle_debug in planner_debug.get("obstacles", []):
                print(
                    "[drive-debug][obstacle] "
                    f"id={obstacle_debug.get('id')} "
                    f"gap={float(obstacle_debug.get('gap', float('nan'))):.3f}m "
                    f"s={float(obstacle_debug.get('s', float('nan'))):.3f}m "
                    f"d={float(obstacle_debug.get('d', float('nan'))):.3f}m "
                    f"speed={float(obstacle_debug.get('speed', float('nan'))):.3f}m/s "
                    f"speed_valid={bool(obstacle_debug.get('speed_valid', False))} "
                    f"type={int(obstacle_debug.get('obs_type', -1))} "
                    f"score={float(obstacle_debug.get('score', 0.0)):.2f} "
                    f"hits={int(obstacle_debug.get('track_hits', 0))} "
                    f"misses={int(obstacle_debug.get('track_misses', 0))} "
                    f"predicted={bool(obstacle_debug.get('track_predicted', False))} "
                    f"pedestrian={bool(obstacle_debug.get('pedestrian', False))}"
                )

    if last_control_wall_time is None:
        control_dt = 0.05
    else:
        control_dt = now - last_control_wall_time
    last_control_wall_time = now
    roundabout_active = bool(
        roundabout_controller is not None
        and roundabout_controller.enabled
    )
    current_planner_debug = getattr(
        roundabout_controller if roundabout_active else rule_planner,
        "last_debug",
        {},
    ) or {}
    current_projection_debug = current_planner_debug.get("projection") or {}
    actual_path_lateral_offset = current_projection_debug.get("d")
    manual_target_d = current_planner_debug.get("manual_target_d")
    control_path_lateral_error = actual_path_lateral_offset
    if (
        current_planner_debug.get("manual_control_active", False)
        and actual_path_lateral_offset is not None
        and manual_target_d is not None
    ):
        control_path_lateral_error = (
            float(actual_path_lateral_offset)
            - float(manual_target_d)
        )
    control_started = time.monotonic()
    steering_feedback = getattr(
        getattr(model, "vehicle_feedback", None),
        "steering_wheel_angle",
        None,
    )
    if roundabout_active:
        command = roundabout_controller.control(
            ego=model.ego,
            obstacles=obstacles,
            global_path=global_plan_data,
            dt=control_dt,
            map_name=map_name,
            steering_feedback=steering_feedback,
        )
        last_rule_plan = roundabout_controller.last_plan
        current_planner_debug = (
            roundabout_controller.last_debug
        )
        current_projection_debug = (
            current_planner_debug.get("projection") or {}
        )
        actual_path_lateral_offset = (
            current_projection_debug.get("d")
        )
    else:
        command = stable_controller.control(
            model.ego,
            last_rule_plan,
            control_dt,
            steering_feedback=steering_feedback,
            path_lateral_offset=control_path_lateral_error,
            path_reference_yaw=current_projection_debug.get("yaw"),
            path_reference_curvature=current_projection_debug.get(
                "kappa"
            ),
        )
        stable_controller.last_debug.update(
            {
                "actual_path_lateral_offset": (
                    actual_path_lateral_offset
                ),
                "manual_target_d": manual_target_d,
                "manual_control_active": bool(
                    current_planner_debug.get(
                        "manual_control_active", False
                    )
                ),
                "manual_collision_bypass": bool(
                    current_planner_debug.get(
                        "manual_collision_bypass", False
                    )
                ),
                "sprint_collision_bypass": bool(
                    current_planner_debug.get(
                        "sprint_collision_bypass", False
                    )
                ),
            }
        )
    gt_startup_age = (
        float("inf")
        if first_ins_ready_ts is None
        else max(0.0, time.time() - first_ins_ready_ts)
    )
    gt_startup_grace_active = bool(
        PERCEPTION_SOURCE == "gt"
        and not gt_ready
        and gt_startup_age <= GT_STARTUP_GRACE_SECONDS
    )
    gt_failsafe_applied = (
        PERCEPTION_SOURCE == "gt"
        and not gt_ready
        and not gt_startup_grace_active
        and not bool(
            current_planner_debug.get(
                "manual_control_active", False
            )
            or current_planner_debug.get(
                "sprint_collision_bypass", False
            )
            or roundabout_active
        )
    )
    if gt_failsafe_applied:
        command.acc = -float(rule_planner.config.max_decel)
        command.speed = 0.0
    control_elapsed = time.monotonic() - control_started
    controller_debug = getattr(
        roundabout_controller if roundabout_active else stable_controller,
        "last_debug",
        {},
    ) or {}
    if DEBUG_DRIVE and now - last_drive_control_debug_ts >= 0.5:
        last_drive_control_debug_ts = now
        print(
            "[drive-debug][control] "
            f"dt={control_dt:.3f}s "
            f"ego_speed={float(getattr(model.ego, 'speed', 0.0)):.3f}m/s "
            f"acc={command.acc:.3f}m/s2 "
            f"target_speed={command.speed:.3f}m/s "
            f"steer={command.steer:.3f}deg"
        )
        print(
            "[drive-debug][control-detail] "
            f"speed_error={float(controller_debug.get('speed_error', float('nan'))):.3f}m/s "
            f"planned_acc={float(controller_debug.get('planned_accel', float('nan'))):.3f}m/s2 "
            f"desired_acc={float(controller_debug.get('desired_acc', float('nan'))):.3f}m/s2 "
            f"accel_jerk_limit={float(controller_debug.get('accel_command_jerk') if controller_debug.get('accel_command_jerk') is not None else float('nan')):.3f}m/s3 "
            f"accel_jerk_actual={float(controller_debug.get('actual_accel_command_jerk', float('nan'))):.3f}m/s3 "
            f"stale_brake_cleared={bool(controller_debug.get('stationary_stale_brake_cleared', False))} "
            f"integral={float(controller_debug.get('speed_integral', float('nan'))):.3f} "
            f"lookahead={float(controller_debug.get('lookahead', float('nan'))):.3f}m "
            f"lateral_error={float(controller_debug.get('lateral_error', float('nan'))):.3f}m "
            f"heading_error={math.degrees(float(controller_debug.get('heading_error', float('nan')))):.3f}deg "
            f"ref_kappa={float(controller_debug.get('reference_curvature', float('nan'))):.4f} "
            f"model_front_angle={float(controller_debug.get('model_front_angle_deg', float('nan'))):.3f}deg "
            f"command_sign={float(controller_debug.get('steering_command_sign', float('nan'))):.0f} "
            f"raw_steer={float(controller_debug.get('raw_steer', float('nan'))):.3f}deg "
            f"filtered_steer={float(controller_debug.get('filtered_steer', float('nan'))):.3f}deg "
            f"steer_rate_limit={float(controller_debug.get('steering_rate_limit', float('nan'))):.3f}deg/s "
            f"steer_rate={float(controller_debug.get('steering_rate', float('nan'))):.3f}deg/s "
            f"estimated_lat_acc={float(controller_debug.get('estimated_lateral_accel', float('nan'))):.3f}m/s2 "
            f"estimated_lat_jerk={float(controller_debug.get('estimated_lateral_jerk', float('nan'))):.3f}m/s3 "
            f"estimated_yaw_rate={float(controller_debug.get('estimated_yaw_rate', float('nan'))):.3f}rad/s "
            f"steer_feedback={float(controller_debug.get('steering_feedback') if controller_debug.get('steering_feedback') is not None else float('nan')):.3f}deg "
            f"steer_tracking_error={float(controller_debug.get('steering_tracking_error') if controller_debug.get('steering_tracking_error') is not None else float('nan')):.3f}deg "
            f"steer_limit={float(controller_debug.get('steering_limit', float('nan'))):.3f}deg "
            f"trajectory_comfort_cap={float(controller_debug.get('trajectory_comfort_speed_cap', float('nan'))):.3f}m/s "
            f"trajectory_comfort_cap_applied={bool(controller_debug.get('trajectory_comfort_cap_applied', False))} "
            f"alignment_speed_cap={float(controller_debug.get('alignment_speed_cap', float('nan'))):.3f}m/s "
            f"strict_alignment_guard={bool(controller_debug.get('strict_alignment_speed_guard', False))} "
            f"path_d={float(controller_debug.get('path_lateral_offset', float('nan'))):.3f}m "
            f"actual_path_d={float(controller_debug.get('actual_path_lateral_offset') if controller_debug.get('actual_path_lateral_offset') is not None else float('nan')):.3f}m "
            f"manual_target_d={float(controller_debug.get('manual_target_d') if controller_debug.get('manual_target_d') is not None else float('nan')):.3f}m "
            f"manual_control={bool(controller_debug.get('manual_control_active', False))} "
            f"collision_bypass={bool(controller_debug.get('manual_collision_bypass', False) or controller_debug.get('sprint_collision_bypass', False))} "
            f"centerline_control={bool(controller_debug.get('centerline_control_active', False))} "
            f"global_heading_error={math.degrees(float(controller_debug.get('global_heading_error', float('nan')))):.3f}deg "
            f"path_d_speed={float(controller_debug.get('estimated_path_lateral_speed', float('nan'))):.3f}m/s "
            f"centerline_lat_acc={float(controller_debug.get('centerline_lat_accel_command', float('nan'))):.3f}m/s2 "
            f"active_lat_acc_limit={float(controller_debug.get('active_lateral_accel_limit', float('nan'))):.3f}m/s2 "
            f"curve_authority={bool(controller_debug.get('curve_authority_active', False))} "
            f"tracking_recovery={bool(controller_debug.get('tracking_recovery_active', False))} "
            f"sprint_mode={bool(controller_debug.get('sprint_mode', False))} "
            f"sprint_phase={controller_debug.get('sprint_phase', 'INACTIVE')} "
            f"sprint_elapsed={float(controller_debug.get('sprint_elapsed', 0.0)):.3f}s "
            f"sprint_alignment_duration={float(controller_debug.get('sprint_alignment_duration', 0.0)):.3f}s "
            f"sprint_pulse_sent={bool(controller_debug.get('sprint_pulse_sent', False))} "
            f"sprint_pulse_elapsed={float(controller_debug.get('sprint_pulse_elapsed', 0.0)):.3f}s "
            f"sprint_pulse_min_duration={float(controller_debug.get('sprint_pulse_min_duration', 0.0)):.3f}s "
            f"sprint_pulse_ack_speed={float(controller_debug.get('sprint_pulse_ack_speed', 0.0)):.3f}m/s "
            f"sprint_pulse_ack={bool(controller_debug.get('sprint_pulse_acknowledged', False))} "
            f"sprint_pulse={bool(controller_debug.get('sprint_accel_pulse_active', False))} "
            f"sprint_recovery={bool(controller_debug.get('sprint_recovery_active', False))} "
            f"roundabout_mode={controller_debug.get('roundabout_mode', 'off')} "
            f"roundabout_exclusive={bool(controller_debug.get('roundabout_exclusive', False))} "
            f"idm_dynamic_gap={float(controller_debug.get('desired_dynamic_gap', float('nan'))):.3f}m "
            f"idm_closing_speed={float(controller_debug.get('closing_speed', float('nan'))):.3f}m/s "
            f"idm_free={float(controller_debug.get('idm_free_road_term', float('nan'))):.3f} "
            f"idm_interaction={float(controller_debug.get('idm_interaction_term', float('nan'))):.3f} "
            f"ttc={float(controller_debug.get('ttc', float('nan'))):.3f}s "
            f"ttc_threshold={float(controller_debug.get('ttc_threshold', float('nan'))):.3f}s "
            f"ttc_emergency={bool(controller_debug.get('ttc_emergency_active', False))} "
            f"curve_speed_limit={float(controller_debug.get('curve_speed_limit', float('nan'))):.3f}m/s "
            f"curve_max_kappa={float(controller_debug.get('lookahead_max_curvature', float('nan'))):.4f} "
            f"path_d_change={float(controller_debug.get('path_offset_change', float('nan'))):.3f}m "
            f"divergence_count={int(controller_debug.get('path_divergence_count', 0))} "
            f"centerline_stop={bool(controller_debug.get('centerline_safety_stop', False))} "
            f"reason={controller_debug.get('reason') or 'none'}"
        )
    if drive_trace_logger is not None:
        _record_drive_trace(
            loop_count=loop_count,
            ego=model.ego,
            obstacles=obstacles,
            plan_result=last_rule_plan,
            planner_debug=current_planner_debug,
            control_command=command,
            controller_debug=controller_debug,
            vehicle_feedback=getattr(model, "vehicle_feedback", None),
            extra={
                "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "fresh_pointcloud": fresh_pointcloud,
                "fresh_gt": fresh_gt,
                "gt_ready": gt_ready,
                "gt_age": gt_age,
                "gt_startup_age": gt_startup_age,
                "gt_startup_grace_seconds": (
                    GT_STARTUP_GRACE_SECONDS
                ),
                "gt_startup_grace_active": (
                    gt_startup_grace_active
                ),
                "gt_failsafe_applied": gt_failsafe_applied,
                "perception_source": PERCEPTION_SOURCE,
                "replanned": replanned,
                "control_dt": control_dt,
                "lidar_get_elapsed": lidar_get_elapsed,
                "perception_elapsed": perception_elapsed,
                "planning_elapsed": planning_elapsed,
                "control_elapsed": control_elapsed,
                "perception_measurement_time": getattr(
                    model,
                    "latest_obstacle_measurement_stamp_s",
                    None,
                ),
                "ins_sequence": last_ins_sequence,
                "ros_vy": getattr(model, "ego_ros_vy", None),
                "yaw_rate": getattr(model, "ego_ros_yawrate", None),
                "lon_acc": getattr(model, "ego_lon_acc", None),
                "lat_acc": getattr(model, "ego_lat_acc", None),
            },
        )
    return command, False


def drain_sensor_queues(drain_pointcloud=True):
    process_npc_truth()
    if drain_pointcloud:
        pointcloud_channel.get_pointcloud()


def process_image_msg(images):
    global img_id
    if save_results:
        save_dir = "results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for i, image in enumerate(images):
        img = image.data.astype(np.uint8).reshape(720, 1280, 3)
        if save_results:
            cv2.imwrite(os.path.join(save_dir, f"cam_{i}_" + str(img_id) + ".jpg"), img)
    img_id += 1


def get_image():
    msg = image_channel.get_image()
    if len(msg) == 0:
        return
    process_image_msg(msg)


def process_notify():
    global start_test
    global recv_prepare
    global local_abort_pending
    global local_abort_sent_ts
    global plan_start_check_remaining
    global plan_start_check_xy
    global plan_start_check_map
    global last_ins_sequence
    global last_ins_position
    global duplicate_ins_count
    global last_duplicate_ins_warn_ts
    global nonmonotonic_ins_count
    global last_nonmonotonic_ins_warn_ts
    global invalid_ins_count
    global last_invalid_ins_warn_ts
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts
    global last_rule_plan
    global last_rule_plan_wall_time
    global last_control_wall_time
    global test_start_received_ts
    ret, msg = notify_channel.get()
    if msg is None:
        return
    if ret >= 0 and msg.type() == MT_NOTIFY:
        notify = Notify()
        data = libMulticastNetwork.getMessageData(msg)
        notify.ParseFromString(data)
        notify_session_id = str(notify.session_id or "").strip()
        current_session_id = str(session_id or "").strip()
        if (
            notify.type in (NT_START_TEST, NT_FINISH_TEST, NT_ABORT_TEST)
            and notify_session_id
            and current_session_id
            and notify_session_id != current_session_id
        ):
            print(
                "[platform-notify][WARN] ignore stale session "
                f"type={notify.type} received={notify_session_id} "
                f"current={current_session_id}",
                flush=True,
            )
            return notify.type
        if notify.type == NT_FINISH_TEST or notify.type == NT_ABORT_TEST :
            deactivate_final_control_publisher()
            if drive_trace_logger is not None:
                drive_trace_logger.end_session(
                    reason=(
                        "finish"
                        if notify.type == NT_FINISH_TEST
                        else "abort"
                    ),
                    metadata={"notify_type": int(notify.type)},
                )
            local_abort_echo = notify.type == NT_ABORT_TEST and local_abort_pending
            if local_abort_echo:
                print(
                    "[episode-event] suppress local abort event; "
                    "wait for next ActorPrepare before clearing ROS path"
                )
            else:
                local_abort_pending = False
                local_abort_sent_ts = 0.0
            print("finish session", notify.type, model.start, "started", start_test)

            model.collision = 0
            model.time_out = False
            start_test = False
            recv_prepare = False
            test_start_received_ts = None
            plan_start_check_remaining = 0
            plan_start_check_xy = None
            plan_start_check_map = ""
            last_ins_sequence = None
            last_ins_position = None
            duplicate_ins_count = 0
            last_duplicate_ins_warn_ts = 0.0
            nonmonotonic_ins_count = 0
            last_nonmonotonic_ins_warn_ts = 0.0
            invalid_ins_count = 0
            last_invalid_ins_warn_ts = 0.0
            reset_ins_start_gate()
            model.start = 0
            model.step = 0
            model.reset_perception_state()
            model.last_pointclouds = []
            reset_episode_ego_state("finish")
            model.invalid_ego_sample_count = 0
            model.last_invalid_ego_sample_warn_wall_time = 0.0
            model.ok = 0
            model.last_steer = 0
            model.time_out = False
            model.start_time = 0
            model.last_time = None
            model.last_a = None
            model.last_h_a = None
            model.last_rot = None
            model.goal = 0
            rule_planner.reset(map_name)
            stable_controller.reset()
            if roundabout_controller is not None:
                roundabout_controller.reset(map_name)
            last_rule_plan = None
            last_rule_plan_wall_time = 0.0
            last_control_wall_time = None

        elif notify.type == NT_START_TEST:
            print("start session")
            start_test = True
            test_start_received_ts = time.time()
            model.start = 1
            activate_final_control_publisher()
            # DriveSim holds the chassis brake while a prepared scenario is
            # waiting to start.  Publish a neutral command immediately when
            # the start notification arrives instead of waiting for the first
            # perception/planning cycle.  Matching the current forward speed
            # releases that inherited brake without requesting a speed step.
            stable_controller.reset()
            last_control_wall_time = None
            startup_speed = startup_hold_speed()
            send_control_cmd(0.0, startup_speed, 0.0)
            print(
                "[startup-control] neutral brake release "
                f"speed={startup_speed:.3f}m/s "
                f"source="
                f"{'session-ins' if first_ins_ready else 'episode-initial-speed'}"
            )
            if drive_trace_logger is not None:
                drive_trace_logger.record_event(
                    "test_start",
                    metadata={"notify_type": int(notify.type)},
                )
        elif notify.type == NT_COLLIDE_ROLE:
            if drive_trace_logger is not None:
                drive_trace_logger.record_event(
                    "collision",
                    metadata={
                        "notify_type": int(notify.type),
                        "notify": " ".join(str(notify).split())[:2000],
                    },
                )
        elif notify.type == NT_DESTROY_ROLE:
            pass
        if notify.type is not None:
            return notify.type


def reset_final_control_publisher(initial_speed):
    global final_control_active
    global final_control_desired

    if not final_control_enabled:
        return
    with final_control_lock:
        final_control_active = False
        final_control_desired = None
        final_control_limiter.reset(initial_speed)


def activate_final_control_publisher():
    global final_control_active

    if not final_control_enabled:
        return
    with final_control_lock:
        final_control_active = True


def initialize_final_control_from_ego(ego_speed):
    """Establish a missing episode baseline without emitting a zero command."""
    global final_control_desired

    if not final_control_enabled:
        return
    with final_control_lock:
        final_control_limiter.reset(ego_speed)
        final_control_desired = (0.0, float(ego_speed), 0.0)


def deactivate_final_control_publisher():
    global final_control_active
    global final_control_desired

    if not final_control_enabled:
        return
    with final_control_lock:
        final_control_active = False
        final_control_desired = None


def _put_control_cmd(target_acc, target_speed, target_steer):
    # Never let a stale braking command launch a stationary vehicle backwards.
    # Preserve negative acceleration only while the vehicle is actually moving.
    ego_speed = float(getattr(getattr(globals().get("model"), "ego", None), "speed", 0.0) or 0.0)
    if target_acc < 0 and ego_speed < 0.08:
        target_acc = 0.0
    cmd = VehicleControl()
    cmd.acceleration = float(target_acc)
    cmd.speed = float(target_speed)
    cmd.steering_control.target_steering_wheel_angle = float(target_steer)
    data = cmd.SerializeToString()
    length = len(data)
    ret = cmd_channel.put(VEHICLE_CONTROL, length, data)
    if ret != 0:
        print("send cmd error")
    return ret


def send_control_cmd(target_acc, target_speed, target_steer):
    global final_control_desired

    if not final_control_enabled:
        return _put_control_cmd(
            target_acc,
            target_speed,
            target_steer,
        )
    with final_control_lock:
        final_control_desired = (
            float(target_acc),
            float(target_speed),
            float(target_steer),
        )
    return 0


def final_control_publisher_loop():
    """Publish the latest desired command through one fixed-rate writer."""
    global final_control_last_log_ts

    print(
        "[final-control] publisher started "
        f"interval={final_control_interval:.3f}s "
        f"safe_accel={final_control_safe_accel:.3f}m/s2 "
        f"one_shot_target={int(final_control_one_shot_target)} "
        f"one_shot_effective_interval="
        f"{final_control_one_shot_effective_interval:.6f}s "
        f"max_speed_step="
        f"{final_control_limiter.max_speed_step:.6f}m",
        flush=True,
    )
    while not final_control_stop_event.is_set():
        cycle_started = time.monotonic()
        with final_control_lock:
            if final_control_active and final_control_desired is not None:
                _, desired_speed, desired_steer = final_control_desired
                ego = getattr(globals().get("model"), "ego", None)
                ego_speed = (
                    None
                    if ego is None
                    else getattr(ego, "speed", None)
                )
                limited = final_control_limiter.step(
                    desired_speed,
                    ego_speed=ego_speed,
                )
                if limited is not None:
                    published_speed, published_accel, previous_speed = (
                        limited
                    )
                    _put_control_cmd(
                        published_accel,
                        published_speed,
                        desired_steer,
                    )
                    if (
                        final_control_one_shot_target
                        and abs(published_accel)
                        > final_control_safe_accel + 1e-6
                    ):
                        print(
                            "[final-control][ONE_SHOT] "
                            f"ego_speed="
                            f"{float(ego_speed) if ego_speed is not None else previous_speed:.6f}m/s "
                            f"target_speed={published_speed:.6f}m/s "
                            f"single_frame_accel="
                            f"{published_accel:.6f}m/s2",
                            flush=True,
                        )
                    now = time.monotonic()
                    if (
                        DEBUG_DRIVE
                        and now - final_control_last_log_ts >= 0.5
                    ):
                        final_control_last_log_ts = now
                        print(
                            "[final-control][LIMIT] "
                            f"desired_speed={desired_speed:.6f}m/s "
                            f"previous_speed={previous_speed:.6f}m/s "
                            f"published_speed={published_speed:.6f}m/s "
                            f"derived_accel={published_accel:.6f}m/s2 "
                            f"ego_speed="
                            f"{float(ego_speed) if ego_speed is not None else float('nan'):.6f}m/s"
                        )
        elapsed = time.monotonic() - cycle_started
        final_control_stop_event.wait(
            max(0.0, final_control_interval - elapsed)
        )


def start_final_control_publisher_thread():
    if not final_control_enabled:
        return None
    thread = threading.Thread(
        target=final_control_publisher_loop,
        name="final-control-publisher",
        daemon=True,
    )
    thread.start()
    return thread


def get_vehicle_feedback():
    global last_drive_feedback_debug_ts

    ret, msg = cmd_channel.get()
    if msg is None or ret < 0:
        return
    if msg.type() == VEHICLE_FEEDBACK:
        feedback = VehicleFeedback()
        data = libMulticastNetwork.getMessageData(msg)
        feedback.ParseFromString(data)
        model.update_vehicle_feedback(feedback)
        now = time.monotonic()
        if DEBUG_DRIVE and now - last_drive_feedback_debug_ts >= 0.5:
            last_drive_feedback_debug_ts = now
            state = model.vehicle_feedback
            print(
                "[drive-debug][feedback] "
                f"steer={state.steering_wheel_angle:.3f}deg "
                f"accelerator={state.accelerator_pedal_position:.3f} "
                f"brake={state.brake_pedal_position:.3f}"
            )
            raw_feedback = " ".join(str(feedback).split())
            print(
                "[drive-debug][feedback-raw] "
                f"{raw_feedback[:1200] or 'empty'}"
            )


def get_vehicle_pose():
    global last_ins_sequence
    global last_ins_position
    global duplicate_ins_count
    global invalid_ins_count
    global first_ins_ready
    global first_ins_ready_ts
    global last_drive_ins_debug_ts
    ins = ins_channel.get_ins()
    # print(ins.sequence_num)
    if ins.sequence_num == 0 or ins.sequence_num > 1000000:
        return
    # The native channel returns a sequence-1 sentinel whose nested position
    # storage is not initialized while the simulator role is still preparing.
    # run1.py discards it before touching position.x/y.
    if ins.sequence_num == 1:
        return
    ins_sequence = int(ins.sequence_num)
    ins_valid, invalid_reason, ins_xy = _ins_sample_status(ins)
    if not ins_valid:
        invalid_ins_count += 1
        _warn_invalid_ins(ins_sequence, ins_xy, invalid_reason)
        return

    invalid_ins_count = 0
    if should_reject_pre_first_ins(ins_sequence, ins_xy):
        return
    if should_reject_nonmonotonic_ins(ins_sequence, ins_xy):
        return

    if (
        last_ins_sequence is not None
        and ins_sequence == int(last_ins_sequence)
        and ins_xy is not None
        and last_ins_position is not None
        and math.hypot(ins_xy[0] - last_ins_position[0], ins_xy[1] - last_ins_position[1]) < 1e-4
    ):
        duplicate_ins_count += 1
        _warn_duplicate_ins(ins_sequence, ins_xy)
        return

    duplicate_ins_count = 0
    last_ins_sequence = ins_sequence
    last_ins_position = ins_xy
    if not model.update_ego(ins):
        return

    if not first_ins_ready:
        first_ins_ready = True
        first_ins_ready_ts = time.time()
        if episode_initial_speed is None:
            initialize_final_control_from_ego(
                float(getattr(model.ego, "speed", 0.0))
            )
        print(
            "[prepare-timing] first_ins_ready "
            f"wall_time={first_ins_ready_ts:.3f} "
            f"ins_seq={last_ins_sequence}"
        )
    check_plan_start_imu(model.ego, last_ins_sequence)
    now = time.monotonic()
    if DEBUG_DRIVE and now - last_drive_ins_debug_ts >= 0.5:
        last_drive_ins_debug_ts = now
        start_distance = float("nan")
        if plan_start_check_xy is not None:
            start_distance = math.hypot(
                float(model.ego.x) - plan_start_check_xy[0],
                float(model.ego.y) - plan_start_check_xy[1],
            )
        print(
            "[drive-debug][ins] "
            f"seq={last_ins_sequence} "
            f"pos=({model.ego.x:.3f},{model.ego.y:.3f}) "
            f"heading={math.degrees(model.ego.theta):.3f}deg "
            f"speed={model.ego.speed:.3f}m/s "
            f"distance_to_init={start_distance:.3f}m"
        )


        #这个地方可能会缓存上回合遗留的自车位置信息。

def ins_receiver_loop():
    """Prime and consume the single INS channel like the known-good run1.py."""
    print(
        "[ins-thread] receiver started mode=run1-compatible "
        "consumer=single",
        flush=True,
    )
    while not ins_receiver_stop_event.is_set():
        # Poll after ActorPrepare establishes the init-state gate, but before
        # ActorPrepareResult permits NT_START_TEST. The early get_ins() calls
        # prime the native subscription and discard its sequence-1 sentinel.
        if not recv_prepare or ins_start_gate_xy is None:
            ins_receiver_stop_event.wait(0.005)
            continue
        try:
            get_vehicle_pose()
            ins_receiver_stop_event.wait(0.001)
        except Exception as exc:
            print(
                f"[ins-thread][WARN] receive/update failed: {exc}",
                flush=True,
            )
            ins_receiver_stop_event.wait(0.01)


def start_ins_receiver_thread():
    thread = threading.Thread(
        target=ins_receiver_loop,
        name="ins-receiver",
        daemon=True,
    )
    thread.start()
    return thread


def abort_test():
    global session_id
    global role_id
    global local_abort_pending
    global local_abort_sent_ts
    print(99999999999999999999999999,'outttttttttttttttttttttttt')
    if drive_trace_logger is not None:
        drive_trace_logger.record_event(
            "local_abort_requested",
            metadata={"role_id": role_id},
        )
    local_abort_pending = True
    local_abort_sent_ts = time.time()
    notify = Notify()
    notify.header.send_ts = int(time.time() * 1000)
    notify.session_id = session_id
    notify.role_id = role_id
    notify.type = NT_ABORT_TEST
    data = notify.SerializeToString()
    length = len(data)
    notify_channel.put(MT_NOTIFY, length, data)

def create_runtime_channels(
    param,
    required_names,
    retry_interval=1.0,
    startup_timeout=0.0,
):
    """Create the native channel set exactly once, matching run_ori.py."""
    del retry_interval, startup_timeout
    required_names = set(required_names)
    channels = libMulticastNetwork.ChannelPtrVector()
    channel_map = {}
    try:
        ret = libMulticastNetwork.create_channels(param, channels)
    except Exception as exc:
        raise RuntimeError(
            "single multicast channel creation raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if ret:
        raise RuntimeError(
            f"single multicast channel creation failed: ret={ret!r}"
        )
    for channel in channels:
        channel_map[channel.name()] = channel
    missing = sorted(required_names.difference(channel_map))
    if missing:
        raise RuntimeError(
            "single multicast channel creation returned an incomplete set: "
            f"missing={','.join(missing)} "
            f"available={','.join(sorted(channel_map)) or '-'}"
        )
    print(
        "[channel-startup] ready single_create=true "
        f"channels={','.join(sorted(channel_map))}"
    )
    return channels, channel_map


def main():
    global map_name
    global pre_map_name
    global loop_count
    global map_ready
    global map_loading
    global prepare_not_before_ts
    global first_control_sent
    global first_control_sent_ts
    global last_prepare_wait_warn_ts
    global last_start_wait_warn_ts

    while 1:
        loop_count += 1
        # ActorPrepare is the authoritative session transition. Poll it first
        # and on every iteration so a new session cannot remain queued behind
        # a stale notification from the previous episode.
        get_prepare()
        notify = process_notify()
        if not recv_prepare:
            # Match run_ori.py: do not consume lidar/INS before ActorPrepare.
            # Some native channel builds do not replay a frame once a client
            # has drained it before the simulator role becomes active.
            get_prepare()
            reconnect_if_prepare_subscription_is_dead()
            now = time.time()
            if now - last_prepare_wait_warn_ts >= 2.0:
                last_prepare_wait_warn_ts = now
                print(
                    "[message-loop][WAIT] ActorPrepare not received; "
                    "notify/prepare are the only channels being polled"
                )
            time.sleep(0.1)
            continue
        if recv_prepare and not start_test:
            # Keep sensor queues untouched until NT_START_TEST. Only the
            # prepare/notify handshake is active in this state.
            if not map_ready or map_loading:
                now = time.time()
                if now - last_start_wait_warn_ts >= 2.0:
                    last_start_wait_warn_ts = now
                    print(
                        "[message-loop][WAIT] preparing map or waiting "
                        "for NT_START_TEST"
                    )
                time.sleep(1.0)
                continue
            if prepare_not_before_ts is not None:
                settle_remaining = prepare_not_before_ts - time.time()
                if settle_remaining > 0.0:
                    time.sleep(min(0.1, settle_remaining))
                    continue
            now = time.time()
            if should_send_prepare_result(now):
                if not publish_first_control_before_prepare():
                    time.sleep(0.03)
                    continue
                prepare()
                pre_map_name = map_name
            time.sleep(0.1)
            continue
        # The run1-compatible receiver thread is the only INS consumer. It is
        # already active before NT_START_TEST, so do not call get_ins() here.
        if hold_until_ego_ready():
            get_vehicle_feedback()
            time.sleep(0.02)
            continue
        if hold_until_global_plan_ready():
            get_vehicle_feedback()
            time.sleep(0.02)
            continue
        if DEBUG_SYNC and model.ego is not None:
            print(
                "[sync-debug] main loop "
                f"wall_time={time.time():.3f} "
                f"loop_count={loop_count} "
                f"ins_seq={last_ins_sequence} "
                f"ego=({model.ego.x}, {model.ego.y}) "
                f"speed={model.ego.speed}"
            )
        control_cycle_started = time.monotonic()
        # get_image()
        drain_sensor_queues(drain_pointcloud=False)
        cmd, done_out = get_pointcloud_msg()
        
        if done_out:
            print("done_out is True, aborting test")
            abort_test()
        if cmd is not None:
            if not first_control_sent:
                first_control_sent = True
                first_control_sent_ts = time.time()
                print(
                    "[prepare-timing] first_control_sent "
                    f"wall_time={first_control_sent_ts:.3f}"
                )
            if DEBUG_SYNC:
                print(
                    "[sync-debug] send control "
                    f"wall_time={time.time():.3f} "
                    f"loop_count={loop_count} "
                    f"acc={cmd.acc} "
                    f"speed={cmd.speed} "
                    f"steer={cmd.steer}"
                )
            # print(cmd.steer)
            # print('cmd.acc, cmd.speed, cmd.steer',cmd.acc, cmd.speed, cmd.steer)
            send_control_cmd(cmd.acc, cmd.speed, cmd.steer)
        get_vehicle_feedback()
        cycle_remaining = CONTROL_LOOP_PERIOD - (time.monotonic() - control_cycle_started)
        if cycle_remaining > 0.0:
            time.sleep(cycle_remaining)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_center", type=str, default="47.110.233.70:52009")
    arg_parser.add_argument("--field_id", type=str, default="field-zd-test1-22-0331134113-888")
    arg_parser.add_argument("--net_interface", type=str, default="usb0")
    arg_parser.add_argument(
        "--channel-startup-timeout",
        type=float,
        default=0.0,
        help=(
            "deprecated compatibility option; channel creation is attempted "
            "once, matching run_ori.py"
        ),
    )
    arg_parser.add_argument(
        "--channel-retry-interval",
        type=float,
        default=1.0,
        help=(
            "deprecated compatibility option; automatic channel retries "
            "are disabled"
        ),
    )
    arg_parser.add_argument(
        "--max_speed",
        "--max-speed",
        dest="max_speed",
        type=float,
        default=None,
        help="maximum planned vehicle speed in m/s",
    )
    arg_parser.add_argument(
        "--max_speed_kmh",
        "--max-speed-kmh",
        dest="max_speed_kmh",
        type=float,
        default=None,
        help="maximum planned vehicle speed in km/h",
    )
    arg_parser.add_argument(
        "--expected_speed_kmh",
        "--expected-speed-kmh",
        dest="expected_speed_kmh",
        type=float,
        default=None,
        help=(
            "explicit evaluator expected speed in km/h; overrides prepare "
            "data and the map-category fallback"
        ),
    )
    arg_parser.add_argument(
        "--use_xodr_expected_speed",
        "--use-xodr-expected-speed",
        dest="use_xodr_expected_speed",
        action="store_true",
        help=(
            "use median XODR speed only when prepare has no expected speed; "
            "off by default because legal and evaluator speeds may differ"
        ),
    )
    arg_parser.add_argument(
        "--route_cache_dir",
        "--route-cache-dir",
        dest="route_cache_dir",
        type=str,
        default=os.environ.get(
            "E2E_GLOBAL_ROUTE_CACHE_DIR",
            os.path.join(
                os.path.dirname(__file__),
                "global_route_cache",
            ),
        ),
        help="persistent XODR and parsed global-route cache directory",
    )
    arg_parser.add_argument(
        "--scenario_overrides",
        "--scenario-overrides",
        dest="scenario_overrides",
        type=str,
        default=os.environ.get(
            "RULE_SCENARIO_OVERRIDES",
            os.path.join(
                os.path.dirname(__file__),
                "scenario_overrides.json",
            ),
        ),
        help=(
            "JSON file containing enabled per-map/per-s manual d/v "
            "overrides; active rules bypass planning collision checks"
        ),
    )
    arg_parser.add_argument(
        "--no_route_cache",
        "--no-route-cache",
        dest="no_route_cache",
        action="store_true",
        help="disable persistent XODR/global-route caching",
    )
    comfort_group = arg_parser.add_mutually_exclusive_group()
    comfort_group.add_argument(
        "--comfort_mode",
        "--comfort-mode",
        dest="comfort_mode",
        action="store_true",
        help=(
            "use a score-safe profile: commanded acceleration +2.8/-2.6 "
            "m/s^2 and longitudinal jerk 4.5 m/s^3, with evaluator lateral "
            "limits 0.5 m/s^2, 1 m/s^3 and yaw rate 0.5 rad/s"
        ),
    )
    comfort_group.add_argument(
        "--no_comfort_mode",
        "--no-comfort-mode",
        dest="comfort_mode",
        action="store_false",
        help="disable evaluator comfort limits and use aggressive tuning",
    )
    arg_parser.set_defaults(
        comfort_mode=(
            os.environ.get("RULE_COMFORT_MODE", "1") != "0"
        )
    )
    arg_parser.add_argument(
        "--override_map_speed_limit",
        "--override-map-speed-limit",
        dest="override_map_speed_limit",
        action="store_true",
        help=(
            "use the configured maximum speed instead of scene/path "
            "speed limits; curve and destination safety caps remain"
        ),
    )
    arg_parser.add_argument(
        "--respect_path_speed_limit",
        "--respect-path-speed-limit",
        dest="respect_path_speed_limit",
        action="store_true",
        help=(
            "respect speed limits encoded in global-path points; disabled "
            "by default because the rule planner applies its own curve cap"
        ),
    )
    arg_parser.add_argument(
        "--stop_at_goal",
        "--stop-at-goal",
        dest="stop_at_goal",
        action="store_true",
        help=(
            "decelerate and stop at the final global-path point; disabled "
            "by default"
        ),
    )
    arg_parser.add_argument(
        "--goal_decel",
        "--goal-decel",
        dest="goal_decel",
        type=float,
        default=None,
        help=(
            "destination braking deceleration in m/s^2 when "
            "--stop_at_goal is enabled"
        ),
    )
    arg_parser.add_argument(
        "--strict_alignment_speed_guard",
        "--strict-alignment-speed-guard",
        dest="strict_alignment_speed_guard",
        action="store_true",
        help=(
            "restore conservative speed caps for heading/lateral tracking "
            "errors"
        ),
    )
    arg_parser.add_argument(
        "--follow_time_headway",
        "--follow-time-headway",
        dest="follow_time_headway",
        type=float,
        default=None,
        help="desired following time headway in seconds",
    )
    arg_parser.add_argument(
        "--minimum_gap",
        "--minimum-gap",
        dest="minimum_gap",
        type=float,
        default=None,
        help="minimum stopped bumper-to-bumper gap in metres",
    )
    arg_parser.add_argument(
        "--max_accel",
        "--max-accel",
        dest="max_accel",
        type=float,
        default=None,
        help="maximum positive acceleration in m/s^2",
    )
    arg_parser.add_argument(
        "--max_decel",
        "--max-decel",
        dest="max_decel",
        type=float,
        default=None,
        help="maximum ordinary braking deceleration in m/s^2",
    )
    arg_parser.add_argument(
        "--max_lon_jerk",
        "--max-lon-jerk",
        dest="max_lon_jerk",
        type=float,
        default=None,
        help="maximum longitudinal jerk in m/s^3",
    )
    arg_parser.add_argument(
        "--max_lat_speed",
        "--max-lat-speed",
        dest="max_lat_speed",
        type=float,
        default=None,
        help="maximum Frenet lateral speed in m/s",
    )
    arg_parser.add_argument(
        "--max_lat_accel",
        "--max-lat-accel",
        dest="max_lat_accel",
        type=float,
        default=None,
        help="maximum Frenet lateral acceleration in m/s^2",
    )
    arg_parser.add_argument(
        "--max_lat_jerk",
        "--max-lat-jerk",
        dest="max_lat_jerk",
        type=float,
        default=None,
        help="maximum Frenet lateral jerk in m/s^3",
    )
    arg_parser.add_argument(
        "--max_curvature",
        "--max-curvature",
        dest="max_curvature",
        type=float,
        default=None,
        help="maximum trajectory curvature in 1/m",
    )
    arg_parser.add_argument(
        "--max_cartesian_lat_accel",
        "--max-cartesian-lat-accel",
        dest="max_cartesian_lat_accel",
        type=float,
        default=None,
        help=(
            "maximum Cartesian lateral acceleration used for curve "
            "speed limits in m/s^2"
        ),
    )
    arg_parser.add_argument(
        "--max_yaw_rate",
        "--max-yaw-rate",
        dest="max_yaw_rate",
        type=float,
        default=None,
        help="maximum yaw rate used for curve speed limits in rad/s",
    )
    arg_parser.add_argument(
        "--curve_speed_factor",
        "--curve-speed-factor",
        dest="curve_speed_factor",
        type=float,
        default=None,
        help=(
            "multiplier applied to curvature-derived speed limits"
        ),
    )
    arg_parser.add_argument(
        "--centerline_feedback_gain",
        "--centerline-feedback-gain",
        dest="centerline_feedback_gain",
        type=float,
        default=None,
        help="direct global-path lateral feedback gain",
    )
    arg_parser.add_argument(
        "--centerline_natural_frequency",
        "--centerline-natural-frequency",
        dest="centerline_natural_frequency",
        type=float,
        default=None,
        help=(
            "global-path lateral recovery frequency in rad/s; "
            "higher values return to d=0 faster"
        ),
    )
    arg_parser.add_argument(
        "--centerline_damping_ratio",
        "--centerline-damping-ratio",
        dest="centerline_damping_ratio",
        type=float,
        default=None,
        help=(
            "global-path lateral damping ratio; 1.0 is critically damped"
        ),
    )
    arg_parser.add_argument(
        "--steering_ratio",
        "--steering-ratio",
        dest="steering_ratio",
        type=float,
        default=None,
        help=(
            "DriveSim steering-wheel command to effective front-wheel "
            "angle ratio; Cam6 default is 1.65"
        ),
    )
    arg_parser.add_argument(
        "--enable_centerline_safety_stop",
        "--enable-centerline-safety-stop",
        dest="enable_centerline_safety_stop",
        action="store_true",
        help=(
            "enable legacy emergency braking when global-path "
            "offset grows; disabled by default"
        ),
    )
    arg_parser.add_argument(
        "--avoidance_speed",
        "--avoidance-speed",
        dest="avoidance_speed",
        type=float,
        default=None,
        help="maximum target speed while bypassing a static obstacle in m/s",
    )
    arg_parser.add_argument(
        "--avoidance_half_width",
        "--avoidance-half-width",
        dest="avoidance_half_width",
        type=float,
        default=None,
        help="maximum absolute Frenet lateral offset for static avoidance",
    )
    arg_parser.add_argument(
        "--minimum_bypass_shift",
        "--minimum-bypass-shift",
        dest="minimum_bypass_shift",
        type=float,
        default=None,
        help=(
            "minimum actual lateral displacement when bypassing a "
            "static obstacle in metres"
        ),
    )
    arg_parser.add_argument(
        "--static_side_clearance",
        "--static-side-clearance",
        dest="static_side_clearance",
        type=float,
        default=None,
        help=(
            "minimum oriented-box edge clearance beside a static "
            "obstacle in metres"
        ),
    )
    arg_parser.add_argument(
        "--lead_lateral_tolerance",
        "--lead-lateral-tolerance",
        dest="lead_lateral_tolerance",
        type=float,
        default=None,
        help=(
            "maximum obstacle-centre lateral offset treated as a lead "
            "vehicle in metres"
        ),
    )
    arg_parser.add_argument(
        "--static_avoidance_min_hits",
        "--static-avoidance-min-hits",
        dest="static_avoidance_min_hits",
        type=int,
        default=None,
        help=(
            "consecutive tracker hits required before full static "
            "obstacle avoidance"
        ),
    )
    arg_parser.add_argument(
        "--static_avoidance_trigger_distance",
        "--static-avoidance-trigger-distance",
        dest="static_avoidance_trigger_distance",
        type=float,
        default=None,
        help=(
            "minimum look-ahead distance that activates static "
            "obstacle avoidance in metres"
        ),
    )
    arg_parser.add_argument(
        "--lateral_sample_step",
        "--lateral-sample-step",
        dest="lateral_sample_step",
        type=float,
        default=None,
        help="spacing between sampled lateral targets in metres",
    )
    arg_parser.add_argument(
        "--collision_margin",
        "--collision-margin",
        dest="collision_margin",
        type=float,
        default=None,
        help="hard collision envelope margin in metres; minimum 0.40",
    )
    arg_parser.add_argument(
        "--ignore_obstacles",
        "--ignore-obstacles",
        dest="ignore_obstacles",
        action="store_true",
        help="ignore all perceived obstacles (unsafe; debugging only)",
    )
    sprint_group = arg_parser.add_mutually_exclusive_group()
    sprint_group.add_argument(
        "--sprint",
        dest="sprint_enabled",
        action="store_true",
        help=(
            "force unrestricted direct-goal sprint on every map and bypass "
            "obstacle decisions"
        ),
    )
    sprint_group.add_argument(
        "--no_sprint",
        "--no-sprint",
        dest="sprint_enabled",
        action="store_false",
        help="disable generic direct-goal sprint",
    )
    arg_parser.set_defaults(sprint_enabled=None)
    arg_parser.add_argument(
        "--sprint_maps",
        "--sprint-maps",
        dest="sprint_maps",
        type=str,
        default=None,
        help=(
            "comma-separated map basenames using sprint, or '*' for all; "
            "used only by environment/library selective mode; --sprint "
            "always forces '*'"
        ),
    )
    arg_parser.add_argument(
        "--sprint_speed",
        "--sprint-speed",
        dest="sprint_speed",
        type=float,
        default=None,
        help="generic sprint target speed in m/s",
    )
    arg_parser.add_argument(
        "--sprint_accel",
        "--sprint-accel",
        dest="sprint_accel",
        type=float,
        default=None,
        help="generic sprint acceleration pulse in m/s^2",
    )
    arg_parser.add_argument(
        "--sprint_alignment_duration",
        "--sprint-alignment-duration",
        dest="sprint_alignment_duration",
        type=float,
        default=None,
        help=(
            "stationary steering-alignment duration before the one-frame "
            "sprint acceleration pulse, seconds"
        ),
    )
    arg_parser.add_argument(
        "--sprint_pulse_min_duration",
        "--sprint-pulse-min-duration",
        dest="sprint_pulse_min_duration",
        type=float,
        default=None,
        help=(
            "minimum time to republish the logical sprint pulse so the "
            "chassis consumes it, seconds"
        ),
    )
    arg_parser.add_argument(
        "--sprint_pulse_ack_speed",
        "--sprint-pulse-ack-speed",
        dest="sprint_pulse_ack_speed",
        type=float,
        default=None,
        help=(
            "INS speed that acknowledges sprint pulse consumption, m/s"
        ),
    )
    arg_parser.add_argument(
        "--sprint_max_steer_deg",
        "--sprint-max-steer-deg",
        dest="sprint_max_steer_deg",
        type=float,
        default=None,
        help="generic sprint steering-wheel angle cap in degrees",
    )
    arg_parser.add_argument(
        "--sprint_recovery_speed",
        "--sprint-recovery-speed",
        dest="sprint_recovery_speed",
        type=float,
        default=None,
        help="speed cap while recovering a divergent sprint in m/s",
    )
    arg_parser.add_argument(
        "--sprint_ignore_obstacles",
        "--sprint-ignore-obstacles",
        dest="sprint_ignore_obstacles",
        action="store_true",
        default=None,
        help="bypass obstacle and missing-GT braking only during sprint",
    )
    arg_parser.add_argument(
        "--roundabout_mode",
        "--roundabout-mode",
        dest="roundabout_mode",
        choices=("off", "follow", "direct"),
        default=os.environ.get(
            "RULE_ROUNDABOUT_MODE", "off"
        ).strip().lower(),
        help=(
            "exclusive roundabout controller selected at launch: "
            "follow cruises on a clear global path and applies IDM to "
            "the nearest path vehicle; direct ignores traffic"
        ),
    )
    arg_parser.add_argument(
        "--roundabout_gap",
        "--roundabout-gap",
        dest="roundabout_gap",
        type=float,
        default=float(
            os.environ.get("RULE_ROUNDABOUT_GAP", "1.0")
        ),
        help="IDM minimum bumper-to-bumper gap in metres",
    )
    arg_parser.add_argument(
        "--roundabout_max_speed",
        "--roundabout-max-speed",
        dest="roundabout_max_speed",
        type=float,
        default=float(
            os.environ.get("RULE_ROUNDABOUT_MAX_SPEED", "30.0")
        ),
        help="roundabout direct/follow speed ceiling in m/s",
    )
    arg_parser.add_argument(
        "--roundabout_max_accel",
        "--roundabout-max-accel",
        dest="roundabout_max_accel",
        type=float,
        default=float(
            os.environ.get("RULE_ROUNDABOUT_MAX_ACCEL", "20.0")
        ),
        help="roundabout longitudinal acceleration limit in m/s^2",
    )
    arg_parser.add_argument(
        "--roundabout_max_decel",
        "--roundabout-max-decel",
        dest="roundabout_max_decel",
        type=float,
        default=float(
            os.environ.get("RULE_ROUNDABOUT_MAX_DECEL", "15.5")
        ),
        help="roundabout braking magnitude limit in m/s^2",
    )
    arg_parser.add_argument(
        "--roundabout_follow_max_accel",
        "--roundabout-follow-max-accel",
        dest="roundabout_follow_max_accel",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_FOLLOW_MAX_ACCEL", "8.0"
            )
        ),
        help="follow-mode acceleration limit in m/s^2",
    )
    arg_parser.add_argument(
        "--roundabout_lane_half_width",
        "--roundabout-lane-half-width",
        dest="roundabout_lane_half_width",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_LANE_HALF_WIDTH", "1.0"
            )
        ),
        help="vehicle is on global path only when abs(Frenet d) is below this",
    )
    arg_parser.add_argument(
        "--roundabout_time_headway",
        "--roundabout-time-headway",
        dest="roundabout_time_headway",
        type=float,
        default=float(
            os.environ.get("RULE_ROUNDABOUT_TIME_HEADWAY", "0.8")
        ),
        help="IDM dynamic time headway in seconds",
    )
    arg_parser.add_argument(
        "--roundabout_idm_comfort_decel",
        "--roundabout-idm-comfort-decel",
        dest="roundabout_idm_comfort_decel",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_IDM_COMFORT_DECEL", "6.0"
            )
        ),
        help="IDM normal braking parameter in m/s^2",
    )
    arg_parser.add_argument(
        "--roundabout_ttc_threshold",
        "--roundabout-ttc-threshold",
        dest="roundabout_ttc_threshold",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_TTC_THRESHOLD", "1.0"
            )
        ),
        help="TTC threshold for emergency maximum braking in seconds",
    )
    arg_parser.add_argument(
        "--roundabout_curve_lat_accel",
        "--roundabout-curve-lat-accel",
        dest="roundabout_curve_lat_accel",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_CURVE_LAT_ACCEL", "4.0"
            )
        ),
        help="lateral-acceleration budget used for curve speed in m/s^2",
    )
    arg_parser.add_argument(
        "--roundabout_curve_lookahead_time",
        "--roundabout-curve-lookahead-time",
        dest="roundabout_curve_lookahead_time",
        type=float,
        default=float(
            os.environ.get(
                "RULE_ROUNDABOUT_CURVE_LOOKAHEAD_TIME", "2.0"
            )
        ),
        help="speed-scaled curvature look-ahead time in seconds",
    )
    arg_parser.add_argument(
        "--pedestrian_conf",
        "--pedestrian-conf",
        dest="pedestrian_conf",
        type=float,
        default=None,
        help="pedestrian detector confidence threshold, range 0..1",
    )
    arg_parser.add_argument(
        "--cyclist_conf",
        "--cyclist-conf",
        dest="cyclist_conf",
        type=float,
        default=None,
        help="cyclist/motorcycle detector confidence threshold, range 0..1",
    )
    arg_parser.add_argument(
        "--vehicle_conf",
        "--vehicle-conf",
        dest="vehicle_conf",
        type=float,
        default=None,
        help="vehicle detector confidence threshold, range 0..1",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="enable rate-limited INS, planning, control, and chassis diagnostics",
    )
    arg_parser.add_argument(
        "--perception_source",
        "--perception-source",
        dest="perception_source",
        choices=("gt", "lidar", "none"),
        default=os.environ.get(
            "E2E_PERCEPTION_SOURCE", "gt"
        ).strip().lower(),
        help=(
            "planning obstacle source: simulator NPC ground truth "
            "(default), PointPillars lidar detections, or none"
        ),
    )
    arg_parser.add_argument(
        "--gt_track_hold_seconds",
        "--gt-track-hold-seconds",
        dest="gt_track_hold_seconds",
        type=float,
        default=None,
        help=(
            "retain and predict a temporarily missing GT track for this "
            "many seconds; use 0 to disable"
        ),
    )
    arg_parser.add_argument(
        "--gt_startup_grace_seconds",
        "--gt-startup-grace-seconds",
        dest="gt_startup_grace_seconds",
        type=float,
        default=GT_STARTUP_GRACE_SECONDS,
        help=(
            "allow this many seconds after first INS for the first GT "
            "frame before applying the missing-GT brake"
        ),
    )
    arg_parser.add_argument(
        "--gt_innovation_gate_m",
        "--gt-innovation-gate-m",
        dest="gt_innovation_gate_m",
        type=float,
        default=None,
        help=(
            "base GT position innovation gate in metres; use inf to "
            "disable jump rejection"
        ),
    )
    arg_parser.add_argument(
        "--gt_innovation_gate_speed",
        "--gt-innovation-gate-speed",
        dest="gt_innovation_gate_speed",
        type=float,
        default=None,
        help=(
            "additional GT innovation allowance per elapsed second"
        ),
    )
    arg_parser.add_argument(
        "--log_dir",
        type=str,
        default=os.environ.get(
            "E2E_RUN_LOG_DIR",
            os.path.join(os.path.dirname(__file__), "debug_logs"),
        ),
        help="directory for per-run terminal text logs",
    )
    arg_parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="disable automatic terminal output logging",
    )
    arg_parser.add_argument(
        "--trace_period",
        "--trace-period",
        dest="trace_period",
        type=float,
        default=float(os.environ.get("E2E_DRIVE_TRACE_PERIOD", "0.10")),
        help="structured drive trace period in seconds",
    )
    arg_parser.add_argument(
        "--no_drive_trace",
        "--no-drive-trace",
        dest="no_drive_trace",
        action="store_true",
        help="disable structured ego/obstacle/planning/control JSONL logging",
    )
    arg_parser.add_argument(
        "--dump_npc_truth",
        "--dump-npc-truth",
        dest="dump_npc_truth",
        action="store_true",
        default=os.environ.get(
            "E2E_DUMP_NPC_TRUTH", "0"
        ) == "1",
        help=(
            "optionally record decoded simulator NPC frames and lossless "
            "raw bytes; NPC boxes are visualized without this flag"
        ),
    )
    arg_parser.add_argument(
        "--npc_truth_dir",
        "--npc-truth-dir",
        dest="npc_truth_dir",
        type=str,
        default=None,
        help=(
            "NPC truth output directory; defaults to "
            "<log_dir>/npc_truth"
        ),
    )
    args = arg_parser.parse_args()
    if args.channel_startup_timeout < 0.0:
        arg_parser.error("--channel-startup-timeout must be non-negative")
    if args.channel_retry_interval <= 0.0:
        arg_parser.error("--channel-retry-interval must be greater than zero")
    if args.max_speed is not None and args.max_speed <= 0.0:
        arg_parser.error("--max_speed must be greater than zero")
    if args.max_speed_kmh is not None and args.max_speed_kmh <= 0.0:
        arg_parser.error("--max_speed_kmh must be greater than zero")
    if (
        args.expected_speed_kmh is not None
        and args.expected_speed_kmh <= 0.0
    ):
        arg_parser.error(
            "--expected_speed_kmh must be greater than zero"
        )
    if args.max_speed is not None and args.max_speed_kmh is not None:
        arg_parser.error(
            "use only one of --max_speed and --max_speed_kmh"
        )
    if args.goal_decel is not None and args.goal_decel <= 0.0:
        arg_parser.error("--goal_decel must be greater than zero")
    if (
        args.override_map_speed_limit
        and args.max_speed is None
        and args.max_speed_kmh is None
        and not math.isfinite(PlannerConfig().max_speed)
    ):
        arg_parser.error(
            "--override_map_speed_limit requires --max_speed or "
            "--max_speed_kmh"
        )
    if (
        args.follow_time_headway is not None
        and args.follow_time_headway < 0.30
    ):
        arg_parser.error(
            "--follow_time_headway must be at least 0.30 seconds"
        )
    if args.minimum_gap is not None and args.minimum_gap < 1.0:
        arg_parser.error("--minimum_gap must be at least 1.0 metre")
    if args.max_accel is not None and args.max_accel <= 0.0:
        arg_parser.error("--max_accel must be greater than zero")
    for dynamics_name in (
        "max_decel",
        "max_lon_jerk",
        "max_lat_speed",
        "max_lat_accel",
        "max_lat_jerk",
        "max_cartesian_lat_accel",
        "max_yaw_rate",
    ):
        dynamics_value = getattr(args, dynamics_name)
        if dynamics_value is not None and dynamics_value <= 0.0:
            arg_parser.error(
                f"--{dynamics_name} must be greater than zero"
            )
    if (
        args.max_curvature is not None
        and not 0.01 <= args.max_curvature <= 1.0
    ):
        arg_parser.error(
            "--max_curvature must be between 0.01 and 1.0"
        )
    if (
        args.centerline_feedback_gain is not None
        and not 0.0 <= args.centerline_feedback_gain <= 5.0
    ):
        arg_parser.error(
            "--centerline_feedback_gain must be between 0.0 and 5.0"
        )
    if (
        args.centerline_natural_frequency is not None
        and not 0.10 <= args.centerline_natural_frequency <= 2.00
    ):
        arg_parser.error(
            "--centerline_natural_frequency must be between 0.10 and 2.00"
        )
    if (
        args.centerline_damping_ratio is not None
        and not 0.40 <= args.centerline_damping_ratio <= 3.00
    ):
        arg_parser.error(
            "--centerline_damping_ratio must be between 0.40 and 3.00"
        )
    if (
        args.steering_ratio is not None
        and not 0.50 <= args.steering_ratio <= 30.00
    ):
        arg_parser.error(
            "--steering_ratio must be between 0.50 and 30.00"
        )
    if (
        args.curve_speed_factor is not None
        and not 0.1 <= args.curve_speed_factor <= 5.0
    ):
        arg_parser.error(
            "--curve_speed_factor must be between 0.1 and 5.0"
        )
    if args.trace_period < 0.02:
        arg_parser.error("--trace_period must be at least 0.02 seconds")
    for gt_tracking_name in (
        "gt_startup_grace_seconds",
        "gt_track_hold_seconds",
        "gt_innovation_gate_m",
        "gt_innovation_gate_speed",
    ):
        gt_tracking_value = getattr(args, gt_tracking_name)
        if (
            gt_tracking_value is not None
            and (
                math.isnan(gt_tracking_value)
                or gt_tracking_value < 0.0
            )
        ):
            arg_parser.error(
                f"--{gt_tracking_name} must be non-negative"
            )
    if args.avoidance_speed is not None and args.avoidance_speed <= 0.0:
        arg_parser.error("--avoidance_speed must be greater than zero")
    if (
        args.avoidance_half_width is not None
        and args.avoidance_half_width <= 0.5
    ):
        arg_parser.error("--avoidance_half_width must be greater than 0.5")
    if (
        args.minimum_bypass_shift is not None
        and not 0.0 <= args.minimum_bypass_shift <= 2.50
    ):
        arg_parser.error(
            "--minimum_bypass_shift must be between 0.0 and 2.50"
        )
    if (
        args.static_side_clearance is not None
        and not 0.20 <= args.static_side_clearance <= 1.50
    ):
        arg_parser.error(
            "--static_side_clearance must be between 0.20 and 1.50"
        )
    if (
        args.lead_lateral_tolerance is not None
        and args.lead_lateral_tolerance < 0.5
    ):
        arg_parser.error(
            "--lead_lateral_tolerance must be at least 0.5"
        )
    if (
        args.static_avoidance_min_hits is not None
        and args.static_avoidance_min_hits < 1
    ):
        arg_parser.error(
            "--static_avoidance_min_hits must be at least 1"
        )
    if (
        args.static_avoidance_trigger_distance is not None
        and args.static_avoidance_trigger_distance < 6.0
    ):
        arg_parser.error(
            "--static_avoidance_trigger_distance must be at least "
            "6.0 metres"
        )
    if (
        args.lateral_sample_step is not None
        and not 0.15 <= args.lateral_sample_step <= 1.0
    ):
        arg_parser.error(
            "--lateral_sample_step must be between 0.15 and 1.0"
        )
    if (
        args.collision_margin is not None
        and args.collision_margin < 0.40
    ):
        arg_parser.error("--collision_margin must be at least 0.40")
    for sprint_name in (
        "sprint_speed",
        "sprint_accel",
        "sprint_pulse_min_duration",
        "sprint_pulse_ack_speed",
        "sprint_max_steer_deg",
    ):
        sprint_value = getattr(args, sprint_name)
        if sprint_value is not None and sprint_value <= 0.0:
            arg_parser.error(
                f"--{sprint_name} must be greater than zero"
            )
    if (
        args.sprint_recovery_speed is not None
        and args.sprint_recovery_speed < 0.0
    ):
        arg_parser.error(
            "--sprint_recovery_speed must be non-negative"
        )
    if (
        args.sprint_alignment_duration is not None
        and args.sprint_alignment_duration < 0.0
    ):
        arg_parser.error(
            "--sprint_alignment_duration must be non-negative"
        )
    if args.sprint_maps is not None and not args.sprint_maps.strip():
        arg_parser.error("--sprint_maps cannot be empty")
    if args.roundabout_gap < 0.1:
        arg_parser.error("--roundabout_gap must be at least 0.1 metre")
    for roundabout_name in (
        "roundabout_max_speed",
        "roundabout_max_accel",
        "roundabout_max_decel",
        "roundabout_follow_max_accel",
        "roundabout_lane_half_width",
        "roundabout_time_headway",
        "roundabout_idm_comfort_decel",
        "roundabout_ttc_threshold",
        "roundabout_curve_lat_accel",
        "roundabout_curve_lookahead_time",
    ):
        roundabout_value = getattr(args, roundabout_name)
        if roundabout_value <= 0.0:
            arg_parser.error(
                f"--{roundabout_name} must be greater than zero"
            )
    for confidence_name in (
        "pedestrian_conf",
        "cyclist_conf",
        "vehicle_conf",
    ):
        confidence_value = getattr(args, confidence_name)
        if (
            confidence_value is not None
            and not 0.0 <= confidence_value <= 1.0
        ):
            arg_parser.error(
                f"--{confidence_name} must be between zero and one"
            )
    terminal_log = (
        None if args.no_log_file else _setup_terminal_log(args.log_dir)
    )
    if not args.no_drive_trace:
        try:
            drive_trace_logger = DriveTraceLogger(
                args.log_dir,
                period_sec=args.trace_period,
                max_trajectory_points=int(
                    os.environ.get(
                        "E2E_DRIVE_TRACE_TRAJECTORY_POINTS",
                        "30",
                    )
                ),
                max_obstacles=int(
                    os.environ.get(
                        "E2E_DRIVE_TRACE_MAX_OBSTACLES",
                        "64",
                    )
                ),
            )
            atexit.register(drive_trace_logger.close)
            print(
                "[drive-trace] "
                f"archive={drive_trace_logger.archive_path}"
            )
            print(
                "[drive-trace] "
                f"latest={drive_trace_logger.latest_path} "
                f"period={drive_trace_logger.period_sec:.3f}s"
            )
        except Exception as exc:
            drive_trace_logger = None
            print(f"[drive-trace][WARN] disabled: {exc}")
    if args.debug:
        DEBUG_DRIVE = True
        os.environ["E2E_DEBUG_OBS"] = "1"
        print("[drive-debug] enabled by --debug")
    PERCEPTION_SOURCE = args.perception_source
    os.environ["E2E_PERCEPTION_SOURCE"] = PERCEPTION_SOURCE
    GT_STARTUP_GRACE_SECONDS = float(
        args.gt_startup_grace_seconds
    )
    EXPECTED_SPEED_CLI_MPS = (
        None
        if args.expected_speed_kmh is None
        else float(args.expected_speed_kmh) / 3.6
    )
    USE_XODR_EXPECTED_SPEED = bool(
        args.use_xodr_expected_speed
    )
    print(f"[perception] planning_source={PERCEPTION_SOURCE.upper()}")
    print(
        "[algorithm-policy] "
        f"version={ALGORITHM_POLICY_VERSION}"
    )
    if args.gt_track_hold_seconds is not None:
        gt_obstacle_adapter.track_hold_seconds = float(
            args.gt_track_hold_seconds
        )
    if args.gt_innovation_gate_m is not None:
        gt_obstacle_adapter.innovation_gate_m = float(
            args.gt_innovation_gate_m
        )
    if args.gt_innovation_gate_speed is not None:
        gt_obstacle_adapter.innovation_gate_speed = float(
            args.gt_innovation_gate_speed
        )
    print(
        "[gt-tracker-config] "
        f"startup_grace={GT_STARTUP_GRACE_SECONDS:.3f}s "
        f"hold={gt_obstacle_adapter.track_hold_seconds:.3f}s "
        f"innovation_gate="
        f"{gt_obstacle_adapter.innovation_gate_m:.3f}m+"
        f"{gt_obstacle_adapter.innovation_gate_speed:.3f}m/s*dt"
    )
    confidence_env = {
        "pedestrian_conf": "PERCEPTION_PEDESTRIAN_CONF",
        "cyclist_conf": "PERCEPTION_CYCLIST_CONF",
        "vehicle_conf": "PERCEPTION_VEHICLE_CONF",
    }
    for confidence_name, env_name in confidence_env.items():
        confidence_value = getattr(args, confidence_name)
        if confidence_value is not None:
            os.environ[env_name] = str(float(confidence_value))
    param = libMulticastNetwork.CreateChannelsParam()

    local_ip = get_ip_address(args.net_interface)

    #######################################################
    ###################### 需要修改 ########################
    param.config_center_addr = args.config_center  # 组播配置中心的ip
    param.local_ip = local_ip  # 本机ip
    param.net_interface_name = args.net_interface  # 本机网卡
    param.field_id = (
        args.field_id
    )  # 唯一的场地id，可以任意字符串，需要和daemon和simulator一致
    #######################################################

    param.log_level = 1  # 1-info, 2-warning, 3-error， 设置不同的组播日志等级
    param.client_name = MULTICAST_CLIENT_NAME
    param.recv_self_msg = False
    session_id = ""
    print(
        "[network-config] "
        f"config_center={args.config_center} "
        f"field_id={args.field_id} "
        f"interface={args.net_interface} "
        f"local_ip={local_ip} "
        f"client_name={param.client_name}"
    )

    required_channel_names = {
        "lidar",
        "notify",
        "vehiclecontrol",
        "prepare",
        "ins",
        "camera",
    }
    if PERCEPTION_SOURCE == "gt":
        required_channel_names.add("npc")
    try:
        channels, channel_map = create_runtime_channels(
            param,
            required_channel_names,
            retry_interval=args.channel_retry_interval,
            startup_timeout=args.channel_startup_timeout,
        )
    except RuntimeError as exc:
        print(f"[channel-startup][FATAL] {exc}")
        sys.exit(1)
    channel_created_ts = time.time()
    # 不同的组播消息通道，用于接收和发送消息
    for c in channel_map.values():
        print("message channel name: {}, id: {}".format(c.name(), c.id()))

    pointcloud_channel = channel_map["lidar"]
    notify_channel = channel_map["notify"]
    cmd_channel = channel_map["vehiclecontrol"]
    prepare_channel = channel_map["prepare"]
    ins_channel = channel_map["ins"]
    image_channel = channel_map["camera"]
    npc_channel = (
        channel_map.get("npc")
        if NPC_TRUTH_ENABLED
        else None
    )
    if not NPC_TRUTH_ENABLED:
        print(
            "[npc-truth] disabled by isolated entrypoint; "
            "background vehicles are not read or decoded"
        )
    if PERCEPTION_SOURCE == "gt" and npc_channel is None:
        print(
            "[perception][FATAL] GT planning requested but channel "
            "'npc' is unavailable"
        )
        sys.exit(2)
    if args.dump_npc_truth:
        if npc_channel is None:
            print(
                "[npc-truth][WARN] --dump_npc_truth requested "
                "but channel 'npc' is unavailable"
            )
        else:
            npc_truth_dir = (
                args.npc_truth_dir
                if args.npc_truth_dir
                else os.path.join(args.log_dir, "npc_truth")
            )
            npc_truth_recorder = _NpcTruthRecorder(
                npc_truth_dir
            )

    if not libMulticastNetwork.InitImageDecoder(6,1600,900):
        print("image decoder init error")
        sys.exit(1)

    img_id = 0
    save_results = False
    recv_prepare = False
    start_test = False
    actor_id = MULTICAST_CLIENT_NAME
    role_id = MULTICAST_CLIENT_NAME

    # Pure Python closed loop: detector -> rule planner -> stable controller.
    model = Predictor()
    global_route_planner = DirectGlobalRoutePlanner(
        map_search_dirs=[
            os.path.join(os.path.dirname(__file__), "maps")
        ],
        persistent_cache_dir=(
            None
            if args.no_route_cache
            else args.route_cache_dir
        ),
    )
    print(
        "[global-plan-cache] "
        f"enabled={not args.no_route_cache} "
        f"dir={os.path.abspath(args.route_cache_dir)}"
    )
    planner_config = PlannerConfig()
    planner_config.scenario_overrides_path = os.path.abspath(
        os.path.expanduser(args.scenario_overrides)
    )
    if args.comfort_mode:
        # Apply the scoring profile first; explicit command-line dynamics
        # below remain valid expert overrides.
        planner_config.enable_comfort_mode()
    if args.max_speed is not None:
        planner_config.max_speed = float(args.max_speed)
    if args.max_speed_kmh is not None:
        planner_config.max_speed = float(args.max_speed_kmh) / 3.6
    if args.override_map_speed_limit:
        planner_config.override_map_speed_limit = True
    if args.respect_path_speed_limit:
        planner_config.respect_path_speed_limit = True
    if args.stop_at_goal:
        planner_config.stop_at_goal = True
    if args.goal_decel is not None:
        planner_config.goal_decel = float(args.goal_decel)
    if args.strict_alignment_speed_guard:
        planner_config.strict_alignment_speed_guard = True
    if args.follow_time_headway is not None:
        planner_config.time_headway = float(args.follow_time_headway)
    if args.minimum_gap is not None:
        planner_config.minimum_gap = float(args.minimum_gap)
    if args.max_accel is not None:
        planner_config.max_accel = float(args.max_accel)
    if args.max_decel is not None:
        planner_config.max_decel = float(args.max_decel)
    if args.max_lon_jerk is not None:
        planner_config.max_lon_jerk = float(args.max_lon_jerk)
    if args.max_lat_speed is not None:
        planner_config.max_lat_speed = float(args.max_lat_speed)
    if args.max_lat_accel is not None:
        planner_config.max_lat_accel = float(args.max_lat_accel)
    if args.max_lat_jerk is not None:
        planner_config.max_lat_jerk = float(args.max_lat_jerk)
    if args.max_curvature is not None:
        planner_config.max_curvature = float(args.max_curvature)
    if args.max_cartesian_lat_accel is not None:
        planner_config.max_lateral_accel = float(
            args.max_cartesian_lat_accel
        )
    if args.max_yaw_rate is not None:
        planner_config.max_yaw_rate = float(args.max_yaw_rate)
    if args.curve_speed_factor is not None:
        planner_config.curve_speed_factor = float(
            args.curve_speed_factor
        )
    if args.centerline_feedback_gain is not None:
        planner_config.centerline_feedback_gain = float(
            args.centerline_feedback_gain
        )
    if args.centerline_natural_frequency is not None:
        planner_config.centerline_natural_frequency = float(
            args.centerline_natural_frequency
        )
    if args.centerline_damping_ratio is not None:
        planner_config.centerline_damping_ratio = float(
            args.centerline_damping_ratio
        )
    if args.steering_ratio is not None:
        planner_config.steering_ratio = float(
            args.steering_ratio
        )
    if args.enable_centerline_safety_stop:
        planner_config.centerline_safety_stop_enabled = True
    if args.avoidance_speed is not None:
        planner_config.static_avoidance_speed = float(
            args.avoidance_speed
        )
    if args.avoidance_half_width is not None:
        planner_config.avoidance_half_width = float(
            args.avoidance_half_width
        )
    if args.minimum_bypass_shift is not None:
        planner_config.minimum_bypass_shift = float(
            args.minimum_bypass_shift
        )
    if args.static_side_clearance is not None:
        planner_config.static_side_clearance = float(
            args.static_side_clearance
        )
    if args.lead_lateral_tolerance is not None:
        planner_config.lead_lateral_tolerance = float(
            args.lead_lateral_tolerance
        )
    if args.static_avoidance_min_hits is not None:
        planner_config.static_avoidance_min_hits = int(
            args.static_avoidance_min_hits
        )
    elif PERCEPTION_SOURCE == "gt":
        # Simulator ground truth has stable role IDs and does not need the
        # detector-oriented two-frame confirmation delay.
        planner_config.static_avoidance_min_hits = 1
    if args.static_avoidance_trigger_distance is not None:
        planner_config.static_avoidance_trigger_distance = float(
            args.static_avoidance_trigger_distance
        )
    if args.lateral_sample_step is not None:
        planner_config.lateral_sample_step = float(
            args.lateral_sample_step
        )
    if args.collision_margin is not None:
        planner_config.collision_margin = float(args.collision_margin)
    if args.ignore_obstacles:
        planner_config.ignore_obstacles = True
    planner_config.configure_sprint(
        enabled=args.sprint_enabled,
        map_names=args.sprint_maps,
        speed=args.sprint_speed,
        accel=args.sprint_accel,
        alignment_duration=args.sprint_alignment_duration,
        pulse_min_duration=args.sprint_pulse_min_duration,
        pulse_ack_speed=args.sprint_pulse_ack_speed,
        max_steer_deg=args.sprint_max_steer_deg,
        recovery_speed=args.sprint_recovery_speed,
        ignore_obstacles=args.sprint_ignore_obstacles,
        unrestricted=(args.sprint_enabled is True),
    )
    if args.sprint_enabled is True:
        print(
            "[sprint-config][UNRESTRICTED] --sprint active: "
            "maps=* obstacle_decisions=BYPASSED"
        )
    print(
        "[planner-config] "
        f"perception_source={PERCEPTION_SOURCE.upper()} "
        f"max_speed={planner_config.max_speed:.3f}m/s "
        f"max_speed_kmh={planner_config.max_speed * 3.6:.3f}km/h "
        f"override_map_speed_limit="
        f"{planner_config.override_map_speed_limit} "
        f"respect_path_speed_limit="
        f"{planner_config.respect_path_speed_limit} "
        f"stop_at_goal={planner_config.stop_at_goal} "
        f"goal_decel={planner_config.goal_decel:.3f}m/s2 "
        f"comfort_mode={planner_config.comfort_mode} "
        f"strict_alignment_speed_guard="
        f"{planner_config.strict_alignment_speed_guard} "
        f"max_accel={planner_config.max_accel:.3f}m/s2 "
        f"max_decel={planner_config.max_decel:.3f}m/s2 "
        f"max_lon_jerk={planner_config.max_lon_jerk:.3f}m/s3 "
        f"max_lat_speed={planner_config.max_lat_speed:.3f}m/s "
        f"max_lat_accel={planner_config.max_lat_accel:.3f}m/s2 "
        f"max_lat_jerk={planner_config.max_lat_jerk:.3f}m/s3 "
        f"max_curvature={planner_config.max_curvature:.4f}1/m "
        f"max_cartesian_lat_accel="
        f"{planner_config.max_lateral_accel:.3f}m/s2 "
        f"max_yaw_rate={planner_config.max_yaw_rate:.3f}rad/s "
        f"curve_speed_factor="
        f"{planner_config.curve_speed_factor:.3f} "
        f"centerline_feedback_gain="
        f"{planner_config.centerline_feedback_gain:.3f} "
        f"centerline_natural_frequency="
        f"{planner_config.centerline_natural_frequency:.3f}rad/s "
        f"centerline_damping_ratio="
        f"{planner_config.centerline_damping_ratio:.3f} "
        f"steering_ratio={planner_config.steering_ratio:.3f} "
        f"scenario_overrides="
        f"{planner_config.scenario_overrides_path} "
        f"centerline_safety_stop_enabled="
        f"{planner_config.centerline_safety_stop_enabled} "
        f"follow_time_headway={planner_config.time_headway:.3f}s "
        f"minimum_gap={planner_config.minimum_gap:.3f}m "
        f"ignore_obstacles={planner_config.ignore_obstacles} "
        f"sprint_enabled={planner_config.sprint_enabled} "
        f"sprint_maps="
        f"{','.join(sorted(planner_config.sprint_map_names)) or 'none'} "
        f"sprint_speed={planner_config.sprint_speed:.3f}m/s "
        f"sprint_accel={planner_config.sprint_accel:.3f}m/s2 "
        f"sprint_alignment_duration="
        f"{planner_config.sprint_alignment_duration:.3f}s "
        f"sprint_pulse_min_duration="
        f"{planner_config.sprint_pulse_min_duration:.3f}s "
        f"sprint_pulse_ack_speed="
        f"{planner_config.sprint_pulse_ack_speed:.3f}m/s "
        f"sprint_steer_cap="
        f"{planner_config.sprint_max_steer_deg:.3f}deg "
        f"sprint_recovery_speed="
        f"{planner_config.sprint_recovery_speed:.3f}m/s "
        f"sprint_ignore_obstacles="
        f"{planner_config.sprint_ignore_obstacles} "
        f"avoidance_speed={planner_config.static_avoidance_speed:.3f}m/s "
        f"avoidance_half_width={planner_config.avoidance_half_width:.3f}m "
        f"minimum_bypass_shift="
        f"{planner_config.minimum_bypass_shift:.3f}m "
        f"static_side_clearance="
        f"{planner_config.static_side_clearance:.3f}m "
        f"lead_lateral_tolerance="
        f"{planner_config.lead_lateral_tolerance:.3f}m "
        f"static_avoidance_min_hits="
        f"{planner_config.static_avoidance_min_hits} "
        f"static_avoidance_trigger_distance="
        f"{planner_config.static_avoidance_trigger_distance:.3f}m "
        f"lateral_sample_step={planner_config.lateral_sample_step:.3f}m "
        f"collision_margin={planner_config.collision_margin:.3f}m "
        f"collision_check_dt={planner_config.collision_check_dt:.3f}s "
        f"non_yielding_replay_traffic="
        f"{planner_config.non_yielding_replay_traffic} "
        f"rear_follow_lateral_tolerance="
        f"{planner_config.rear_follow_lateral_tolerance:.3f}m"
    )
    rule_planner = RuleBasedPlanner(planner_config)
    stable_controller = StableController(rule_planner.config)
    roundabout_controller = RoundaboutController(
        mode=args.roundabout_mode,
        desired_gap=args.roundabout_gap,
        max_speed=args.roundabout_max_speed,
        max_accel=args.roundabout_max_accel,
        max_decel=args.roundabout_max_decel,
        follow_max_accel=args.roundabout_follow_max_accel,
        lane_half_width=args.roundabout_lane_half_width,
        time_headway=args.roundabout_time_headway,
        idm_comfort_decel=args.roundabout_idm_comfort_decel,
        ttc_emergency=args.roundabout_ttc_threshold,
        curve_lateral_accel=args.roundabout_curve_lat_accel,
        curve_lookahead_time=args.roundabout_curve_lookahead_time,
        wheelbase=planner_config.controller_wheelbase,
        steering_ratio=planner_config.steering_ratio,
        max_steer_deg=min(
            28.0, planner_config.max_steering_wheel_deg
        ),
    )
    print(
        "[roundabout-config] "
        f"mode={roundabout_controller.mode} "
        f"exclusive={roundabout_controller.enabled} "
        f"gap={roundabout_controller.desired_gap:.3f}m "
        f"max_speed={roundabout_controller.max_speed:.3f}m/s "
        f"max_accel={roundabout_controller.max_accel:.3f}m/s2 "
        f"follow_max_accel="
        f"{roundabout_controller.follow_max_accel:.3f}m/s2 "
        f"max_decel={roundabout_controller.max_decel:.3f}m/s2 "
        f"lane_half_width="
        f"{roundabout_controller.lane_half_width:.3f}m "
        f"time_headway={roundabout_controller.time_headway:.3f}s "
        f"idm_comfort_decel="
        f"{roundabout_controller.idm_comfort_decel:.3f}m/s2 "
        f"ttc={roundabout_controller.ttc_emergency:.3f}s "
        f"curve_lat_accel="
        f"{roundabout_controller.curve_lateral_accel:.3f}m/s2 "
        f"curve_lookahead_time="
        f"{roundabout_controller.curve_lookahead_time:.3f}s"
    )
    pre_map_name = None
    map_name = None
    ins_receiver_thread = start_ins_receiver_thread()
    final_control_thread = start_final_control_publisher_thread()
    try:
        main()
    finally:
        deactivate_final_control_publisher()
        final_control_stop_event.set()
        if final_control_thread is not None:
            final_control_thread.join(timeout=1.0)
        ins_receiver_stop_event.set()
        ins_receiver_thread.join(timeout=1.0)
        model.close()
        if drive_trace_logger is not None:
            drive_trace_logger.close()
        if npc_truth_recorder is not None:
            npc_truth_recorder.close()
        if terminal_log is not None:
            terminal_log.close()
