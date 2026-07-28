import argparse
import json
import math
import os
import re
import sys
import time
import multiprocessing as mp
import threading


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

import cv2
import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseArray
    from nav_msgs.msg import Path
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    Pose = None
    PoseArray = None
    Path = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    String = None

from dsac_main import Main
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
from config import Config
import sparse4d_config as s4d_cfg
from npc import decode_npc_payload

# ===== Sparse4Dv3 替换 PointPillars：改动 begin =====
from sparse4d_bridge import (
    DEFAULT_CAM_ORDER,
    DEFAULT_LIDAR_TO_EGO,
    LocalSparse4DDetector,
    ego_pose_to_matrix,
)
# ===== Sparse4Dv3 替换 PointPillars：改动 end =====

DEBUG_SYNC = os.environ.get("E2E_DEBUG_SYNC", "0") == "1"
loop_count = 0
last_ins_sequence = None
last_ins_position = None
duplicate_ins_count = 0
last_duplicate_ins_warn_ts = 0.0
invalid_ins_count = 0
last_invalid_ins_warn_ts = 0.0
map_loading = False
map_ready = False
prepare_received_ts = None
map_ready_ts = None
prepare_sent_ts = None
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
global_plan_expected_frame_id = ""
global_plan_expected_start_xy = None
global_plan_min_stamp_ns = 0
last_global_plan_hold_warn_ts = 0.0
ins_start_gate_enabled = os.environ.get("E2E_INS_START_GATE_ENABLED", "1") == "1"
ins_start_gate_tolerance = float(os.environ.get("E2E_INS_START_GATE_TOL", "50.0"))
ins_start_gate_xy = None
ins_start_gate_map = ""
ins_start_gate_reject_count = 0
last_ins_start_gate_warn_ts = 0.0
ins_receiver_stop_event = threading.Event()

# Obstacle input gate. Camera/Sparse4D remains the default so existing launch
# commands keep their current behaviour.
OBSTACLE_SOURCE_ALIASES = {
    "camera": "camera",
    "sparse4d": "camera",
    "npc": "npc_truth",
    "truth": "npc_truth",
    "npc_truth": "npc_truth",
}
NPC_TRUTH_CHANNEL_NAMES = (
    "npc",
    "vts/perception/obstacles",
)
NPC_TRUTH_DEBUG = False
NPC_TRUTH_EMPTY_LOG_INTERVAL_S = float(
    os.environ.get("E2E_NPC_TRUTH_EMPTY_LOG_INTERVAL_S", "1.0")
)
NPC_TRUTH_ROLE_LOG_LIMIT = max(
    0, int(os.environ.get("E2E_NPC_TRUTH_ROLE_LOG_LIMIT", "3"))
)


def _normalize_obstacle_source(value):
    key = str(value or "").strip().lower().replace("-", "_")
    if key not in OBSTACLE_SOURCE_ALIASES:
        raise ValueError(
            "invalid obstacle source {!r}; use camera or npc_truth".format(value)
        )
    return OBSTACLE_SOURCE_ALIASES[key]


obstacle_source = _normalize_obstacle_source(
    os.environ.get("E2E_OBSTACLE_SOURCE", "camera")
)
npc_channel = None
npc_truth_receive_count = 0
npc_truth_decode_fail_count = 0
npc_truth_poll_count = 0
npc_truth_empty_poll_count = 0
npc_truth_negative_ret_count = 0
npc_truth_receive_error_count = 0
npc_truth_consecutive_empty_count = 0
npc_truth_channel_name = ""
latest_npc_truth_frame = None
latest_npc_truth_frame_sequence = 0
consumed_npc_truth_frame_sequence = 0
last_npc_truth_warn_wall_time = 0.0
last_npc_truth_debug_wall_time = 0.0
last_npc_truth_receive_wall_time = None

# ===== Sparse4Dv3 替换 PointPillars：改动 begin =====
sparse4d_detector = None
first_camera_ready = False
first_camera_ready_ts = None
last_detection_result = None
last_detection_wall_time = 0.0
# 检测结果最多沿用多久（秒）；超时就当作没有障碍物，避免用过期目标做规划
detection_hold_timeout = float(os.environ.get("SPARSE4D_HOLD_TIMEOUT", "0.5"))
detection_fail_count = 0
last_detection_warn_ts = 0.0

# ===== 新增调试开关：SPARSE4D_DEBUG=1 打开逐帧详细打印 =====
SPARSE4D_DEBUG = os.environ.get("SPARSE4D_DEBUG", "1") == "1"
_ego_debug_last_ts = 0.0
_ego_debug_interval = float(os.environ.get("SPARSE4D_DEBUG_EGO_INTERVAL", "1.0"))
_camera_empty_debug_last_ts = 0.0
_camera_empty_debug_interval = float(os.environ.get("SPARSE4D_DEBUG_CAM_INTERVAL", "1.0"))
last_camera_receive_monotonic = None
last_camera_source_timestamp = None
camera_receive_count = 0
# ego.theta 的单位：默认当作弧度。如果调试打印出的 raw value 看起来像角度
# （数值是几十/几百而不是 -pi~pi 范围），设置环境变量
# SPARSE4D_EGO_THETA_IS_DEGREES=1 再跑一次。
_EGO_THETA_IS_DEGREES = os.environ.get("SPARSE4D_EGO_THETA_IS_DEGREES", "0") == "1"


def _reset_sparse4d_state():
    """回合切换时清空检测侧的所有跨帧状态。"""
    global first_camera_ready
    global first_camera_ready_ts
    global last_detection_result
    global last_detection_wall_time
    global detection_fail_count
    global last_detection_warn_ts
    global last_camera_receive_monotonic
    global last_camera_source_timestamp
    global camera_receive_count
    global npc_truth_receive_count
    global npc_truth_decode_fail_count
    global npc_truth_poll_count
    global npc_truth_empty_poll_count
    global npc_truth_negative_ret_count
    global npc_truth_receive_error_count
    global npc_truth_consecutive_empty_count
    global latest_npc_truth_frame
    global latest_npc_truth_frame_sequence
    global consumed_npc_truth_frame_sequence
    global last_npc_truth_warn_wall_time
    global last_npc_truth_debug_wall_time
    global last_npc_truth_receive_wall_time

    first_camera_ready = False
    first_camera_ready_ts = None
    last_detection_result = None
    last_detection_wall_time = 0.0
    detection_fail_count = 0
    last_detection_warn_ts = 0.0
    last_camera_receive_monotonic = None
    last_camera_source_timestamp = None
    camera_receive_count = 0
    npc_truth_receive_count = 0
    npc_truth_decode_fail_count = 0
    npc_truth_poll_count = 0
    npc_truth_empty_poll_count = 0
    npc_truth_negative_ret_count = 0
    npc_truth_receive_error_count = 0
    npc_truth_consecutive_empty_count = 0
    latest_npc_truth_frame = None
    latest_npc_truth_frame_sequence = 0
    consumed_npc_truth_frame_sequence = 0
    last_npc_truth_warn_wall_time = 0.0
    last_npc_truth_debug_wall_time = 0.0
    last_npc_truth_receive_wall_time = None
    if sparse4d_detector is not None:
        sparse4d_detector.mark_reset()
    if globals().get("npc_channel") is not None:
        drained = drain_npc_truth_queue()
        if drained:
            print(
                f"[npc-truth] discarded {drained} stale frame(s) "
                "at episode reset"
            )
    current_model = globals().get("model")
    if current_model is not None:
        current_model.local_detection_frame = None
        current_model.local_npc_truth_frame = None
# ===== Sparse4Dv3 替换 PointPillars：改动 end =====


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

    # ===== 修改 begin：AITownReconstructed_V0103_200518.xodr 允许 UTM 大坐标 =====
    current_map = globals().get("map_name", "")
    if _is_aitown_map(current_map):
        if abs(x) > 10000000.0 or abs(y) > 10000000.0:
            return False
    else:
        # 其它地图仍然保持原来的局部坐标限制
        if abs(x) > 10000.0 or abs(y) > 10000.0:
            print(66666666666666666666)
            return False
    # ===== 修改 end =====

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
    if not _valid_imu_xy(xy[0], xy[1]):
        return False, "invalid position", xy

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
    global last_global_plan_hold_warn_ts

    global_plan_ready = False
    global_plan_ready_ts = None
    global_plan_ready_points = 0
    global_plan_ready_frame_id = ""
    global_plan_ready_start_dist = float("inf")
    global_plan_expected_frame_id = _xodr_frame_id(map_id)
    global_plan_expected_start_xy = (float(start_x), float(start_y))
    global_plan_min_stamp_ns = 0
    last_global_plan_hold_warn_ts = 0.0


def _on_global_plan_ready_msg(msg):
    global global_plan_ready
    global global_plan_ready_ts
    global global_plan_ready_points
    global global_plan_ready_frame_id
    global global_plan_ready_start_dist

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

    if global_plan_ready:
        return False

    node = _ensure_global_plan_ready_subscription()
    if node is not None and rclpy is not None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)

    if global_plan_ready:
        print(
            "[global-plan-ready] planning finished; release start hold "
            f"frame={global_plan_ready_frame_id} "
            f"points={global_plan_ready_points} "
            f"start_dist={global_plan_ready_start_dist:.3f}"
        )
        return False

    now = time.time()
    if now - last_global_plan_hold_warn_ts > 1.0:
        last_global_plan_hold_warn_ts = now
        print(
            "[global-plan-ready] waiting for /global_plan after start; "
            "send zero-speed hold control"
        )
    send_control_cmd(-0.1, 0.0, 0.0)
    return True


def hold_until_ego_ready():
    # ===== 新增：这个 gate 之前完全没有日志，如果卡在这里主循环永远不会
    # 走到 get_detection_msg，图像自然也不会发给 Sparse4D 服务端。 =====
    if first_ins_ready and model.ego is not None:
        return False

    send_control_cmd(-0.1, 0.0, 0.0)
    return True



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
        # RosPlanningBridge owns its node from a dedicated background
        # executor. Keep request/event spinning on a separate node so the
        # same ROS node is never spun concurrently by two threads.
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
        # This node may be synchronously spun while publishing an episode
        # event, so it must not reuse the background-executor bridge node.
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

def prepare():
    global prepare_sent_ts
    prepare_sent_ts = time.time()
    print("send prepare result")
    print(
        "[prepare-timing] prepare_result_sent "
        f"wall_time={prepare_sent_ts:.3f} "
        f"prepare_received={prepare_received_ts} "
        f"map_ready={map_ready_ts}"
    )
    send_prepare_result = ActorPrepareResult()
    send_prepare_result.session_id = session_id
    send_prepare_result.actor_id = actor_id
    send_prepare_result.result = True
    data = send_prepare_result.SerializeToString()
    length = len(data)
    ret = prepare_channel.put(MT_ACTOR_PREPARE_RESULT, length, data)
    if ret != 0:
        print("send prepare msg error")


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
    global invalid_ins_count
    global last_invalid_ins_warn_ts
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts
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
            print("[prepare][WARN] ignore ActorPrepare with empty session_id", flush=True)
            return
        if incoming_session_id == current_session_id:
            # The daemon may retransmit Prepare while awaiting its result.
            # Never reset an already prepared/running episode for that case.
            return
        if current_session_id and not _is_newer_session(
            incoming_session_id, current_session_id
        ):
            print(
                "[prepare][WARN] ignore stale or unordered ActorPrepare "
                f"received={incoming_session_id} current={current_session_id}",
                flush=True,
            )
            return
        # ActorPrepare is an authoritative episode boundary.  In particular,
        # a new prepare can arrive while the prior episode's abort acknowledgement
        # is absent; do not let its old NT_START_TEST state leak into the new map.
        start_test = False
        prepare_received_ts = time.time()
        map_loading = True
        map_ready = False
        map_ready_ts = None
        first_ins_ready = False
        first_ins_ready_ts = None
        first_pointcloud_ready = False
        first_pointcloud_ready_ts = None
        first_control_sent = False
        first_control_sent_ts = None
        recv_prepare = True
        # ===== Sparse4Dv3：新回合必须清掉时序缓存和 track id =====
        _reset_sparse4d_state()
        session_id = incoming_session_id
        print(
            "[prepare-timing] prepare_received "
            f"wall_time={prepare_received_ts:.3f} "
            f"session_id={session_id}"
        )
        publish_episode_event("reset", repeat=1, wait_subscriber=0.2)
        local_abort_pending = False
        local_abort_sent_ts = 0.0
        plan_start_check_remaining = 0
        plan_start_check_xy = None
        plan_start_check_map = ""
        last_ins_sequence = None
        last_ins_position = None
        duplicate_ins_count = 0
        last_duplicate_ins_warn_ts = 0.0
        invalid_ins_count = 0
        last_invalid_ins_warn_ts = 0.0
        reset_ins_start_gate()
        model.last_ego = None
        model.invalid_ego_sample_count = 0
        model.last_invalid_ego_sample_warn_wall_time = 0.0
        model.last_ego_speed_pose = None
        model.last_ego_speed_pose_wall_time = 0.0

        brief_data = json.loads(prepare_msg.archive_info.brief_data)
        weather =  brief_data["environment"]['weather']
        model.weather = weather
        dsac_main.change_map(brief_data, weather)   #这个地方需要保留：因为STT需要地图信息。
        model.start = 0
        #print('change_map')
        model.change_map(brief_data["zjl_odv_file"])
        map_name = brief_data["zjl_odv_file"]
        init_state = brief_data["testees"][0]["init_state"]
        target_state = brief_data["testees"][0]["target_state"]
        #print(target_state)
        role_id = brief_data["testees"][0]["role_id"]
        start_ins_start_gate(init_state["x"], init_state["y"], map_name)
        model.set_destination(     
            target_state["x"],
            target_state["y"],
            np.deg2rad(target_state["orientation_z"]),
        )
        # Publish one planning request and wait until the matching /global_plan is ready.
        plan_request_sent = publish_global_plan_request_from_brief_data(
            brief_data,
            wait_global_plan=True,
        )
        map_loading = False
        map_ready = bool(plan_request_sent)
        if map_ready:
            map_ready_ts = time.time()
            print(
                "[prepare-timing] map_ready "
                f"wall_time={map_ready_ts:.3f} "
                f"elapsed={map_ready_ts - prepare_received_ts:.3f} "
                f"map={map_name} "
                f"global_plan_points={global_plan_ready_points}"
            )
        else:
            print(
                "[prepare-timing][WARN] map not ready because /global_plan is not ready; "
                "ActorPrepareResult will not be sent yet"
            )


# ===== Sparse4Dv3 替换 PointPillars：改动 begin =====
def _ego_lidar2global():
    """由 INS 自车位姿构造 lidar -> global 的 4x4 矩阵。

    ego.theta 由 Predictor.update_ego() 直接读取 INS heading，单位为弧度。
    DEFAULT_LIDAR_TO_EGO 当前不包含原来的 -90 度轴旋转。

    ===== 新增调试打印（2026-07-19）=====
    之前这个函数任何一步失败都是静默 return None，外面只能看到笼统的
    "ego pose unavailable"。现在每一个失败分支都打印具体原因，并且第一次
    发现 ego 对象上既没有 yaw 也没有 heading 属性时，会把 ego 对象上所有
    可读属性名打印出来，方便直接定位真实的字段名。
    """
    global _ego_debug_last_ts

    def _should_print():
        # 限频打印，避免刷屏；每 _ego_debug_interval 秒最多打一次
        global _ego_debug_last_ts
        now = time.time()
        if now - _ego_debug_last_ts < _ego_debug_interval:
            return False
        _ego_debug_last_ts = now
        return True

    ego = getattr(model, "ego", None)
    if ego is None:
        if SPARSE4D_DEBUG and _should_print():
            print("[sparse4d-debug] _ego_lidar2global: model.ego is None")
        return None

    try:
        x = float(ego.x)
        y = float(ego.y)
    except Exception as exc:
        if SPARSE4D_DEBUG and _should_print():
            print(f"[sparse4d-debug] _ego_lidar2global: failed reading ego.x/ego.y: {exc}")
        return None

    z = float(getattr(ego, "z", 0.0) or 0.0)

    try:
        yaw = float(ego.theta)
    except Exception as exc:
        if SPARSE4D_DEBUG and _should_print():
            print(f"[sparse4d-debug] failed reading ego.theta: {exc}")
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
        if SPARSE4D_DEBUG and _should_print():
            print(
                f"[sparse4d-debug] _ego_lidar2global: non-finite value "
                f"x={x} y={y} yaw={yaw}"
            )
        return None

    if SPARSE4D_DEBUG and _should_print():
        print(
            "[sparse4d-debug] _ego_lidar2global with heading: "
            f"x={x:.3f} y={y:.3f} z={z:.3f} "
            f"yaw_used={yaw:.4f}rad yaw_deg={math.degrees(yaw):.3f}"
        )

    ego2global = ego_pose_to_matrix(x, y, z, yaw)
    return ego2global @ np.asarray(DEFAULT_LIDAR_TO_EGO, dtype=np.float64)


def _warn_detection(reason):
    global last_detection_warn_ts
    now = time.time()
    if now - last_detection_warn_ts < 1.0:
        return
    last_detection_warn_ts = now
    print(f"[sparse4d][WARN] {reason} count={detection_fail_count}")


def _normalize_camera_timestamp(value):
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp <= 0.0:
        return None
    if timestamp > 1e17:
        timestamp *= 1e-9
    elif timestamp > 1e14:
        timestamp *= 1e-6
    elif timestamp > 1e11:
        timestamp *= 1e-3
    return timestamp


def _camera_source_time(messages):
    for field_name in ("measurement_time", "camera_timestamp", "timestamp_sec"):
        values = [
            _normalize_camera_timestamp(getattr(message, field_name, None))
            for message in messages
        ]
        if values and all(value is not None for value in values):
            ordered = sorted(values)
            middle = len(ordered) // 2
            timestamp = (
                ordered[middle]
                if len(ordered) % 2
                else 0.5 * (ordered[middle - 1] + ordered[middle])
            )
            return timestamp, field_name, max(values) - min(values)

    header_values = [
        _normalize_camera_timestamp(
            getattr(getattr(message, "header", None), "send_ts", None)
        )
        for message in messages
    ]
    if header_values and all(value is not None for value in header_values):
        ordered = sorted(header_values)
        middle = len(ordered) // 2
        timestamp = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return timestamp, "header.send_ts", max(header_values) - min(header_values)
    return None, "none", None


def _print_camera_frequency(messages):
    global last_camera_receive_monotonic
    global last_camera_source_timestamp
    global camera_receive_count

    receive_now = time.monotonic()
    receive_dt = (
        receive_now - last_camera_receive_monotonic
        if last_camera_receive_monotonic is not None
        else None
    )
    source_timestamp, source_field, source_spread = _camera_source_time(messages)
    source_dt = (
        source_timestamp - last_camera_source_timestamp
        if source_timestamp is not None and last_camera_source_timestamp is not None
        else None
    )
    camera_receive_count += 1
    print(
        "[camera-frequency] "
        f"frame={camera_receive_count} cams={len(messages)} "
        f"source_field={source_field} "
        f"source_ts={source_timestamp:.6f} "
        if source_timestamp is not None
        else (
            "[camera-frequency] "
            f"frame={camera_receive_count} cams={len(messages)} "
            "source_field=none source_ts=n/a "
        ),
        end="",
        flush=True,
    )
    print(
        f"source_dt={source_dt:.4f}s source_hz={1.0 / source_dt:.2f} "
        if source_dt is not None and source_dt > 0.0
        else "source_dt=n/a source_hz=n/a ",
        end="",
        flush=True,
    )
    print(
        f"receive_dt={receive_dt:.4f}s receive_hz={1.0 / receive_dt:.2f} "
        if receive_dt is not None and receive_dt > 0.0
        else "receive_dt=n/a receive_hz=n/a ",
        end="",
        flush=True,
    )
    print(
        f"six_cam_spread={source_spread:.6f}s"
        if source_spread is not None
        else "six_cam_spread=n/a",
        flush=True,
    )
    last_camera_receive_monotonic = receive_now
    if source_timestamp is not None:
        last_camera_source_timestamp = source_timestamp


def get_camera_detection():
    """取一帧 6 路环视图像，在当前进程执行 Sparse4Dv3 检测。

    ===== 新增调试打印（2026-07-19）=====
    逐步打印：是否拿到图像、lidar2global 是否算出来、是否真正调用了
    sparse4d_detector.detect()、返回结果如何 —— 方便确认图像数据到底有没有
    进入本地 Sparse4D 模型。
    """
    global first_camera_ready
    global first_camera_ready_ts
    global last_detection_result
    global last_detection_wall_time
    global detection_fail_count
    global _camera_empty_debug_last_ts

    msgs = image_channel.get_image()
    if not msgs:
        if SPARSE4D_DEBUG:
            now = time.time()
            if now - _camera_empty_debug_last_ts >= _camera_empty_debug_interval:
                _camera_empty_debug_last_ts = now
                print("[sparse4d-debug] image_channel.get_image() returned empty this loop")
        return None

    # ===== 积压丢帧修复（2026-07-22）=====
    # 感知端单帧总耗时（decode+preprocess+inference+postprocess）通常明显
    # 高于相机通道的产帧间隔。如果这里只取一次 get_image()，拿到的永远是
    # 队列里最老、尚未被消费的一帧；随着运行时间增加，积压只会越来越多，
    # 表现为“画面与实际场景不同步”且不断加重，严重时还会因为底层视频流
    # 解码器跳帧/丢包导致画面模糊、拖影。
    #
    # 这里持续把通道排空到没有更新的帧为止，只保留最新一批 6 路图像用于
    # 本次检测；被跳过的旧帧直接丢弃。这样即使吞吐率不够导致掉帧，也不会
    # 出现“越跑越滞后”的单调累积问题。
    drained_stale_frames = 0
    while True:
        newer_msgs = image_channel.get_image()
        if not newer_msgs:
            break
        msgs = newer_msgs
        drained_stale_frames += 1
    if SPARSE4D_DEBUG and drained_stale_frames:
        print(
            "[sparse4d-debug] dropped %d backlog camera frame(s), "
            "using newest frame only" % drained_stale_frames
        )

    _print_camera_frequency(msgs)

    # if SPARSE4D_DEBUG:
    #     print(f"[sparse4d-debug] got {len(msgs)} camera image msg(s) from image_channel")
    if SPARSE4D_DEBUG:
        print(f"[sparse4d-debug] got {len(msgs)} camera image msg(s) from image_channel")
        for i, m in enumerate(msgs):
            try:
                arr = np.asarray(m.data)
                print(
                    f"[sparse4d-debug] cam[{i}] "
                    f"raw_len={arr.size} dtype={arr.dtype} "
                    f"min={arr.min()} max={arr.max()}"
                )
            except Exception as exc:
                print(f"[sparse4d-debug] cam[{i}] failed to inspect data: {exc}")

    if not first_camera_ready:
        first_camera_ready = True
        first_camera_ready_ts = time.time()
        print(
            "[prepare-timing] first_camera_ready "
            f"wall_time={first_camera_ready_ts:.3f} cams={len(msgs)}"
        )

    lidar2global = _ego_lidar2global()
    if lidar2global is None:
        detection_fail_count += 1
        _warn_detection("ego pose unavailable, skip detection")
        return None

    # 用 INS 时间戳保证时间轴单调；拿不到就退化为墙上时间
    timestamp = None
    if last_ins_sequence is not None:
        stamp = getattr(getattr(model, "ego", None), "timestamp", None)
        if stamp is not None:
            try:
                timestamp = float(stamp)
            except Exception:
                timestamp = None
    if timestamp is None:
        timestamp = time.time()

    if SPARSE4D_DEBUG:
        print(
            f"[sparse4d-debug] calling sparse4d_detector.detect() "
            f"with {len(msgs)} images, timestamp={timestamp:.3f}"
        )
    try:
        rsp = sparse4d_detector.detect(msgs, lidar2global, timestamp=timestamp)
    except Exception as exc:
        detection_fail_count += 1
        _warn_detection("local inference failed: %s: %s" %
                        (type(exc).__name__, exc))
        return None
    if rsp is None:
        detection_fail_count += 1
        _warn_detection("local detection unavailable")
        return None

    # if SPARSE4D_DEBUG:
    #     print(
    #         f"[sparse4d-debug] detect() returned OK: "
    #         f"num_dets={rsp.get('num_dets')} "
    #         f"decode_ms={rsp.get('decode_ms')} "
    #         f"preprocess_ms={rsp.get('preprocess_ms')} "
    #         f"inference_ms={rsp.get('inference_ms')} "
    #         f"postprocess_ms={rsp.get('postprocess_ms')} "
    #         f"preprocess_detail={rsp.get('preprocess_detail')} "
    #         f"total_ms={rsp.get('latency_ms')}"
    #     )

    detection_fail_count = 0
    last_detection_result = rsp
    last_detection_wall_time = time.time()
    if DEBUG_SYNC:
        print(
            "[sync-debug] sparse4d "
            f"frame_id={rsp.get('frame_id')} "
            f"num_dets={rsp.get('num_dets')} "
            f"latency_ms={rsp.get('latency_ms')}"
        )
    return rsp

def _npc_channel_debug_identity():
    """Return configured and process-local channel identity for diagnostics."""
    if npc_channel is None:
        return "name=<none> id=<none> object=<none>"

    def _read_channel_member(member_name):
        try:
            value = getattr(npc_channel, member_name)
            return value() if callable(value) else value
        except Exception as exc:
            return "<{}:{}>".format(type(exc).__name__, exc)

    return "name={!r} id={!r} object={}".format(
        _read_channel_member("name"),
        _read_channel_member("id"),
        hex(id(npc_channel)),
    )


def _npc_message_type(msg):
    try:
        return msg.type()
    except Exception as exc:
        return "<{}:{}>".format(type(exc).__name__, exc)


def _npc_payload_debug_summary(payload):
    payload = bytes(payload)
    role_count = (
        int.from_bytes(payload[:4], byteorder="little", signed=False)
        if len(payload) >= 4 else None
    )
    return (
        "bytes={} role_count_header={} head_hex={}".format(
            len(payload), role_count, payload[:24].hex()
        )
    )


def _npc_decoded_roles_debug_summary(decoded):
    roles = decoded.get("roles", []) if isinstance(decoded, dict) else []
    samples = []
    for role in roles[:NPC_TRUTH_ROLE_LOG_LIMIT]:
        position = role.get("position", {})
        vector = role.get("vector_raw", {})
        samples.append(
            "{}({}) pos=({:.2f},{:.2f},{:.2f}) vel=({:.2f},{:.2f},{:.2f}) "
            "frame={} ts={:.3f}".format(
                role.get("role_name", "?"),
                role.get("model_name", "?"),
                float(position.get("x", float("nan"))),
                float(position.get("y", float("nan"))),
                float(position.get("z", float("nan"))),
                float(vector.get("x", float("nan"))),
                float(vector.get("y", float("nan"))),
                float(vector.get("z", float("nan"))),
                role.get("frame_counter", "?"),
                float(role.get("timestamp_s", float("nan"))),
            )
        )
    return "roles={} sample=[{}]".format(len(roles), "; ".join(samples))


def get_npc_truth_detection(max_messages=128):
    """Drain the NPC channel and return the newest successfully decoded frame."""
    global npc_truth_receive_count
    global npc_truth_decode_fail_count
    global npc_truth_poll_count
    global npc_truth_empty_poll_count
    global npc_truth_negative_ret_count
    global npc_truth_receive_error_count
    global npc_truth_consecutive_empty_count
    global latest_npc_truth_frame
    global latest_npc_truth_frame_sequence
    global last_npc_truth_warn_wall_time
    global last_npc_truth_debug_wall_time
    global last_npc_truth_receive_wall_time

    if npc_channel is None:
        return None

    npc_truth_poll_count += 1
    latest = None
    latest_message_type = None
    latest_payload_size = 0
    drained = 0
    for _ in range(max(1, int(max_messages))):
        try:
            ret, msg = npc_channel.get()

            if msg is None:
                npc_truth_empty_poll_count += 1
                npc_truth_consecutive_empty_count += 1
                now = time.time()
                if (
                    NPC_TRUTH_DEBUG
                    and now - last_npc_truth_debug_wall_time
                    >= NPC_TRUTH_EMPTY_LOG_INTERVAL_S
                ):
                    last_npc_truth_debug_wall_time = now
                    print(
                        "[npc-truth][EMPTY] "
                        f"poll={npc_truth_poll_count} ret={ret!r} "
                        f"consecutive_empty={npc_truth_consecutive_empty_count} "
                        f"empty_total={npc_truth_empty_poll_count} "
                        f"received_total={npc_truth_receive_count} "
                        f"decode_failed={npc_truth_decode_fail_count} "
                        f"state=prepare:{globals().get('recv_prepare', None)} "
                        f"start:{globals().get('start_test', None)} "
                        f"session={globals().get('session_id', '')!r} "
                        f"{_npc_channel_debug_identity()}",
                        flush=True,
                    )
                break
            if ret < 0:
                npc_truth_negative_ret_count += 1
                if NPC_TRUTH_DEBUG:
                    print(
                        "[npc-truth][NEGATIVE_RET] "
                        f"poll={npc_truth_poll_count} ret={ret!r} "
                        f"type={_npc_message_type(msg)!r} "
                        f"{_npc_channel_debug_identity()}",
                        flush=True,
                    )
                continue
            npc_truth_consecutive_empty_count = 0
            message_type = _npc_message_type(msg)
            try:
                payload = bytes(libMulticastNetwork.getMessageData(msg))
            except Exception as exc:
                npc_truth_receive_error_count += 1
                print(
                    "[npc-truth][PAYLOAD_ERROR] "
                    f"poll={npc_truth_poll_count} ret={ret!r} "
                    f"type={message_type!r} error={type(exc).__name__}: {exc} "
                    f"{_npc_channel_debug_identity()}",
                    flush=True,
                )
                continue
            drained += 1
            npc_truth_receive_count += 1
            now = time.time()
            inter_arrival_s = (
                None if last_npc_truth_receive_wall_time is None
                else now - last_npc_truth_receive_wall_time
            )
            last_npc_truth_receive_wall_time = now
            if NPC_TRUTH_DEBUG:
                print(
                    "[npc-truth][RX] "
                    f"poll={npc_truth_poll_count} ret={ret!r} "
                    f"type={message_type!r} "
                    f"inter_arrival_s="
                    f"{inter_arrival_s if inter_arrival_s is not None else 'first'} "
                    f"{_npc_payload_debug_summary(payload)} "
                    f"{_npc_channel_debug_identity()}",
                    flush=True,
                )
            try:
                decoded = decode_npc_payload(payload)
            except Exception as exc:
                npc_truth_decode_fail_count += 1
                now = time.time()
                if now - last_npc_truth_warn_wall_time >= 1.0:
                    last_npc_truth_warn_wall_time = now
                    print(
                        "[npc-truth][WARN] decode failed "
                        f"count={npc_truth_decode_fail_count} "
                        f"error={type(exc).__name__}: {exc} "
                        f"{_npc_payload_debug_summary(payload)} "
                        f"type={message_type!r} "
                        f"{_npc_channel_debug_identity()}",
                        flush=True,
                    )
                continue
            if isinstance(decoded, dict):
                latest = decoded

        except Exception as exc:
            npc_truth_receive_error_count += 1
            now = time.time()
            if now - last_npc_truth_warn_wall_time >= 1.0:
                last_npc_truth_warn_wall_time = now
 
            break

    if latest is not None:
        latest_npc_truth_frame = latest
        latest_npc_truth_frame_sequence += 1
        now = time.time()
        if now - last_npc_truth_debug_wall_time >= 0.5:
            last_npc_truth_debug_wall_time = now

    else:
        now = time.time()
        if now - last_npc_truth_debug_wall_time >= 1.0:
            last_npc_truth_debug_wall_time = now

    return latest

def drain_npc_truth_queue(max_messages=128):
    """Discard queued NPC messages without decoding them."""
    if npc_channel is None:
        return 0
    drained = 0
    for _ in range(max(1, int(max_messages))):
        try:
            ret, msg = npc_channel.get()
        except Exception:
            break
        if msg is None:
            break
        if ret >= 0:
            drained += 1
    return drained


def get_detection_msg(dsac_main, notify):
    """Select camera/Sparse4D or simulator NPC truth for obstacle planning.

    Both branches are converted by Predictor into the same world-frame
    ObstacleState contract before entering planning.
    """
    if obstacle_source == "npc_truth":
        global consumed_npc_truth_frame_sequence
        truth = get_npc_truth_detection()
        if truth is not None:
            consumed_npc_truth_frame_sequence = (
                latest_npc_truth_frame_sequence
            )
        elif (
            latest_npc_truth_frame is not None
            and latest_npc_truth_frame_sequence
            > consumed_npc_truth_frame_sequence
        ):
            # drain_sensor_queues() polls NPC first, matching test.py. Consume
            # that newest buffered frame exactly once as a fresh measurement.
            truth = latest_npc_truth_frame
            consumed_npc_truth_frame_sequence = (
                latest_npc_truth_frame_sequence
            )

        if truth is not None:
            print('turth_npc',truth)
            return model.infer_npc_truth(truth, dsac_main, notify)
        # No new truth frame: let Predictor propagate its bounded obstacle
        # cache instead of treating an old truth frame as a fresh measurement.
        return model.infer([], dsac_main, notify)

    rsp = get_camera_detection()
    if rsp is None:
        if (
            last_detection_result is not None
            and time.time() - last_detection_wall_time <= detection_hold_timeout
        ):
            rsp = last_detection_result
        else:
            rsp = {"results": [], "num_dets": 0, "frame": "lidar"}

    # 传递完整帧，保留 Sparse4D 的 frame_id、timestamp 和原生 tracking_id。
    if hasattr(model, "infer_detections"):
        return model.infer_detections(rsp, dsac_main, notify)
    return model.infer(rsp, dsac_main, notify)


def drain_sensor_queues(drain_camera=True):
    """清空传感器队列，避免上一回合的残留帧被当成当前帧。"""
    if drain_camera:
        image_channel.get_image()
    if obstacle_source == "npc_truth":
        # Mirror test.py: continuously poll the simulator truth channel during
        # prepare/wait states and immediately before each active control cycle.
        # get_detection_msg() consumes the newest buffered frame exactly once.
        get_npc_truth_detection()
    else:
        drain_npc_truth_queue()
    # 点云通道已不参与感知，但仍要排空，否则组播缓冲会一直堆积
    pointcloud_channel.get_pointcloud()
# ===== Sparse4Dv3 替换 PointPillars：改动 end =====


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
    global invalid_ins_count
    global last_invalid_ins_warn_ts
    global ins_start_gate_xy
    global ins_start_gate_map
    global ins_start_gate_reject_count
    global last_ins_start_gate_warn_ts
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
            and notify_session_id and current_session_id
            and notify_session_id != current_session_id
        ):
            print(
                "[platform-notify][WARN] ignore stale session "
                f"type={notify.type} received={notify_session_id} "
                f"current={current_session_id}",
                flush=True,
            )
            return notify.type
        if (
            notify.type == NT_ABORT_TEST
            and not notify_session_id and current_session_id
            and start_test and not local_abort_pending
            and not model.rl_fallback_active
        ):
            print(
                "[platform-notify][WARN] ignore unsolicited empty-session "
                "abort while RL is active; waiting for a session-matched "
                f"notify current_session={current_session_id} role_id={role_id}",
                flush=True,
            )
            return notify.type
        # if notify.type == NT_ABORT_TEST or notify.type == NT_FINISH_TEST:
        #     if notify.type == NT_ABORT_TEST and model.start == 1:
        #         done_type = "termination"
        #         dsac_main.finish(model.collision, done_type)
        #         model.collision = 0
        #         abort_test()
        #     print("finish session")
        #     start_test = False
        #     recv_prepare = False
        if notify.type == NT_FINISH_TEST or notify.type == NT_ABORT_TEST :
            print(
                "[platform-finish] accepted notify "
                f"type={notify.type} session_id={notify.session_id} "
                f"role_id={notify.role_id} "
                f"local_abort_pending={local_abort_pending} "
                f"rl_fallback_active={model.rl_fallback_active}",
                flush=True,
            )
            local_abort_requested = local_abort_pending
            local_abort_echo = notify.type == NT_ABORT_TEST and local_abort_requested
            if local_abort_echo:
                print(
                    "[episode-event] suppress local abort event; "
                    "wait for next ActorPrepare before clearing ROS path"
                )
            else:
                publish_episode_event(
                    "abort" if notify.type == NT_ABORT_TEST else "finish",
                    repeat=1,
                )
                local_abort_pending = False
                local_abort_sent_ts = 0.0
            episode_started = start_test or dsac_main.step > 0
            print("finish session", notify.type, model.start, dsac_main.step, "started", episode_started)
            if episode_started:
                done_type = (
                    model.rl_fallback_done_type
                    if model.rl_fallback_active and model.rl_fallback_done_type
                    else "termination"
                )

                dsac_main.finish(
                    model.collision,
                    done_type,
                    model.rl_fallback_active or (
                        notify.type == NT_ABORT_TEST and model.start == 1
                    ),
                )
            else:
                print("ignore finish before start_test; skip dsac_main.finish")
                dsac_main.step = 0
                dsac_main.start = 0

            model.collision = 0
            model.time_out = False
            model.rl_fallback_active = False
            model.rl_fallback_done_type = None
            # done_type = "termination"

            # if model.time_out:
            #     done_type = 'time_out'

            # dsac_main.finish(model.collision, done_type)
            # model.collision = 0
            # model.time_out = False


            start_test = False
            recv_prepare = False
            _reset_sparse4d_state()
            plan_start_check_remaining = 0
            plan_start_check_xy = None
            plan_start_check_map = ""
            last_ins_sequence = None
            last_ins_position = None
            duplicate_ins_count = 0
            last_duplicate_ins_warn_ts = 0.0
            invalid_ins_count = 0
            last_invalid_ins_warn_ts = 0.0
            reset_ins_start_gate()
            model.start = 0
            model.step = 0
            model.planning_step = 0
            model.obs_gap = 0
            model.cached_obstacles = []
            model.cached_obstacles_wall_time = 0.0
            model.latest_obstacle_measurement_stamp_s = None
            model.local_detection_frame = None
            model.local_npc_truth_frame = None
            model.last_pointclouds = []
            model.last_ego = None
            model.invalid_ego_sample_count = 0
            model.last_invalid_ego_sample_warn_wall_time = 0.0
            model.last_ego_speed_pose = None
            model.last_ego_speed_pose_wall_time = 0.0
            model.ok = 0
            model.last_steer = 0
            model.time_out = False
            model.start_time = 0
            model.last_time = None
            model.last_a = None
            model.last_h_a = None
            model.last_rot = None
            model.goal = 0

        elif notify.type == NT_START_TEST:
            _reset_sparse4d_state()
            # Drop the final sensor frame buffered at the episode boundary.
            # Detection resumes on the next main-loop iteration with a frame
            # produced after the new episode has started.
            drain_sensor_queues()
            model.rl_fallback_active = False
            model.rl_fallback_done_type = None
            print(
                "[platform-start] received notify "
                f"session_id={notify.session_id} role_id={notify.role_id}"
            )
            start_test = True
            model.start = 1
        elif notify.type == NT_DESTROY_ROLE:
            pass
        if notify.type is not None:
            return notify.type


def send_control_cmd(target_acc, target_speed, target_steer):
    if target_speed <= 1e-4 and target_acc < 0:
        target_acc = 0.0
    cmd = VehicleControl()
    # 目标加速度
    target_acc=100.0
    # target_steer=1
    target_speed=30
    # target_acc=2.0
    cmd.acceleration = target_acc

    # 目标速度

    cmd.speed = target_speed
    # 目标方向盘转角
    print('target_steer',target_steer)


    cmd.steering_control.target_steering_wheel_angle = target_steer #这里方向盘转角的转换是否有问题
    data = cmd.SerializeToString()
    length = len(data)
    ret = cmd_channel.put(VEHICLE_CONTROL, length, data)
    if ret != 0:
        print("send cmd error")


def get_vehicle_feedback():
    ret, msg = cmd_channel.get()
    if msg is None or ret < 0:
        return
    if msg.type() == VEHICLE_FEEDBACK:
        feedback = VehicleFeedback()
        data = libMulticastNetwork.getMessageData(msg)
        feedback.ParseFromString(data)
        model.update_vehicle_feedback(feedback)


def get_vehicle_pose():
    global last_ins_sequence
    global last_ins_position
    global duplicate_ins_count
    global invalid_ins_count
    global first_ins_ready
    global first_ins_ready_ts
    ins = ins_channel.get_ins()
    # Use a monotonic clock so predictor can measure from the instant this
    # run-loop receives the sample until it is handed to the ROS publisher.
    ins_received_monotonic = time.monotonic()
    # print(ins.sequence_num)
    if ins.sequence_num == 0 or ins.sequence_num > 1000000:
        return
    if ins.sequence_num==1:
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
    if not model.update_ego(ins, received_monotonic=ins_received_monotonic):
        return
    model.publish_latest_ego(dsac_main, ins)

    if not first_ins_ready:
        first_ins_ready = True
        first_ins_ready_ts = time.time()
        print(
            "[prepare-timing] first_ins_ready "
            f"wall_time={first_ins_ready_ts:.3f} "
            f"ins_seq={last_ins_sequence}"
        )
    check_plan_start_imu(model.ego, last_ins_sequence)


def ins_receiver_loop():
    """Receive and publish INS independently from point-cloud inference."""
    print("[ins-thread] receiver started")
    while not ins_receiver_stop_event.is_set():
        # The gate is installed only after ActorPrepare has reset all episode
        # state. Do not publish stale samples between episodes.
        if not recv_prepare or ins_start_gate_xy is None:
            ins_receiver_stop_event.wait(0.005)
            continue
        try:
            get_vehicle_pose()
            # get_ins() is non-blocking on some channel implementations.
            # Avoid a busy loop while retaining sub-frame receive latency.
            ins_receiver_stop_event.wait(0.001)
        except Exception as exc:
            print(f"[ins-thread][WARN] receive/update failed: {exc}")
            ins_receiver_stop_event.wait(0.01)
        #这个地方可能会缓存上回合遗留的自车位置信息。

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

def main():
    global map_name
    global pre_map_name
    global loop_count
    global map_ready
    global map_loading
    global first_control_sent
    global first_control_sent_ts

    while 1:
        loop_count += 1
        # ActorPrepare is the authoritative session transition.  Poll it even
        # after a previous Prepare has completed, otherwise a new Prepare can
        # stay queued while its Notify is rejected as belonging to a newer
        # session.
        get_prepare()
        notify = process_notify()
        if local_abort_pending and recv_prepare:
            # The simulator may start the next episode without echoing our
            # NT_ABORT_TEST for the previous one.  Keep consuming
            # ActorPrepare here: otherwise this branch waits forever on the
            # old session and the new prepare cannot receive its result.
            get_prepare()
            if local_abort_pending:
                drain_sensor_queues()
                get_vehicle_feedback()
                time.sleep(0.02)
                continue
        if not recv_prepare:
            drain_sensor_queues()
            get_prepare()
            time.sleep(0.1)
            continue
        if recv_prepare and not start_test:
            drain_sensor_queues()
            if not map_ready or map_loading:
                time.sleep(1.0)
                continue
            prepare()
            pre_map_name = map_name
            map_ready = False
            time.sleep(0.1)
            continue
        if hold_until_ego_ready():
            drain_sensor_queues()
            get_vehicle_feedback()
            time.sleep(0.02)
            continue
        if hold_until_global_plan_ready():
            drain_sensor_queues()
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
        # get_image()
        # 注意：相机通道不在这里 drain。get_camera_detection() 内部会自己把
        # 通道排空到最新一帧再送入检测，这样既能避免积压导致的画面滞后，
        # 又不会在还没准备好推理时就提前丢帧。
        drain_sensor_queues(drain_camera=obstacle_source != "camera")
        cmd, done_out = get_detection_msg(dsac_main, notify)
        if done_out:
            print("done_out is True, aborting test")
            abort_test()
            continue
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

            send_control_cmd(cmd.acc, cmd.speed, cmd.steer)
        get_vehicle_feedback()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_center", type=str, default="47.110.233.70:52009")
    arg_parser.add_argument("--field_id", type=str, default="field-zd-test1-22-0331134113-874")
    arg_parser.add_argument("--net_interface", type=str, default="enxaa9f27179018")
    arg_parser.add_argument(
        "--obstacle-source",
        "--obstacle_source",
        dest="obstacle_source",
        choices=("camera", "npc_truth"),
        default=obstacle_source,
        help=(
            "obstacle input gate: camera runs Sparse4D; npc_truth uses the "
            "simulator NPC ground-truth multicast channel"
        ),
    )
    args = arg_parser.parse_args()
    obstacle_source = _normalize_obstacle_source(args.obstacle_source)
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
    param.client_name = "apollo_testee"
    param.recv_self_msg = False
    session_id = ""
    print('local_ip',local_ip)
    channels = libMulticastNetwork.ChannelPtrVector()
    ret = libMulticastNetwork.create_channels(param, channels)
    if ret:
        print("create channels failed, ret: {}".format(ret))
        sys.exit(1)
    channel_map = {}
    for channel in channels:
        print(
            "message channel name: {}, id: {}".format(
                channel.name(), channel.id()
            )
        )
        channel_map[channel.name()] = channel

    pointcloud_channel = channel_map["lidar"]
    notify_channel = channel_map["notify"]
    cmd_channel = channel_map["vehiclecontrol"]
    prepare_channel = channel_map["prepare"]
    ins_channel = channel_map["ins"]
    image_channel = channel_map["camera"]
    npc_channel = channel_map.get("npc")
    configured_npc_channel = str(
        os.environ.get("E2E_NPC_TRUTH_CHANNEL", "")
    ).strip()
    # test.py/test1.py establish that simulator truth is carried by the
    # multicast channel named "npc". Keep it first even if an old shell still
    # exports E2E_NPC_TRUTH_CHANNEL=vts/perception/obstacles.
    npc_channel_candidates = list(NPC_TRUTH_CHANNEL_NAMES)
    if (
        configured_npc_channel
        and configured_npc_channel not in npc_channel_candidates
    ):
        npc_channel_candidates.append(configured_npc_channel)

    npc_channel = None
    npc_truth_channel_name = ""
    for candidate_name in npc_channel_candidates:
        if candidate_name in channel_map:
            npc_channel = channel_map[candidate_name]
            npc_truth_channel_name = candidate_name
            break
    if obstacle_source == "npc_truth" and npc_channel is None:
        raise RuntimeError(
            "obstacle source npc_truth selected, but none of the multicast "
            "channels {} is available; available={}".format(
                npc_channel_candidates,
                sorted(channel_map.keys()),
            )
        )
    # if obstacle_source == "npc_truth":
        # print(
        #     "[npc-truth][CONFIG] selected multicast channel "
        #     f"name={npc_truth_channel_name!r} "
        #     f"id={npc_channel.id()} "
        #     f"candidates={npc_channel_candidates} "
        #     f"config_center={args.config_center!r} "
        #     f"field_id={args.field_id!r} "
        #     f"local_ip={local_ip!r} "
        #     f"interface={args.net_interface!r} "
        #     f"configured_override={configured_npc_channel!r} "
        #     f"debug={NPC_TRUTH_DEBUG}",
        #     flush=True,
        # )
        # print(
        #     "[npc-truth][CONFIG] available_channels="
        #     + repr(
        #         sorted(
        #             (name, channel.id())
        #             for name, channel in channel_map.items()
        #         )
        #     ),
        #     flush=True,
        # )

    if not libMulticastNetwork.InitImageDecoder(6,1600,900):
        print("image decoder init error")
        sys.exit(1)

    img_id = 0
    save_results = False
    recv_prepare = False
    start_test = False
    actor_id = "apollo_testee"
    role_id = "apollo_testee"

    use_epre_dsac = True
    
    dsac_main = Main(use_epre_dsac = use_epre_dsac)
    model = Predictor()
    model.set_obstacle_source(obstacle_source)
    pre_map_name = None
    map_name = None

    # ===== Sparse4Dv3 替换 PointPillars：改动 begin =====
    print(f"[obstacle-source] active={obstacle_source}", flush=True)
    if obstacle_source == "camera":
        print(s4d_cfg.summary(), flush=True)
        sparse4d_detector = LocalSparse4DDetector(
            sparse4d_root=s4d_cfg.SPARSE4D_ROOT,
            config_path=s4d_cfg.CONFIG_PATH,
            checkpoint_path=s4d_cfg.CHECKPOINT_PATH,
            device=s4d_cfg.DEVICE,
            score_threshold=s4d_cfg.SCORE_THRESHOLD,
            calib_file=s4d_cfg.CALIB_PATH or None,
            cam_order=DEFAULT_CAM_ORDER,
            camera_index_order=s4d_cfg.CAMERA_INDEX_ORDER,
            lidar2ego=DEFAULT_LIDAR_TO_EGO,
            camera_width=s4d_cfg.CAMERA_WIDTH,
            camera_height=s4d_cfg.CAMERA_HEIGHT,
            channel_order=s4d_cfg.CAMERA_CHANNEL_ORDER,
            use_fp16=s4d_cfg.USE_FP16,
            vis_mode=s4d_cfg.VIS_MODE,
            vis_out_dir=s4d_cfg.VIS_OUT_DIR,
            vis_bev_range=s4d_cfg.VIS_BEV_RANGE,
            vis_bev_size=s4d_cfg.VIS_BEV_SIZE,
            vis_cam_scale=s4d_cfg.VIS_CAM_SCALE,
        )
        print(
            "[sparse4d] local detector ready "
            f"cams={sparse4d_detector.cam_order}"
        )
    else:
        sparse4d_detector = None

    # ===== Sparse4Dv3 替换 PointPillars：改动 end =====

    ins_receiver_thread = start_ins_receiver_thread()
    try:
        main()
    finally:
        ins_receiver_stop_event.set()
        ins_receiver_thread.join(timeout=1.0)
        if sparse4d_detector is not None:
            sparse4d_detector.close()
