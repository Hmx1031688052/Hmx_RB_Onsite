import math
import os
import sys
import time
import torch
import xml.etree.ElementTree as ET
from enum import Enum


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
        if os.path.isdir(os.path.join(base, "modules")):
            if base not in sys.path:
                sys.path.insert(0, base)
            return
        e2e_base = os.path.join(base, "e2e")
        if os.path.isdir(os.path.join(e2e_base, "modules")):
            if e2e_base not in sys.path:
                sys.path.insert(0, e2e_base)
            return


_setup_project_paths()

from mmdet3d.apis import inference_detector, init_model
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random
try:
    from speed_limits import scene_speed_limit_for_map as shared_scene_speed_limit_for_map
except ImportError:
    from .speed_limits import scene_speed_limit_for_map as shared_scene_speed_limit_for_map
try:
    from perception_web_visualizer import PerceptionWebVisualizer
except ImportError:
    from .perception_web_visualizer import PerceptionWebVisualizer
try:
    from stable_obstacle_tracker import StableObstacleTracker
except ImportError:
    from .stable_obstacle_tracker import StableObstacleTracker

try:
    import rclpy
    from sensor_msgs.msg import Imu
    from visualization_msgs.msg import Marker, MarkerArray
    try:
        from nav_msgs.msg import Path
    except ImportError:
        Path = None
    try:
        from veh_interfaces.msg import Datarecord
    except ImportError:
        Datarecord = None
except ImportError:
    rclpy = None
    Imu = None
    Marker = None
    MarkerArray = None
    Path = None
    Datarecord = None


class RoleType(Enum):
    """Perception role labels kept independent of the old C++ planner."""

    VEHICLE = 1
    PEDESTRIAN = 2
    UNKNOWN = 3


class Box2d(object):
    """Mutable structured object shared by perception and the rule planner."""

    pass


class VehicleFeedbackState(object):
    """Small Python replacement for the former planner binding type."""

    pass


class ControlCommand(object):
    """Legacy bridge command container; the new run path does not instantiate it."""

    pass


PERCEPTION_THREE_CLASS = os.environ.get("PERCEPTION_THREE_CLASS", "1") == "1"


def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def GetYawfromQuaternion(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# def transform(xyz): #这个部分我需要看看是否这个坐标系的转换是否有问题。是否需要改变。旧参数
#     """
#     lidar坐标系转换为车身坐标系
#     """
#     # lidar extrinsics，可以是其他值，参考simulator的传感器配置
#     transform_coeff = {
#         "x": 1.78832,
#         "y": 0.00,
#         "z": 1.24129,
#         "pitch": 0.0,
#         "yaw": 0.0,
#         "roll": 0.0,
#     }
#     xyz[0] -= transform_coeff["x"]  #这里到底是还是减？
#     xyz[1] -= transform_coeff["y"]
#     xyz[2] -= transform_coeff["z"]
#     return xyz

def transform(xyz):
    transform_coeff = {
        "x": 0.1,
        "y": 0.013,
        "z": 1.755,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }
    xyz[0] += transform_coeff["x"]
    xyz[1] += transform_coeff["y"]
    xyz[2] += transform_coeff["z"]
    return xyz

KITTI_LABEL_PEDESTRIAN = 0
KITTI_LABEL_CYCLIST = 1
KITTI_LABEL_VEHICLE = 2

OBS_TYPE_VEHICLE = 1
OBS_TYPE_CYCLIST = 3
OBS_TYPE_PEDESTRIAN = 4


def detector_label_to_obstacle_type(label):
    try:
        label = int(label)
    except Exception:
        return OBS_TYPE_VEHICLE
    if label == KITTI_LABEL_PEDESTRIAN:
        return OBS_TYPE_PEDESTRIAN
    if label == KITTI_LABEL_CYCLIST:
        return OBS_TYPE_CYCLIST
    return OBS_TYPE_VEHICLE


def obstacle_type_to_role_type(obstacle_type):
    if obstacle_type == OBS_TYPE_PEDESTRIAN:
        return RoleType.PEDESTRIAN
    if obstacle_type == OBS_TYPE_CYCLIST:
        return RoleType.UNKNOWN
    return RoleType.VEHICLE

class Command:
    pass

STEERING_RATIO = 11.0
WHEEL_BASE_M = 3.38
MAX_STEERING_WHEEL_DEG = 42.0


def limit_steer_command_by_speed(steering_wheel_deg, speed):
    """Limit a steering-wheel-angle command expressed in degrees.

    Adjust these constants directly when tuning:
    - min_speed: below this speed, keep the original steering command.
    - lat_acc_limit: smaller means stricter high-speed steering.
    The input and return value are steering-wheel degrees.
    """
    min_speed = 8.0
    lat_acc_limit = 2.5

    steer = max(
        -MAX_STEERING_WHEEL_DEG,
        min(MAX_STEERING_WHEEL_DEG, float(steering_wheel_deg)),
    )
    speed = max(0.0, float(speed or 0.0))
    if speed < min_speed:
        return steer

    max_front_wheel_rad = math.atan(
        lat_acc_limit * WHEEL_BASE_M / max(speed * speed, 1e-6)
    )
    max_steer = min(
        MAX_STEERING_WHEEL_DEG,
        math.degrees(max_front_wheel_rad) * STEERING_RATIO,
    )
    return max(-max_steer, min(max_steer, steer))


def limit_target_speed(speed):
    """Limit all planning/control target speeds to reduce control oscillation."""
    max_speed = 33.0
    return max(0.0, min(max_speed, float(speed or 0.0)))


def compute_safety_cost(
    collision_done, near_collision=False, min_ttc=None, ttc_threshold=1.5
):
    """Compute a binary cost independently from the reward signal."""
    ttc_risk = min_ttc is not None and np.isfinite(min_ttc) and min_ttc < ttc_threshold
    return float(bool(collision_done) or bool(near_collision) or bool(ttc_risk))


def scene_speed_limit_for_map(map_file):
    return shared_scene_speed_limit_for_map(map_file)


def ensure_planner_compatible_map(map_path):
    try:
        tree = ET.parse(map_path)
    except ET.ParseError:
        return map_path

    root = tree.getroot()
    changed = False
    for road in root.findall("road"):
        if not road.get("name"):
            road.set("name", "road_{}".format(road.get("id", "unknown")))
            changed = True

    if not changed:
        return map_path

    cache_dir = os.path.join(os.path.dirname(map_path), ".planner_cache")
    os.makedirs(cache_dir, exist_ok=True)
    patched_path = os.path.join(cache_dir, os.path.basename(map_path))
    tree.write(patched_path, encoding="UTF-8", xml_declaration=True)
    return patched_path


class RosPlanningBridge:
    def __init__(self):
        self.enabled = False
        self.node = None
        self.ego_pub = None
        self.obstacles_pub = None
        self.rl_pub = None
        self.map_name=''
        self.ctrl_sub = None
        self.global_path_sub = None
        self.glo_path_sub = None
        self.latest_ctrl = None
        self.latest_ctrl_wall_time = 0.0
        self.latest_global_path = None
        self.latest_global_path_wall_time = 0.0
        self.ctrl_timeout = float(os.environ.get("E2E_ROS_CTRL_TIMEOUT", "0.5"))
        self.ctrl_hold_timeout = float(os.environ.get("E2E_ROS_CTRL_HOLD_TIMEOUT", "2.0"))
        self.ego_speed_hold_timeout = float(os.environ.get("E2E_ROS_EGO_SPEED_HOLD_TIMEOUT", "0.8"))
        self.ego_pose_speed_max = float(os.environ.get("E2E_ROS_EGO_POSE_SPEED_MAX", "80.0"))
        self.steer_scale = float(os.environ.get("E2E_ROS_STEER_SCALE", "1.0"))
        self.rl_lateral_limit = float(os.environ.get("E2E_RL_LATERAL_LIMIT", "5.0"))
        # Hot-path console I/O can delay the INS publisher thread and point
        # cloud processing. Keep it opt-in while retaining warning/error logs.
        self.debug_hot_path = os.environ.get("E2E_DEBUG_HOT_PATH", "0") == "1"
        self.last_ros_control_cmd = None
        self.last_valid_ego_speed = 0.0
        self.last_valid_ego_speed_wall_time = 0.0
        self.last_published_ego_speed = 0.0
        self.last_published_ego_speed_wall_time = 0.0
        self.last_ego_speed_pose = None
        self.last_ego_speed_pose_wall_time = 0.0
        self.last_published_ego_sample_id = None
        self.last_invalid_ego_position_wall_time = 0.0
        self.speed_pid_integral = 0.0
        self.speed_pid_last_error = 0.0
        self.speed_pid_last_time = None
        self._disable_reason = None

        if os.environ.get("E2E_ENABLE_ROS_PLANNING", "1") == "0":
            self._disable_reason = "disabled by E2E_ENABLE_ROS_PLANNING=0"
            return
        if rclpy is None:
            self._disable_reason = "rclpy is not available"
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node("wutfsd_predictor_planning_bridge")
            self.ego_pub = self.node.create_publisher(Imu, "/global_info", 10)
            self.obstacles_pub = self.node.create_publisher(MarkerArray, "/obs_info_local", 10)
            self.rl_pub = self.node.create_publisher(Imu, "/RL_ctrl_info", 10)
            self.ctrl_sub = self.node.create_subscription(
                Imu,
                "/ctrl_info",
                self._on_ctrl,
                10,
            )
            if Datarecord is not None:
                self.global_path_sub = self.node.create_subscription(
                    Datarecord,
                    "/global_path",
                    self._on_global_path_record,
                    10,
                )
            if Path is not None:
                self.glo_path_sub = self.node.create_subscription(
                    Path,
                    "/glo_path",
                    self._on_global_path_msg,
                    10,
                )
            self.enabled = True
            print("[ros-bridge] enabled: /global_info /obs_info_local /RL_ctrl_info -> /ctrl_info")
        except Exception as exc:
            self._disable_reason = str(exc)
            self.enabled = False
            print(f"[ros-bridge] disabled: {exc}")

    def spin_once(self, timeout_sec=0.0):
        if not self.enabled:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        except Exception as exc:
            self.enabled = False
            self._disable_reason = str(exc)
            print(f"[ros-bridge] spin failed, disabled: {exc}")

    def _resolve_ego_speed(self, ego, ego_speed, ego_x, ego_y):#这个函数主要是用来对速度更新不准确时候的来调用。
        now = time.time()
        raw_speed = ego_speed if ego_speed is not None else getattr(ego, "speed", 0.0)
        try:
            raw_speed = float(raw_speed)
        except Exception:
            raw_speed = float("nan")

        speed_eps = 0.05
        pose_speed = None
        if self.last_ego_speed_pose is not None and self.last_ego_speed_pose_wall_time > 0.0:
            dt = now - self.last_ego_speed_pose_wall_time
            if 1e-3 <= dt <= 1.0:
                last_x, last_y = self.last_ego_speed_pose
                candidate = math.hypot(ego_x - last_x, ego_y - last_y) / dt
                if math.isfinite(candidate) and candidate <= self.ego_pose_speed_max:
                    pose_speed = candidate
        self.last_ego_speed_pose = (ego_x, ego_y)
        self.last_ego_speed_pose_wall_time = now

        if math.isfinite(raw_speed) and raw_speed > speed_eps:
            speed = max(0.0, raw_speed)
            self._remember_valid_ego_speed(speed, now)
            return self._remember_published_ego_speed(speed, now)

        if self.last_valid_ego_speed_wall_time > 0.0:
            age = now - self.last_valid_ego_speed_wall_time
            if age <= self.ego_speed_hold_timeout and self.last_valid_ego_speed > speed_eps:
                return self._remember_published_ego_speed(self.last_valid_ego_speed, now)

        if pose_speed is not None and pose_speed > speed_eps:
            speed = max(0.0, pose_speed)
            self._remember_valid_ego_speed(speed, now)
            return self._remember_published_ego_speed(speed, now)

        speed = max(0.0, raw_speed) if math.isfinite(raw_speed) else 0.0
        return self._remember_published_ego_speed(speed, now)

    def _remember_valid_ego_speed(self, speed, now):
        if math.isfinite(speed) and speed >= 0.0:
            self.last_valid_ego_speed = float(speed)
            self.last_valid_ego_speed_wall_time = now

    def _remember_published_ego_speed(self, speed, now):
        speed = max(0.0, float(speed))
        self.last_published_ego_speed = speed
        self.last_published_ego_speed_wall_time = now
        return speed

    def _control_ego_speed(self, ego_speed):
        try:
            speed = float(ego_speed)
        except Exception:
            speed = float("nan")
        if math.isfinite(speed) and speed > 0.05:
            return max(0.0, speed)
        now = time.time()
        if (
            self.last_published_ego_speed_wall_time > 0.0
            and now - self.last_published_ego_speed_wall_time <= self.ego_speed_hold_timeout
        ):
            return max(0.0, self.last_published_ego_speed)
        return max(0.0, speed) if math.isfinite(speed) else 0.0

    def _reset_ego_speed_hold(self):
        self.last_valid_ego_speed = 0.0
        self.last_valid_ego_speed_wall_time = 0.0
        self.last_published_ego_speed = 0.0
        self.last_published_ego_speed_wall_time = 0.0
        self.last_ego_speed_pose = None
        self.last_ego_speed_pose_wall_time = 0.0
        self.last_invalid_ego_position_wall_time = 0.0
        self.last_published_ego_sample_id = None

    def _valid_ego_position(self, ego_x, ego_y):
        if not math.isfinite(ego_x) or not math.isfinite(ego_y):
            return False

        map_name = str(getattr(self, "map_name", "") or "")
        map_name = map_name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
        if map_name and not map_name.endswith(".xodr"):
            map_name += ".xodr"

        if map_name == "aitownreconstructed_v0103_200518.xodr":
            # AITown 使用 UTM 大坐标，允许 x≈78万、y≈335万。
            return abs(ego_x) <= 10000000.0 and abs(ego_y) <= 10000000.0

        if abs(ego_x) > 1000.0 or abs(ego_y) > 1000.0:
            return False

        return True

    def publish_ego_state(
        self,
        ego,
        vehicle_feedback,
        ego_speed=None,
        ego_vy=0.0,
        ego_yawrate=0.0,
        ego_acc=0.0,
        latitude=0.0,
        longitude=0.0,
        heading_deg=None,
        ego_sample_id=None,
    ):
        if not self.enabled or ego is None:
            return False

        ego_x = float(getattr(ego, "x", 0.0))
        ego_y = float(getattr(ego, "y", 0.0))
        if not self._valid_ego_position(ego_x, ego_y):
            now = time.time()
            if now - self.last_invalid_ego_position_wall_time > 1.0:
                self.last_invalid_ego_position_wall_time = now
                print(f"[ros-bridge] invalid ego position: x={ego_x} y={ego_y}")
            return False

        now_msg = self.node.get_clock().now().to_msg()
        resolved_ego_speed = self._resolve_ego_speed(ego, ego_speed, ego_x, ego_y)

        publish_ego = ego_sample_id is None or ego_sample_id != self.last_published_ego_sample_id
        if publish_ego:
            ego_msg = Imu()
            ego_msg.header.stamp = now_msg
            ego_msg.header.frame_id = "map"
            # /global_info unit contract (legacy Imu field packing):
            # x/y [m, map], yaw [rad], vx/vy [m/s, vehicle],
            # yaw rate [rad/s], steering feedback [steering-wheel deg].
            ego_msg.orientation.x = ego_x
            ego_msg.orientation.y = ego_y
            ego_msg.orientation.z = float(ego.theta)
            ego_msg.orientation.w = float(resolved_ego_speed)
            ego_msg.linear_acceleration.x = float(ego_vy)
            ego_msg.linear_acceleration.y = float(ego_yawrate)
            ego_msg.linear_acceleration.z = float(ego_acc)
            ego_msg.angular_velocity.x = float(getattr(vehicle_feedback, "steering_wheel_angle", 0.0))
            ego_msg.orientation_covariance[0] = float(latitude)
            ego_msg.orientation_covariance[1] = float(longitude)
            ego_msg.orientation_covariance[2] = float(
                math.degrees(ego.theta) if heading_deg is None else heading_deg
            )
            ego_msg.orientation_covariance[3] = 1.0
            self.ego_pub.publish(ego_msg)
            self.last_published_ego_sample_id = ego_sample_id
            return True
        return False

    def publish_obstacles(self, obstacles):
        if not self.enabled:
            return

        now_msg = self.node.get_clock().now().to_msg()
        obs_msg = MarkerArray()
        for index, ob in enumerate(obstacles or []):
            marker = Marker()
            marker.header.stamp = now_msg
            marker.header.frame_id = "map"
            marker.id = self._obstacle_id(ob, index)
            marker.type = self._obstacle_type(ob)
            marker.action = Marker.ADD
            marker.pose.position.x = float(ob.x)
            marker.pose.position.y = float(ob.y)
            marker.pose.position.z = float(max(0.0, getattr(ob, "speed", 0.0)) * 3.6)
            marker.pose.orientation.x = float(getattr(ob, "theta", 0.0))
            # Marker is legacy transport here: x carries yaw already, so y
            # explicitly carries whether z is a measured/filtered speed.
            # Downstream may conservatively treat an invalid speed as zero.
            marker.pose.orientation.y = 1.0 if getattr(ob, "speed_valid", False) else 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(max(0.1, getattr(ob, "length", 0.0)))
            marker.scale.y = float(max(0.1, getattr(ob, "width", 0.0)))
            marker.scale.z = 1.0
            obs_msg.markers.append(marker)
        self.obstacles_pub.publish(obs_msg)

    def publish_rl_decision(self, target_speed, lateral_offset):
        if not self.enabled:
            return

        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.orientation.x = float(limit_target_speed(target_speed))
        msg.orientation.y = float(max(-self.rl_lateral_limit, min(self.rl_lateral_limit, lateral_offset)))
        self.rl_pub.publish(msg)

    def get_control_command(self, ego_speed):
        if not self.enabled:
            return None
        self.spin_once(0.0)
        ego_speed = self._control_ego_speed(ego_speed)
        if self.latest_ctrl is None:
            return self.last_ros_control_cmd
        ctrl_age = time.time() - self.latest_ctrl_wall_time
        if ctrl_age > self.ctrl_hold_timeout:
            return None
        if ctrl_age > self.ctrl_timeout:
            return self.last_ros_control_cmd

        target_speed = limit_target_speed(self.latest_ctrl["speed"])
        brake = max(0.0, self.latest_ctrl["brake"])
        if brake > 0.0:
            acc = -self._normalize_brake_pressure(brake)
            self._reset_speed_pid()
        else:
            acc = self._speed_pid_acc(target_speed, ego_speed)
            if self.debug_hot_path:
                print(
                    f"[ros-bridge] ego_speed={ego_speed:.3f} "
                    f"target_speed={target_speed:.3f} acc={acc:.3f}"
                )
        cmd = ControlCommand()
        # /ctrl_info orientation.x and ControlCommand.steer are both
        # steering-wheel angles in degrees.  No hidden rad/percentage scale.
        cmd.steer = limit_steer_command_by_speed(
            self.latest_ctrl["steer"] * self.steer_scale,
            ego_speed,
        )
        cmd.speed = limit_target_speed(target_speed)
        cmd.acc = acc
        self.last_ros_control_cmd = cmd
        return cmd

    def clear_control(self):
        self.latest_ctrl = None
        self.latest_ctrl_wall_time = 0.0
        self.last_ros_control_cmd = None
        self._reset_ego_speed_hold()
        self._reset_speed_pid()

    def _normalize_brake_pressure(self, brake):
        max_brake_pressure = 25.0
        brake = max(0.0, float(brake or 0.0))
        return max(0.0, min(1.0, brake / max_brake_pressure))

    def _speed_pid_acc(self, target_speed, ego_speed):
        kp = 0.45
        ki = 0.02
        kd = 0.03
        integral_limit = 10.0

        now = time.time()
        if self.speed_pid_last_time is None:
            dt = 0.05
        else:
            dt = max(1e-3, min(0.2, now - self.speed_pid_last_time))

        error = float(target_speed) - max(0.0, float(ego_speed or 0.0))
        self.speed_pid_integral = max(
            -integral_limit,
            min(integral_limit, self.speed_pid_integral + error * dt),
        )
        derivative = (error - self.speed_pid_last_error) / dt
        acc = kp * error + ki * self.speed_pid_integral + kd * derivative

        self.speed_pid_last_error = error
        self.speed_pid_last_time = now
        return max(-1.0, min(1.0, acc))

    def _reset_speed_pid(self):
        self.speed_pid_integral = 0.0
        self.speed_pid_last_error = 0.0
        self.speed_pid_last_time = None

    def _on_ctrl(self, msg):
        self.latest_ctrl = {
            "steer": float(msg.orientation.x),
            "speed": float(msg.orientation.y),
            "brake": float(msg.orientation.z),
            "mode": float(msg.orientation.w),
            "turn_light": float(msg.linear_acceleration.x),
        }
        self.latest_ctrl_wall_time = time.time()

    def get_global_path(self):
        if self.latest_global_path is None:
            return None
        path = {}
        for key, value in self.latest_global_path.items():
            if isinstance(value, list):
                path[key] = list(value)
            else:
                path[key] = value
        return path

    def clear_global_path(self):
        self.latest_global_path = None
        self.latest_global_path_wall_time = 0.0

    def _on_global_path_record(self, msg):
        try:
            count = int(getattr(msg, "p1", 0) or 0)
        except Exception:
            count = 0
        xs = list(getattr(msg, "data1", []))
        ys = list(getattr(msg, "data2", []))
        yaws = list(getattr(msg, "data3", []))
        kappas = list(getattr(msg, "data4", []))
        stations = list(getattr(msg, "data5", []))
        if count <= 0:
            count = min(len(xs), len(ys))
        count = min(count, len(xs), len(ys))
        if count < 2:
            return

        frame_id = ""
        header = getattr(msg, "header", None)
        if header is not None:
            frame_id = getattr(header, "frame_id", "") or ""
        self._store_global_path(
            xs[:count],
            ys[:count],
            yaws[:count],
            kappas[:count],
            stations[:count],
            frame_id,
            "/global_path",
        )

    def _on_global_path_msg(self, msg):
        xs = []
        ys = []
        for pose_stamped in getattr(msg, "poses", []):
            pose = getattr(pose_stamped, "pose", None)
            position = getattr(pose, "position", None)
            if position is None:
                continue
            xs.append(getattr(position, "x", 0.0))
            ys.append(getattr(position, "y", 0.0))
        if len(xs) < 2:
            return
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
        self._store_global_path(xs, ys, None, None, None, frame_id, "/glo_path")

    def _store_global_path(self, xs, ys, yaws=None, kappas=None, stations=None, frame_id="", source=""):
        points = []
        for x, y in zip(xs or [], ys or []):
            try:
                x = float(x)
                y = float(y)
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if points and math.hypot(x - points[-1][0], y - points[-1][1]) < 1e-6:
                continue
            points.append((x, y))
        if len(points) < 2:
            return

        path_x = [p[0] for p in points]
        path_y = [p[1] for p in points]
        count = len(path_x)
        path_yaw = self._path_float_list(yaws, count)
        path_kappa = self._path_float_list(kappas, count, default=0.0)
        path_s = self._path_float_list(stations, count)

        if path_yaw is None:
            path_yaw = self._compute_path_yaw(path_x, path_y)
        if path_s is None or (len(path_s) > 1 and path_s[-1] <= path_s[0]):
            path_s = self._compute_path_station(path_x, path_y)
        if path_kappa is None:
            path_kappa = [0.0] * count

        self.latest_global_path = {
            "x": path_x,
            "y": path_y,
            "yaw": path_yaw,
            "kappa": path_kappa,
            "s": path_s,
            "frame_id": frame_id,
            "source": source,
            "stamp": time.time(),
        }
        self.latest_global_path_wall_time = time.time()

    def _path_float_list(self, values, count, default=None):
        if values is None:
            return None
        result = []
        for index in range(count):
            try:
                value = float(values[index])
            except Exception:
                if default is None:
                    return None
                value = default
            if not math.isfinite(value):
                if default is None:
                    return None
                value = default
            result.append(value)
        return result

    def _compute_path_station(self, xs, ys):
        stations = [0.0]
        for index in range(1, len(xs)):
            stations.append(
                stations[-1] + math.hypot(xs[index] - xs[index - 1], ys[index] - ys[index - 1])
            )
        return stations

    def _compute_path_yaw(self, xs, ys):
        yaws = []
        count = len(xs)
        for index in range(count):
            if index < count - 1:
                dx = xs[index + 1] - xs[index]
                dy = ys[index + 1] - ys[index]
            else:
                dx = xs[index] - xs[index - 1]
                dy = ys[index] - ys[index - 1]
            yaws.append(math.atan2(dy, dx))
        return yaws

    def _obstacle_id(self, ob, index):
        try:
            return int(ob.id)
        except Exception:
            return index + 1

    def _obstacle_type(self, ob):
        obs_type = getattr(ob, "obs_type", None)
        if obs_type is not None:
            try:
                obs_type = int(obs_type)
                if obs_type in (OBS_TYPE_VEHICLE, OBS_TYPE_CYCLIST, OBS_TYPE_PEDESTRIAN):
                    return obs_type
            except Exception:
                pass

        role_type = getattr(ob, "roleType", None)
        if role_type == RoleType.VEHICLE or str(role_type) == "RoleType.VEHICLE":
            return OBS_TYPE_VEHICLE
        if role_type == RoleType.PEDESTRIAN or str(role_type) == "RoleType.PEDESTRIAN":
            return OBS_TYPE_PEDESTRIAN
        if role_type == RoleType.UNKNOWN or str(role_type) == "RoleType.UNKNOWN":
            return OBS_TYPE_CYCLIST
        return OBS_TYPE_VEHICLE


class Predictor:
    def __init__(self):
        if PERCEPTION_THREE_CLASS:
            config = "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
            checkpoint = (
                "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
            )
        else:
            config = "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
            checkpoint = (
                "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
            )
        self.config = config
        self.checkpoint = checkpoint
        self.detector_device = os.environ.get("DETECT_DEVICE", "cuda:0")
        self.perception_source = os.environ.get(
            "E2E_PERCEPTION_SOURCE", "gt"
        ).strip().lower()
        if self.perception_source == "lidar":
            self.model = init_model(
                config,
                checkpoint,
                device=self.detector_device,
            )
        else:
            self.model = None
            print(
                "[perception] source=GT; PointPillars model loading skipped"
            )
        self.perception_web_visualizer = None
        if os.environ.get("E2E_PERCEPTION_WEB", "1") == "1":
            self.perception_web_visualizer = PerceptionWebVisualizer(
                host=os.environ.get("E2E_PERCEPTION_WEB_HOST", "0.0.0.0"),
                port=int(os.environ.get("E2E_PERCEPTION_WEB_PORT", "8765")),
                max_points=int(
                    os.environ.get("E2E_PERCEPTION_WEB_MAX_POINTS", "7000")
                ),
                gt_only=(
                    os.environ.get(
                        "E2E_PERCEPTION_WEB_GT_ONLY", "1"
                    )
                    == "1"
                ),
            )
            if not self.perception_web_visualizer.start():
                self.perception_web_visualizer = None
        self.default_map = "AITownReconstructed_V0103_200518.xodr"
        self.ego = None
        self.vehicle_feedback = VehicleFeedbackState()
        self.vehicle_feedback.steering_wheel_angle = 0.0
        self.vehicle_feedback.brake_pedal_position = 0.0
        self.vehicle_feedback.accelerator_pedal_position = 0.0
        # Vehicle control is owned exclusively by RuleBasedPlanner and
        # StableController in run.py.  Predictor only returns perception data.
        self.ros_bridge = None
        self.ros_control_debug_last_time = 0.0
        self.ego_ros_speed = 0.0
        self.ego_ros_vy = 0.0
        self.ego_ros_yawrate = 0.0
        self.ego_ros_acc = 0.0
        self.ego_ros_ax = 0.0
        self.ego_ros_ay = 0.0
        self.ego_lon_acc = 0.0
        self.ego_lat_acc = 0.0
        self.ego_ros_latitude = 0.0
        self.ego_ros_longitude = 0.0
        self.ego_ros_heading_deg = None
        self.ego_ros_sequence = None
        self.last_ego_update_wall_time = None
        self.last_ego_update_sequence = None
        self.previous_ego_yaw = None
        self.previous_ego_yaw_wall_time = None
        self.collision = 0
        self.start = 0
        self.time_pre = None
        self.action0 = 0
        self.pre_obstacles = []
        self.pre_pre_obstaclea = []
        self.pre_pre_pre_obstaclea = []
        # World-frame obstacle velocity state.  It is intentionally separate
        # from detector/association state: only a stable final track ID with
        # two timestamped measurements may publish a valid speed.
        self.obstacle_velocity_tracks = {}
        self.rule_obstacle_tracks = {}
        self.rule_next_obstacle_id = 1
        self.perception_confidence_thresholds = {
            KITTI_LABEL_PEDESTRIAN: max(
                0.0,
                min(
                    1.0,
                    float(os.environ.get("PERCEPTION_PEDESTRIAN_CONF", "0.30")),
                ),
            ),
            KITTI_LABEL_CYCLIST: max(
                0.0,
                min(
                    1.0,
                    float(os.environ.get("PERCEPTION_CYCLIST_CONF", "0.25")),
                ),
            ),
            KITTI_LABEL_VEHICLE: max(
                0.0,
                min(
                    1.0,
                    float(os.environ.get("PERCEPTION_VEHICLE_CONF", "0.55")),
                ),
            ),
        }
        self.stable_obstacle_tracker = StableObstacleTracker(
            confirmation_hits=int(
                os.environ.get("E2E_TRACK_CONFIRM_HITS", "2")
            ),
            coast_time=float(os.environ.get("E2E_TRACK_COAST_SEC", "0.60")),
            max_age=float(os.environ.get("E2E_TRACK_MAX_AGE_SEC", "1.20")),
        )
        self.last_detection_debug_wall_time = 0.0
        self.last_tracking_debug_wall_time = 0.0
        self.obstacle_velocity_min_dt_s = 0.02
        self.obstacle_velocity_max_dt_s = 0.50
        self.obstacle_velocity_max_accel_mps2 = 8.0
        self.obstacle_velocity_filter_alpha = 0.35
        self.obstacle_velocity_track_timeout_s = 1.0
        self.latest_obstacle_measurement_stamp_s = None
        self.ego_pre_x = 0
        self.ego_pre_y = 0
        self.ego_pre2_x = 0
        self.ego_pre2_y = 0
        self.total_episode = 0
        self.k = 0
        self.time1 = []
        self.time3 = []
        self.timeall = []
        self.time_out = False
        self.collision_xishu = 1
        self.step = 0
        self.ok = 0
        self.last_ego = None
        self.invalid_ego_sample_count = 0
        self.last_invalid_ego_sample_warn_wall_time = 0.0
        self.near_origin_ego_reject_count = 0
        self.last_near_origin_ego_reject_warn_wall_time = 0.0
        self.near_origin_ego_radius = float(os.environ.get("E2E_EGO_NEAR_ORIGIN_RADIUS", "0.5"))
        self.near_origin_ego_jump_min = float(os.environ.get("E2E_EGO_NEAR_ORIGIN_JUMP_MIN", "20.0"))
        self.obs_gap = 0
        self.obstacle_hold_max_sec = float(os.environ.get("E2E_OBS_CACHE_HOLD_SEC", "0.8"))
        self.cached_obstacles = []
        self.cached_obstacles_wall_time = 0.0
        self.last_pointclouds = []
        self.planning_step = 0
        self.last_steer = 0
        self.start_time = 0
        self.weather = {'rain':0, 'fog':0, 'cloud':0,'snow':0}
        self.last_time = None
        self.last_a = None
        self.last_h_a = None
        self.last_rot = None
        self.frenet_reward_last_s = None
        self.frenet_reward_last_time = None
        self.frenet_reward_last_lon_acc = None
        self.frenet_reward_last_lat_action = None
        self.frenet_reward_info = {}
        self.goal = 0
        self.obstacles_id_dict = {'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'10':{},'11':{},'12':{},'13':{},'14':{},'15':{},'16':{},'17':{},'18':{},'19':{},'20':{},'21':{},'22':{},'23':{},'24':{},'25':{},'26':{},'27':{},'28':{},'29':{},'30':{},'31':{},'32':{},'33':{},'34':{},'35':{},'36':{},'37':{},'38':{},'39':{},'40':{}}
        self.debug_sync = os.environ.get("E2E_DEBUG_SYNC", "0") == "1"
        self.debug_obs = os.environ.get("E2E_DEBUG_OBS", "0") == "1"
        self.debug_hot_path = os.environ.get("E2E_DEBUG_HOT_PATH", "0") == "1"
        self.debug_visualize_block = os.environ.get("E2E_DEBUG_VIS_BLOCK", "0") == "1"
        self.visualize_detections_enabled = os.environ.get("E2E_VIS_DETECTIONS", "0") == "1"
        print(
            "[perception-config] "
            f"pedestrian_conf={self.perception_confidence_thresholds[KITTI_LABEL_PEDESTRIAN]:.2f} "
            f"cyclist_conf={self.perception_confidence_thresholds[KITTI_LABEL_CYCLIST]:.2f} "
            f"vehicle_conf={self.perception_confidence_thresholds[KITTI_LABEL_VEHICLE]:.2f} "
            f"track_confirm_hits={self.stable_obstacle_tracker.confirmation_hits} "
            f"track_coast={self.stable_obstacle_tracker.coast_time:.2f}s "
            f"track_max_age={self.stable_obstacle_tracker.max_age:.2f}s"
        )
        self.infer_count = 0
        self.ros_map_alignment_initialized = False
        self.ros_map_alignment_enabled = False
        self.ros_map_offset_x = 0.0
        self.ros_map_offset_y = 0.0
        self.ros_map_alignment_mode = "none"

    def restore_compressed_sector(self, points, 
                                compressed_start, 
                                compressed_end,
                                original_start=0,
                                original_end=180,
                                angle_tol=1e-6,
                                center_attenuation=1):
        """
        还原被压缩的扇形区域，并添加中点距离衰减效果
        
        参数:
            points: 输入点云 (Nx4 numpy数组) [x,y,z,intensity,...]
            compressed_start: 压缩后的扇形起始角度(度)
            compressed_end: 压缩后的扇形结束角度(度)
            original_start: 原始扇形起始角度(度)
            original_end: 原始扇形结束角度(度)
            angle_tol: 角度比较容差
            center_attenuation: 中点角度的距离衰减系数(0.8表示压缩到80%)
        
        返回:
            还原后的点云（带中点距离衰减）
        """
        # 转换为极坐标（使用0-360度表示）
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) * 180 / np.pi  # [-180, 180]
        theta = np.where(theta < 0, theta + 360, theta)  # [0, 360]
        
        # 计算原始扇形中点角度（处理360°环绕）
        original_center = (original_start + original_end) / 2 % 360
        original_half_range = abs(original_end - original_start) / 2
        
        # 计算压缩比例
        compressed_range = (compressed_end - compressed_start) % 360
        original_range = (original_end - original_start) % 360
        compression_ratio = original_range / (compressed_range + angle_tol)
        
        # 确定目标扇形区域内的点
        if compressed_start < compressed_end:
            in_sector = (theta >= compressed_start - angle_tol) & (theta <= compressed_end + angle_tol)
        else:  # 跨越0度的情况
            in_sector = (theta >= compressed_start - angle_tol) | (theta <= compressed_end + angle_tol)
        
        # 初始化结果
        theta_restored = theta.copy()
        r_adjusted = r.copy()
        
        # 对目标区域内的点进行处理
        if compressed_range > 0:
            # 1. 角度还原
            normalized = ((theta[in_sector] - compressed_start) % 360) / compressed_range
            theta_restored[in_sector] = (original_start + normalized * original_range) % 360
            
            # 2. 计算衰减系数（中点=center_attenuation，边界=1.0）
            # 计算当前角度与中点的角度差（考虑360°环绕）
            angle_diff = np.abs((theta_restored[in_sector] - original_center + 180) % 360 - 180)
            
            # 归一化到[0,1]：0表示中点，1表示边界
            normalized_diff = np.clip(angle_diff / original_half_range, 0, 1)
            
            # 线性衰减：中点=center_attenuation，边界=1.0
            attenuation_factor = center_attenuation + (1 - center_attenuation) * normalized_diff
            
            # 应用距离衰减
            r_adjusted[in_sector] = r[in_sector] * attenuation_factor
        
        # 处理角度闭合
        theta_restored = theta_restored % 360
        
        # 转换回笛卡尔坐标
        theta_rad = np.deg2rad(theta_restored)
        x_new = r_adjusted * np.cos(theta_rad)
        y_new = r_adjusted * np.sin(theta_rad)
        
        # 保留其他特征（如强度等）
        if points.shape[1] > 3:
            return np.column_stack((x_new, y_new, z, points[:, 3:]))
        else:
            return np.column_stack((x_new, y_new, z))
    
    def visualize_raw_pointcloud(self, points):
        if self.debug_visualize_block:
            print(
                "[sync-debug] visualize enter "
                f"wall_time={time.time():.3f} "
                f"infer_count={self.infer_count} "
                f"ego=({getattr(self.ego, 'x', None)}, {getattr(self.ego, 'y', None)}) "
                f"speed={getattr(self.ego, 'speed', None)}"
            )
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Raw PointCloud")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

        vis.add_geometry(pcd)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(coord_frame)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 2

        vis.run()
        vis.destroy_window()
        if self.debug_visualize_block:
            print(
                "[sync-debug] visualize exit "
                f"wall_time={time.time():.3f} "
                f"infer_count={self.infer_count} "
                f"ego=({getattr(self.ego, 'x', None)}, {getattr(self.ego, 'y', None)}) "
                f"speed={getattr(self.ego, 'speed', None)}"
            )



    def process_pointcloud_msg(self, pointclouds, big, map_type, map_file): #处理点云数据输出障碍物信息
        # print(pointclouds.shape)
        for pointcloud in pointclouds: # pointclouds(点云数据)
            self.latest_obstacle_measurement_stamp_s = self._pointcloud_timestamp_seconds(
                pointcloud
            )

            try:
                # print(self.weather)
                time1 = time.time()
                points = pointcloud.points  # [N,4] 格式为[x,y,z,intensity]
                # raw_points = points.copy()
                # self.visualize_raw_pointcloud(raw_points)
                # 定义旋转函数
                def rotate_points(points, angle_deg, origin=(0, 0)):
                    """顺时针旋转点云（角度制）"""
                    angle_rad = np.radians(angle_deg)
                    ox, oy = origin
                    rotated_points = points.copy()
                    # 计算旋转后的x,y
                    rotated_points[:, 0] = ox + np.cos(angle_rad) * (points[:, 0] - ox) + np.sin(angle_rad) * (points[:, 1] - oy)
                    rotated_points[:, 1] = oy - np.sin(angle_rad) * (points[:, 0] - ox) + np.cos(angle_rad) * (points[:, 1] - oy)
                    return rotated_points

                if (self.weather['rain'] >= 0.15 or self.weather['fog'] >= 0.15 or self.weather['snow'] >= 0.15) or (big and (self.weather['rain'] > 0.01 or self.weather['fog'] > 0.01 or self.weather['snow'] > 0.01)):
                    if self.weather['fog'] < 0.1 and (self.weather['rain']<0.3 and self.weather['snow'] < 0.3):
                        r = 5
                    elif self.weather['fog'] < 0.2 and (self.weather['rain']<0.7 and self.weather['snow'] < 0.6):
                        r = 15
                    elif self.weather['fog'] < 0.3 and (self.weather['rain']<0.7 and self.weather['snow'] < 0.6):
                        r = 25
                    elif self.weather['fog'] < 0.4:
                        r = 30
                    else:
                        r = 30
                    points = rotate_points(points, r)

                # 1. 分离前后向点云
                # 前向点云 (x > 0)
                ego_vehicle_mask = (points[:, 0] > -1.5) & (points[:, 0] < 1.6) & \
                   (points[:, 1] > -0.6) & (points[:, 1] < 0.6)

                forward_mask = (points[:, 0] > 0) & (points[:, 0] < 60) & \
                            (points[:, 1] > -50) & (points[:, 1] < 50) & \
                            (points[:, 2] > -10.7)  & (~ego_vehicle_mask)
                forward_points = points[forward_mask]
                
                # 后向点云 (x < 0)
                backward_mask = (points[:, 0] > -60) & (points[:, 0] < 0) & \
                            (points[:, 1] > -50) & (points[:, 1] < 50) & \
                            (points[:, 2] > -10.7)  & (~ego_vehicle_mask)
                backward_points = points[backward_mask].copy()
                car_conf = 0.65
                if (self.weather['rain'] >= 0.15 or self.weather['fog'] >= 0.15 or self.weather['snow'] >= 0.15) or (big and (self.weather['rain'] > 0.01 or self.weather['fog'] > 0.01 or self.weather['snow'] > 0.01)):
                    
                    if self.weather['fog'] < 0.1 and (self.weather['rain']<0.3 and self.weather['snow'] < 0.3):
                        f = 3
                        b = 36
                        a1 = 60
                        a2 = 150
                        a3 = 27
                        b1 = 273
                        b2 = 315
                        b3 = 236
                    elif self.weather['fog'] < 0.2 and (self.weather['rain']<0.7 and self.weather['snow'] < 0.6):
                        f = 7
                        b = 46
                        a1 = 67
                        a2 = 150
                        a3 = 23
                        b1 = 277
                        b2 = 315
                        b3 = 237
                    elif self.weather['fog'] < 0.3 and (self.weather['rain']<0.7 and self.weather['snow'] < 0.6):
                        f = 10
                        b = 61
                        a1 = 88
                        a2 = 150
                        a3 = 18
                        b1 = 283
                        b2 = 315
                        b3 = 230
                    elif self.weather['fog'] < 0.4:
                        f = 10
                        b = 78
                        a1 = 83
                        a2 = 150
                        a3 = 10
                        b1 = 290
                        b2 = 315
                        b3 = 223
                    else:
                        f = 12
                        b = 95
                        # print('badbadbadbadweather')
                        a1 = 83
                        a2 = 150
                        a3 = 10
                        b1 = 290
                        b2 = 315
                        b3 = 223
                        
                    f -= r
                    b -= r
                    
                    forward_points = rotate_points(forward_points, f)
                    backward_points = rotate_points(backward_points, b)
                    
                
                combined_points = np.concatenate([forward_points, backward_points])
                
                
                if (self.weather['rain'] >= 0.15 or self.weather['fog'] >= 0.15 or self.weather['snow'] >= 0.15) or (big and (self.weather['rain'] > 0.01 or self.weather['fog'] > 0.01 or self.weather['snow'] > 0.01)):
                    point2 = self.restore_compressed_sector(combined_points, a1, a2, a3, a2)
                    combined_points_jiuzheng = self.restore_compressed_sector(point2, b1, b2, b3, b2)
                    forward_points = combined_points_jiuzheng[combined_points_jiuzheng[:, 0] > 0]
                    backward_points = combined_points_jiuzheng[combined_points_jiuzheng[:, 0] < 0]

                # Keep detector inputs and visualization data in explicit
                # coordinate frames. The backward detector sees an x-mirrored
                # copy, while visualization remains in the original vehicle
                # frame so restored backward boxes line up with their points.
                forward_detector_points = forward_points.copy()
                backward_visualization_points = backward_points.copy()
                backward_detector_points = backward_points.copy()
                backward_detector_points[:, 0] *= -1
                visualization_points = np.concatenate(
                    [forward_detector_points, backward_visualization_points],
                    axis=0,
                )
                visualization_point_sources = np.concatenate(
                    [
                        np.zeros(len(forward_detector_points), dtype=np.uint8),
                        np.ones(len(backward_visualization_points), dtype=np.uint8),
                    ]
                )
                if self.perception_web_visualizer is not None:
                    self.perception_web_visualizer.publish_points(
                        visualization_points
                    )
                


                # 2. 分别进行检测；空点云不能送入 mmdet3d，否则 voxel coors 会越界。
                def empty_detection():
                    return (
                        np.empty((0, 7), dtype=float),
                        np.empty((0,), dtype=int),
                        np.empty((0,), dtype=float),
                    )

                def run_detector(points):
                    if points is None or len(points) == 0:
                        return empty_detection()
                    with torch.inference_mode():
                        result, _ = inference_detector(self.model, points)
                    return (
                        result._pred_instances_3d.bboxes_3d.tensor.cpu().numpy(),
                        result._pred_instances_3d.labels_3d.cpu().numpy(),
                        result._pred_instances_3d.scores_3d.cpu().numpy(),
                    )

                forward_box3ds, forward_labels, forward_scores = run_detector(
                    forward_detector_points
                )
                backward_box3ds, backward_labels, backward_scores = run_detector(
                    backward_detector_points
                )
                backward_box3ds[:, 0] *= -1  # 还原x坐标
                backward_box3ds[:, 6] = np.pi - backward_box3ds[:, 6]
                backward_box3ds[:, 6] = (
                    backward_box3ds[:, 6] + np.pi
                ) % (2 * np.pi) - np.pi

                if self.debug_hot_path:
                    print(
                        "[det-frame]",
                        f"forward_points={len(forward_detector_points)}",
                        f"backward_points={len(backward_detector_points)}",
                        f"forward_boxes={len(forward_box3ds)}",
                        f"backward_boxes={len(backward_box3ds)}",
                    )

                # 合并结果
                box3ds = np.concatenate([forward_box3ds, backward_box3ds])
                labels = np.concatenate([forward_labels, backward_labels])
                scores_3d = np.concatenate([forward_scores, backward_scores])
                box_sources = np.concatenate(
                    [
                        np.zeros(len(forward_box3ds), dtype=np.uint8),
                        np.ones(len(backward_box3ds), dtype=np.uint8),
                    ]
                )
        
            except Exception as exc:
                print(f'CUDA-error: {exc}')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return[]
            
            # 4. 筛选高置信度检测

            try:
                raw_int_labels = labels.astype(int)
                raw_scores_3d = scores_3d.copy()
                int_labels = labels.astype(int)
                confidence_thresholds = dict(
                    self.perception_confidence_thresholds
                )
                if map_file == 'tongji.xodr':
                    confidence_thresholds[KITTI_LABEL_VEHICLE] = min(
                        confidence_thresholds[KITTI_LABEL_VEHICLE],
                        0.30,
                    )
                confidence_mask = np.array([
                    scores_3d[i] >= confidence_thresholds.get(int_labels[i], 0.3)
                    for i in range(len(scores_3d))
                ], dtype=bool)
                
                # 应用过滤
                box3ds = box3ds[confidence_mask]
                labels = labels[confidence_mask]
                scores_3d = scores_3d[confidence_mask]
                box_sources = box_sources[confidence_mask]

                
                
            except Exception as exc:
                print(f"confidence-filter-error: {exc}")
                confidence_thresholds = dict(
                    self.perception_confidence_thresholds
                )
                raw_int_labels = labels.astype(int)
                raw_scores_3d = scores_3d.copy()
                fallback_threshold = min(confidence_thresholds.values())
                confidence_mask = scores_3d >= fallback_threshold
                box3ds = box3ds[confidence_mask]
                labels = labels[confidence_mask]
                scores_3d = scores_3d[confidence_mask]
                box_sources = box_sources[confidence_mask]

            
            

            if self.visualize_detections_enabled:
                self.visualize_detections(
                    visualization_points,
                    box3ds,
                    labels[scores_3d > 0],
                    scores_3d[scores_3d > 0],
                    point_sources=visualization_point_sources,
                    box_sources=box_sources[scores_3d > 0],
                )

            # Stable IDs/velocities are assigned later by
            # ``_associate_rule_obstacles``.  Keeping this detector stage
            # stateless keeps perception independent of the old tracker.
            if PERCEPTION_THREE_CLASS:
                valid_mask = np.isin(
                    labels,
                    [KITTI_LABEL_PEDESTRIAN, KITTI_LABEL_CYCLIST, KITTI_LABEL_VEHICLE],
                )
            elif map_type == "AI_town":
                valid_mask = (labels == 0) | (labels == 2)
            else:
                valid_mask = labels == 2
            box3ds = box3ds[valid_mask]
            labels = labels[valid_mask]
            scores_3d = scores_3d[valid_mask]
            box_sources = box_sources[valid_mask]

            if self.debug_obs or self.debug_hot_path:
                debug_now = time.time()
                if debug_now - self.last_detection_debug_wall_time >= 0.5:
                    self.last_detection_debug_wall_time = debug_now
                    class_names = {
                        KITTI_LABEL_PEDESTRIAN: "ped",
                        KITTI_LABEL_CYCLIST: "cyclist",
                        KITTI_LABEL_VEHICLE: "vehicle",
                    }
                    fields = []
                    kept_int_labels = labels.astype(int)
                    for detector_label in (
                        KITTI_LABEL_PEDESTRIAN,
                        KITTI_LABEL_CYCLIST,
                        KITTI_LABEL_VEHICLE,
                    ):
                        raw_mask = raw_int_labels == detector_label
                        kept_mask = kept_int_labels == detector_label
                        maximum = (
                            float(np.max(raw_scores_3d[raw_mask]))
                            if np.any(raw_mask)
                            else 0.0
                        )
                        fields.append(
                            f"{class_names[detector_label]}="
                            f"{int(np.sum(kept_mask))}/{int(np.sum(raw_mask))}"
                            f"(max={maximum:.2f},"
                            f"thr={confidence_thresholds[detector_label]:.2f})"
                        )
                    print(
                        "[det-stats] "
                        f"points={len(visualization_points)} "
                        + " ".join(fields)
                    )

            if self.perception_web_visualizer is not None:
                self.perception_web_visualizer.publish(
                    visualization_points,
                    box3ds,
                    labels,
                    scores_3d,
                )

            class SimpleBBox(object):
                pass

            bboxes = []
            for index in range(len(box3ds)):
                bbox = SimpleBBox()
                bbox.node_id = index
                bbox.x = float(box3ds[index, 0])
                bbox.y = float(box3ds[index, 1])
                bbox.z = float(box3ds[index, 2])
                bbox.length = float(box3ds[index, 3])
                bbox.width = float(box3ds[index, 4])
                bbox.heading = float(box3ds[index, 6])
                bbox.type = int(labels[index])
                bbox.score = float(scores_3d[index])
                bbox.vx = 0.0
                bbox.vy = 0.0
                bbox.vz = 0.0
                bboxes.append(bbox)
            return self.process_pubrole(bboxes, big)
        
 
        
    def visualize_detections(self, filtered_points, box3ds, labels, scores,
                        point_sources=None, box_sources=None,
                        show_labels=True, save_path=None, 
                        point_size=2, bg_color=(0.1,0.1,0.1)):
        """
        兼容所有Open3D版本的可视化方案
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 设置背景
        opt = vis.get_render_option()
        opt.background_color = np.array(bg_color)
        opt.point_size = point_size

        # 1. 添加点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points[:, :3])
        if point_sources is not None and len(point_sources) == len(filtered_points):
            point_colors = np.empty((len(filtered_points), 3), dtype=float)
            point_colors[np.asarray(point_sources) == 0] = [0.65, 0.65, 0.65]
            point_colors[np.asarray(point_sources) == 1] = [0.85, 0.65, 0.20]
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(pcd)

        # 2. 颜色映射
        color_map = {
            0: [1, 0, 0],  # 车-红
            1: [0, 1, 0],  # 人-绿
            2: [0, 0, 1]   # 自行车-蓝
        }

        # 3. 添加检测框
        for i in range(len(box3ds)):
            box = box3ds[i]
            label = int(labels[i])
            score = scores[i]

            # 创建3D框
            center = box[:3]
            dimensions = box[3:6]
            yaw = box[6]

            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = center
            bbox.extent = dimensions
            bbox.R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
            if box_sources is not None and i < len(box_sources):
                bbox.color = [0.1, 1.0, 0.1] if int(box_sources[i]) == 0 else [0.1, 0.4, 1.0]
            else:
                bbox.color = color_map.get(label, [1, 1, 0])
            vis.add_geometry(bbox)

            source_name = "unknown"
            if box_sources is not None and i < len(box_sources):
                source_name = "forward" if int(box_sources[i]) == 0 else "backward"
            print(
                "[det-box]",
                f"source={source_name}",
                f"xyz=({box[0]:.2f},{box[1]:.2f},{box[2]:.2f})",
                f"lwh=({box[3]:.2f},{box[4]:.2f},{box[5]:.2f})",
                f"yaw={box[6]:.3f}",
                f"label={label}",
                f"score={score:.3f}",
            )

            # 4. 文本标签解决方案
            if show_labels:
                # 方案1：使用LineSet创建简易标签
                text_pos = center + [0, 0, dimensions[2]/2 + 0.3]
                points = [center, text_pos]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines))
                line_set.colors = o3d.utility.Vector3dVector([bbox.color])
                vis.add_geometry(line_set)
                
                # 在终端打印标签信息（作为视觉参考）
                print(f"Object {i}: Pos={center}, Label={label}, Score={score:.2f}, Size(LxWxH)=({dimensions[0]:.2f}x{dimensions[1]:.2f}x{dimensions[2]:.2f}")

        # 5. 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)

        # 6. 运行可视化
        vis.run()
        vis.destroy_window()

    def process_pubrole(self, bboxes, big):
        obstacles = []
        for bbox in bboxes:
            ob = Box2d()
            ob.id = str(bbox.node_id)
            xx, yy, zz = transform([bbox.x, bbox.y, bbox.z])

            theta = -self.ego.theta 
            xx_new = xx*math.cos(theta) + yy*math.sin(theta)
            yy_new = -xx*math.sin(theta) + yy*math.cos(theta)
            ob.x = xx_new + self.ego.x 
            ob.y = yy_new + self.ego.y 
            ob.length = bbox.length
            ob.width = bbox.width

            world_theta = bbox.heading + self.ego.theta
            world_theta = (world_theta + np.pi) % (2 * np.pi) - np.pi  # 归一化
            ob.theta = world_theta

            ob.theta = world_theta

            if PERCEPTION_THREE_CLASS:
                obs_type = detector_label_to_obstacle_type(getattr(bbox, "type", KITTI_LABEL_VEHICLE))
            else:
                obs_type = OBS_TYPE_VEHICLE
            try:
                ob.obs_type = obs_type
            except Exception:
                pass
            ob.roleType = obstacle_type_to_role_type(obs_type)
            ob.detector_label = int(
                getattr(bbox, "type", KITTI_LABEL_VEHICLE)
            )
            ob.score = float(getattr(bbox, "score", 0.0))
                

            vx = bbox.vx
            vy = bbox.vy
            vz = bbox.vz
            ob.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if ob.speed < 0.01:
                ob.is_static = True
            else:
                ob.is_static = False
            ob.is_virtual = False
            obstacles.append(ob)
        return obstacles

    def change_map(self, new_map):
        self.map_name = str(new_map or "")
        if hasattr(self, "ros_bridge") and self.ros_bridge is not None:
            self.ros_bridge.map_name = self.map_name

        self.ros_map_alignment_initialized = False
        self.ros_map_alignment_enabled = False
        self.ros_map_offset_x = 0.0
        self.ros_map_offset_y = 0.0
        self.ros_map_alignment_mode = "none"
        self._reset_frenet_reward_state()
        self.reset_perception_state()

        if new_map == self.default_map:
            return
        new_map_path = os.path.join(os.path.dirname(__file__), "maps", new_map)
        if not os.path.exists(new_map_path):
            raise FileNotFoundError(new_map_path)
        self.default_map = new_map

    def close(self):
        visualizer = getattr(self, "perception_web_visualizer", None)
        if visualizer is not None:
            visualizer.stop()
            self.perception_web_visualizer = None



    def set_destination(self, x, y, theta):
        if hasattr(self, "ros_bridge") and self.ros_bridge is not None:
            self.ros_bridge.clear_global_path()
        self._reset_frenet_reward_state()
        self.reset_perception_state()
        # Destination geometry comes from /global_plan in the pure-Python path.
        del x, y, theta

    def _ins_float(self, ins, object_name, attr_name):
        obj = getattr(ins, object_name, None)
        if obj is None:
            return None
        try:
            return float(getattr(obj, attr_name))
        except Exception:
            return None

    def _valid_vehicle_size_value(self, value, minimum):
        return (
            value is not None
            and math.isfinite(value)
            and minimum <= value <= 30.0
        )

    def _vehicle_size_or_default(self, length, width):
        default_length = float(
            os.environ.get("E2E_EGO_LENGTH_M", "4.60")
        )
        default_width = float(
            os.environ.get("E2E_EGO_WIDTH_M", "1.90")
        )
        if self._valid_vehicle_size_value(length, 2.50):
            ego_length = float(length)
        elif (
            self.last_ego is not None
            and self._valid_vehicle_size_value(
                getattr(self.last_ego, "length", None), 2.50
            )
        ):
            ego_length = float(self.last_ego.length)
        else:
            ego_length = default_length

        if self._valid_vehicle_size_value(width, 1.20):
            ego_width = float(width)
        elif (
            self.last_ego is not None
            and self._valid_vehicle_size_value(
                getattr(self.last_ego, "width", None), 1.20
            )
        ):
            ego_width = float(self.last_ego.width)
        else:
            ego_width = default_width

        return ego_length, ego_width

    def _is_aitown_map(self):
        map_name = str(getattr(self, "map_name", "") or "")
        map_name = map_name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
        if map_name and not map_name.endswith(".xodr"):
            map_name += ".xodr"
        return map_name == "aitownreconstructed_v0103_200518.xodr"

    def _ins_sample_status(self, ins):
        x = self._ins_float(ins, "position", "x")
        y = self._ins_float(ins, "position", "y")
        length = self._ins_float(ins, "veh_size", "x")
        width = self._ins_float(ins, "veh_size", "y")

        if x is None or y is None:
            return False, "missing position", x, y, length, width
        if not math.isfinite(x) or not math.isfinite(y):
            return False, "non-finite position", x, y, length, width

        # 只有非 AITown 地图才按局部坐标范围过滤
        if not self._is_aitown_map():
            if abs(x) > 1000.0 or abs(y) > 1000.0:
                return False, "position out of range", x, y, length, width
        else:
            # AITown 地图使用 UTM 大坐标
            if abs(x) > 10000000.0 or abs(y) > 10000000.0:
                return False, "position out of range", x, y, length, width

        return True, "", x, y, length, width

    def _is_near_origin_ego_jump(self, x, y):
        if self.last_ego is None:
            return False

        try:
            last_x = float(self.last_ego.x)
            last_y = float(self.last_ego.y)
            x = float(x)
            y = float(y)
        except Exception:
            return False

        if not all(math.isfinite(value) for value in (last_x, last_y, x, y)):
            return False

        current_origin_dist = math.hypot(x, y)
        last_origin_dist = math.hypot(last_x, last_y)
        jump_dist = math.hypot(x - last_x, y - last_y)
        return (
            current_origin_dist <= self.near_origin_ego_radius
            and last_origin_dist >= self.near_origin_ego_jump_min
            and jump_dist >= self.near_origin_ego_jump_min
        )

    def _warn_near_origin_ego_rejected(self, ins, x, y):
        now = time.time()
        if now - self.last_near_origin_ego_reject_warn_wall_time < 1.0:
            return
        self.last_near_origin_ego_reject_warn_wall_time = now
        sequence = getattr(ins, "sequence_num", "unknown")
        last_x = getattr(self.last_ego, "x", None)
        last_y = getattr(self.last_ego, "y", None)
        jump_dist = None
        try:
            jump_dist = math.hypot(float(x) - float(last_x), float(y) - float(last_y))
        except Exception:
            pass
        print(
            "[ego][WARN] near-origin INS jump ignored "
            f"seq={sequence} count={self.near_origin_ego_reject_count} "
            f"pos=({x}, {y}) last=({last_x}, {last_y}) jump={jump_dist}"
        )

    def _normalize_frame_id(self,frame_id):
        frame_id = str(frame_id or "").strip().replace("\\", "/")
        frame_id = frame_id.rsplit("/", 1)[-1]
        if frame_id and not frame_id.lower().endswith(".xodr"):
            frame_id = f"{frame_id}.xodr"
        return frame_id.lower()


    def _is_aitown_map(self, map_id=None):
        if map_id is None:
            map_id = getattr(self, "map_name", "")
        return self._normalize_frame_id(map_id) == "aitownreconstructed_v0103_200518.xodr"

    def _warn_invalid_ego_sample(self, ins, reason, x, y, length, width):
        now = time.time()
        if now - self.last_invalid_ego_sample_warn_wall_time < 1.0:
            return
        self.last_invalid_ego_sample_warn_wall_time = now
        sequence = getattr(ins, "sequence_num", "unknown")
        print(
            "[ego][WARN] invalid INS ignored "
            f"seq={sequence} count={self.invalid_ego_sample_count} "
            f"reason={reason} x={x} y={y} length={length} width={width}"
        )

    def update_ego(self, ins):
        # print(ins)
        # print(ins.veh_size.x, ins.veh_size.y)
        valid_sample, reason, raw_x, raw_y, raw_length, raw_width = self._ins_sample_status(ins)
        if not valid_sample:
            self.invalid_ego_sample_count += 1
            self._warn_invalid_ego_sample(ins, reason, raw_x, raw_y, raw_length, raw_width)
            self.ego = self.last_ego
            return False

        if self._is_near_origin_ego_jump(raw_x, raw_y):
            self.near_origin_ego_reject_count += 1
            self._warn_near_origin_ego_rejected(ins, raw_x, raw_y)
            self.ego = self.last_ego
            return False

        self.invalid_ego_sample_count = 0
        self.near_origin_ego_reject_count = 0
        self.ego = Box2d()
        self.ego.id = "testee1"
        ego_length, ego_width = self._vehicle_size_or_default(raw_length, raw_width)
        self.ego.width = ego_width
        self.ego.length = ego_length
        self.ego.x = ins.position.x
        self.ego.y = ins.position.y

        self.ego.theta = ins.heading
        current_yaw = float(self.ego.theta)
        current_yaw_wall_time = time.monotonic()
        derived_yawrate = None
        yaw_dt = None
        if (
            self.previous_ego_yaw is not None
            and self.previous_ego_yaw_wall_time is not None
        ):
            yaw_dt = current_yaw_wall_time - self.previous_ego_yaw_wall_time
            if yaw_dt > 1e-6:
                yaw_delta = math.atan2(
                    math.sin(current_yaw - self.previous_ego_yaw),
                    math.cos(current_yaw - self.previous_ego_yaw),
                )
                derived_yawrate = yaw_delta / yaw_dt

        raw_yawrate = float(
            getattr(getattr(ins, "angular_velocity", None), "z", 0.0)
        )
        if self.debug_hot_path:
            print(
                "[ego-yaw]",
                f"seq={getattr(ins, 'sequence_num', None)}",
                f"yaw_deg={math.degrees(current_yaw):.3f}",
                f"raw_yawrate={raw_yawrate:.3f}",
                f"derived_yawrate={derived_yawrate:.3f}"
                if derived_yawrate is not None
                else "derived_yawrate=n/a",
                f"dt={yaw_dt:.4f}s" if yaw_dt is not None else "dt=n/a",
            )
        self.previous_ego_yaw = current_yaw
        self.previous_ego_yaw_wall_time = current_yaw_wall_time

        vx = ins.linear_velocity.x
        vy = ins.linear_velocity.y
        vz = ins.linear_velocity.z
        # INS velocity is expressed in the map frame, while /global_info is
        # consumed as vehicle longitudinal/lateral velocity. Rotate it by the
        # current yaw before publishing. Do not include vertical velocity in
        # road speed.
        self.ego.speed = math.hypot(vx, vy)
        cos_yaw = math.cos(self.ego.theta)
        sin_yaw = math.sin(self.ego.theta)
        ego_forward_speed = vx * cos_yaw + vy * sin_yaw
        ego_lateral_speed = -vx * sin_yaw + vy * cos_yaw
        self.ego_ros_speed = max(0.0, ego_forward_speed)
        self.ego_ros_vy = ego_lateral_speed

        ax = ins.linear_acceleration.x
        ay = ins.linear_acceleration.y
        az = ins.linear_acceleration.z
        self.ego.acc = math.sqrt(ax * ax + ay * ay + az * az)
        self.ego_ros_acc = self.ego.acc
        self.ego_ros_ax = ax
        self.ego_ros_ay = ay
        # Keep publishing the raw INS yaw rate for now. The wrapped finite-
        # difference value above is diagnostic only until it is validated.
        self.ego_ros_yawrate = raw_yawrate
        self.ego_lon_acc = ax
        self.ego_lat_acc = ay
        self.ego_ros_latitude = getattr(getattr(ins, "position", None), "latitude", 0.0)
        self.ego_ros_longitude = getattr(getattr(ins, "position", None), "longitude", 0.0)
        self.ego_ros_heading_deg = math.degrees(self.ego.theta)
        self.ego_ros_sequence = getattr(ins, "sequence_num", None)

        self.ego.is_static = False
        self.ego.is_virtual = False

        if self.debug_hot_path:
            print(
                11111,
                "seq",
                getattr(ins, "sequence_num", None),
                self.ego.x,
                self.ego.y,
                self.ego.speed,
                self.ego.width,
                self.ego.length,
            )

        # Near-zero starts are allowed; only implausible jumps to origin are rejected above.
        self.last_ego = self.ego
        self.last_ego_update_wall_time = time.monotonic()
        self.last_ego_update_sequence = self.ego_ros_sequence
        return True

    def publish_latest_ego(self, main, ins=None):
        if self.ego is None or self.ros_bridge is None:
            return

        self._ensure_ros_map_alignment(main)
        ros_ego = self._copy_for_ros_map_frame(self.ego)
        ego_sample_id = self.last_ego_update_sequence
        published = self.ros_bridge.publish_ego_state(
            ros_ego,
            self.vehicle_feedback,
            ego_speed=self.ego_ros_speed,
            ego_vy=self.ego_ros_vy,
            ego_yawrate=self.ego_ros_yawrate,
            ego_acc=self.ego_ros_acc,
            latitude=self.ego_ros_latitude,
            longitude=self.ego_ros_longitude,
            heading_deg=self.ego_ros_heading_deg,
            ego_sample_id=ego_sample_id,
        )
        if (self.debug_hot_path and published and
                self.last_ego_update_wall_time is not None):
            publish_delay = time.monotonic() - self.last_ego_update_wall_time
            print(
                "[ego-pub]",
                f"seq={ego_sample_id}",
                f"update_to_pub={publish_delay:.3f}s",
                f"x={ros_ego.x:.3f}",
                f"y={ros_ego.y:.3f}",
                f"speed={self.ego_ros_speed:.3f}",
            )

    def update_vehicle_feedback(self, vehicle_feedback):
        steering_feedback = vehicle_feedback.steering_feedback
        if hasattr(steering_feedback, "steering_wheel_angle"):
            self.vehicle_feedback.steering_wheel_angle = (
                steering_feedback.steering_wheel_angle
            )
        elif hasattr(steering_feedback, "target_steering_wheel_angle"):
            self.vehicle_feedback.steering_wheel_angle = (
                steering_feedback.target_steering_wheel_angle
            )
        if hasattr(vehicle_feedback.brake_feedback, "brake_pedal_position"):
            self.vehicle_feedback.brake_pedal_position = (
                vehicle_feedback.brake_feedback.brake_pedal_position
            )
        if hasattr(vehicle_feedback.driving_feedback, "accelerator_pedal_position"):
            self.vehicle_feedback.accelerator_pedal_position = (
                vehicle_feedback.driving_feedback.accelerator_pedal_position
            )

    def _ensure_ros_map_alignment(self, main):
        if self.ros_map_alignment_initialized or self.ego is None:
            return

        env = getattr(main, "env", None)
        if env is None:
            self.ros_map_alignment_initialized = True
            return

        try:
            start_x = float(getattr(env, "x_start", 0.0))
            start_y = float(getattr(env, "y_start", 0.0))
            ego_x = float(self.ego.x)
            ego_y = float(self.ego.y)
        except Exception:
            self.ros_map_alignment_initialized = True
            return

        start_is_map_coord = max(abs(start_x), abs(start_y)) > 10000.0
        ego_is_map_coord = max(abs(ego_x), abs(ego_y)) > 10000.0
        self.ros_map_alignment_enabled = start_is_map_coord != ego_is_map_coord
        if self.ros_map_alignment_enabled:
            self.ros_map_offset_x = start_x - ego_x
            self.ros_map_offset_y = start_y - ego_y
            self.ros_map_alignment_mode = (
                "local_to_map" if start_is_map_coord else "map_to_local"
            )
            print(
                "[ros-map-align] enabled "
                f"mode={self.ros_map_alignment_mode} "
                f"start=({start_x:.3f}, {start_y:.3f}) "
                f"first_ego=({ego_x:.3f}, {ego_y:.3f}) "
                f"offset=({self.ros_map_offset_x:.3f}, {self.ros_map_offset_y:.3f})"
            )
        else:
            self.ros_map_offset_x = 0.0
            self.ros_map_offset_y = 0.0
            self.ros_map_alignment_mode = "none"
        self.ros_map_alignment_initialized = True

    def _copy_for_ros_map_frame(self, obj):
        if obj is None or not self.ros_map_alignment_enabled:
            return obj
        aligned = self._clone_box2d(obj)
        aligned.x = float(getattr(obj, "x", 0.0)) + self.ros_map_offset_x
        aligned.y = float(getattr(obj, "y", 0.0)) + self.ros_map_offset_y
        return aligned

    def _clone_box2d(self, obj):
        cloned = Box2d()
        for attr_name in (
            "id",
            "x",
            "y",
            "theta",
            "speed",
            "speed_valid",
            "world_vx",
            "world_vy",
            "acc",
            "length",
            "width",
            "roleType",
            "obs_type",
            "detector_label",
            "score",
            "track_hits",
            "track_misses",
            "track_age",
            "track_predicted",
            "is_static",
            "is_virtual",
        ):
            if hasattr(obj, attr_name):
                try:
                    setattr(cloned, attr_name, getattr(obj, attr_name))
                except Exception:
                    pass
        return cloned

    @staticmethod
    def _pointcloud_timestamp_seconds(pointcloud):
        """Return the source-measurement time in seconds, never wall time."""
        for raw in (
            getattr(pointcloud, "timestamp_sec", None),
            getattr(pointcloud, "lidar_timestamp", None),
        ):
            try:
                stamp = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(stamp) or stamp <= 0.0:
                continue
            # lidar_timestamp is used by more than one simulator adapter.
            # Normalize common epoch representations only.
            if stamp > 1e17:       # nanoseconds since epoch
                stamp *= 1e-9
            elif stamp > 1e14:     # microseconds since epoch
                stamp *= 1e-6
            elif stamp > 1e11:     # milliseconds since epoch
                stamp *= 1e-3
            if math.isfinite(stamp):
                return stamp
        return None

    @staticmethod
    def _set_obstacle_speed(obstacle, speed_mps, valid, world_vx=0.0, world_vy=0.0):
        """Apply the ROS obstacle-speed contract: non-negative m/s + validity."""
        speed_mps = float(speed_mps) if math.isfinite(speed_mps) else 0.0
        try:
            obstacle.speed = max(0.0, speed_mps)
            obstacle.speed_valid = bool(valid)
            obstacle.world_vx = float(world_vx) if valid else 0.0
            obstacle.world_vy = float(world_vy) if valid else 0.0
        except Exception:
            # Box2d normally permits these attributes.  Do not let a legacy
            # binding prevent a conservative zero-speed publication.
            obstacle.speed = 0.0

    @staticmethod
    def _reference_yaw_at(main, x, y):
        """Get the tangent heading of the active global/reference path."""
        env = getattr(main, "env", None)
        if env is None or not hasattr(env, "_active_reference_path"):
            return None
        try:
            path, _ = env._active_reference_path()
            projection = env._project_to_reference_path(float(x), float(y))
            if path is None or projection is None:
                return None
            yaws = path.get("yaw")
            index = int(projection.get("index", 0))
            if yaws is None or len(yaws) == 0:
                return None
            yaw = float(yaws[max(0, min(index, len(yaws) - 1))])
            return yaw if math.isfinite(yaw) else None
        except Exception:
            return None

    def _update_obstacle_longitudinal_speeds(self, obstacles, measurement_stamp, main):
        """Estimate signed road-tangent velocity from stable world-frame tracks."""
        if measurement_stamp is None:
            for obstacle in obstacles:
                self._set_obstacle_speed(obstacle, 0.0, False)
            return

        seen_ids = set()
        for obstacle in obstacles:
            track_id = str(getattr(obstacle, "id", "-1"))
            try:
                x = float(obstacle.x)
                y = float(obstacle.y)
            except (AttributeError, TypeError, ValueError):
                self._set_obstacle_speed(obstacle, 0.0, False)
                continue
            ref_yaw = self._reference_yaw_at(main, x, y)
            if track_id == "-1" or ref_yaw is None:
                self._set_obstacle_speed(obstacle, 0.0, False)
                continue

            seen_ids.add(track_id)
            previous = self.obstacle_velocity_tracks.get(track_id)
            valid = False
            v_s = 0.0
            world_vx = 0.0
            world_vy = 0.0
            if previous is not None:
                dt = measurement_stamp - previous["stamp"]
                if self.obstacle_velocity_min_dt_s <= dt <= self.obstacle_velocity_max_dt_s:
                    raw_vx = (x - previous["world_x"]) / dt
                    raw_vy = (y - previous["world_y"]) / dt
                    raw_speed = math.hypot(raw_vx, raw_vy)
                    # An association jump is not a physical velocity sample.
                    if raw_speed <= 35.0:
                        raw_v_s = raw_vx * math.cos(ref_yaw) + raw_vy * math.sin(ref_yaw)
                        if previous.get("valid", False):
                            max_delta = self.obstacle_velocity_max_accel_mps2 * dt
                            limited_v_s = max(
                                previous["v_s"] - max_delta,
                                min(previous["v_s"] + max_delta, raw_v_s),
                            )
                            v_s = previous["v_s"] + self.obstacle_velocity_filter_alpha * (
                                limited_v_s - previous["v_s"]
                            )
                        else:
                            v_s = raw_v_s
                        valid = math.isfinite(v_s)
                        if valid:
                            world_vx = v_s * math.cos(ref_yaw)
                            world_vy = v_s * math.sin(ref_yaw)

            self._set_obstacle_speed(obstacle, v_s, valid, world_vx, world_vy)
            self.obstacle_velocity_tracks[track_id] = {
                "world_x": x,
                "world_y": y,
                "stamp": measurement_stamp,
                "v_s": v_s if valid else 0.0,
                "valid": valid,
            }
            if self.debug_obs:
                print(
                    "[obs-speed]",
                    f"id={track_id}", f"valid={int(valid)}",
                    f"stamp={measurement_stamp:.3f}", f"v_s={v_s:.3f}",
                )

        stale_ids = [
            track_id for track_id, state in self.obstacle_velocity_tracks.items()
            if track_id not in seen_ids and measurement_stamp - state["stamp"] > self.obstacle_velocity_track_timeout_s
        ]
        for track_id in stale_ids:
            del self.obstacle_velocity_tracks[track_id]

    def _clone_box2d_for_ros(self, obj):
        return self._clone_box2d(obj)

    def _predict_cached_obstacles(self, age_sec):
        predicted = []
        dt = max(0.0, float(age_sec or 0.0))
        for ob in self.cached_obstacles:
            pred = self._clone_box2d(ob)
            if getattr(ob, "speed_valid", False):
                world_vx = float(getattr(ob, "world_vx", 0.0) or 0.0)
                world_vy = float(getattr(ob, "world_vy", 0.0) or 0.0)
                pred.x = float(getattr(ob, "x", 0.0) or 0.0) + world_vx * dt
                pred.y = float(getattr(ob, "y", 0.0) or 0.0) + world_vy * dt
            else:
                # A stale track with no velocity measurement must not be
                # moved using an invented ego-speed surrogate.
                pred.x = float(getattr(ob, "x", 0.0) or 0.0)
                pred.y = float(getattr(ob, "y", 0.0) or 0.0)
            predicted.append(pred)
        return predicted

    def reset_perception_state(self):
        """Clear state owned by the detector-to-structured-obstacle adapter."""
        self.rule_obstacle_tracks = {}
        self.rule_next_obstacle_id = 1
        self.stable_obstacle_tracker.reset()
        self.cached_obstacles = []
        self.cached_obstacles_wall_time = 0.0
        self.obstacle_velocity_tracks = {}
        self.latest_obstacle_measurement_stamp_s = None
        self.obs_gap = 0
        self.planning_step = 0

    @staticmethod
    def _rule_obstacle_type_key(obstacle):
        return str(getattr(obstacle, "roleType", getattr(obstacle, "obs_type", "unknown")))

    def _allocate_rule_track_id(self, used_ids):
        for _ in range(1000):
            candidate = str(self.rule_next_obstacle_id)
            self.rule_next_obstacle_id += 1
            if self.rule_next_obstacle_id > 999:
                self.rule_next_obstacle_id = 1
            if candidate not in self.rule_obstacle_tracks and candidate not in used_ids:
                return candidate
        return "-1"

    def _associate_rule_obstacles_legacy(self, obstacles, measurement_stamp):
        """Assign stable IDs and timestamp-based world velocities.

        This is intentionally independent of the former RL environment.  It
        consumes only detector boxes in the map frame and keeps a small
        constant-velocity nearest-neighbour tracker for the rule planner.
        """
        now_stamp = measurement_stamp
        if now_stamp is None or not math.isfinite(float(now_stamp)):
            now_stamp = time.monotonic()
        now_stamp = float(now_stamp)
        used_ids = set()
        associated = []
        ego_speed = max(0.0, float(getattr(self.ego, "speed", 0.0) or 0.0))
        gate = max(2.5, min(8.0, 1.5 + 0.25 * ego_speed))

        for obstacle in obstacles or []:
            try:
                x = float(obstacle.x)
                y = float(obstacle.y)
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if self.ego is not None and math.hypot(x - self.ego.x, y - self.ego.y) <= 1.8:
                continue

            type_key = self._rule_obstacle_type_key(obstacle)
            best_id = None
            best_distance = float("inf")
            for track_id, track in self.rule_obstacle_tracks.items():
                if track_id in used_ids or track.get("type") != type_key:
                    continue
                dt = now_stamp - float(track.get("stamp", now_stamp))
                if dt < 0.0 or dt > self.obstacle_velocity_track_timeout_s:
                    continue
                pred_x = track["x"] + track.get("vx", 0.0) * dt
                pred_y = track["y"] + track.get("vy", 0.0) * dt
                distance = math.hypot(x - pred_x, y - pred_y)
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None or best_distance > gate:
                best_id = self._allocate_rule_track_id(used_ids)
                previous = None
            else:
                previous = self.rule_obstacle_tracks.get(best_id)

            valid_speed = False
            vx = 0.0
            vy = 0.0
            if previous is not None:
                dt = now_stamp - float(previous["stamp"])
                if self.obstacle_velocity_min_dt_s <= dt <= self.obstacle_velocity_max_dt_s:
                    raw_vx = (x - previous["x"]) / dt
                    raw_vy = (y - previous["y"]) / dt
                    if math.hypot(raw_vx, raw_vy) <= 35.0:
                        if previous.get("valid", False):
                            old_vx = float(previous.get("vx", 0.0))
                            old_vy = float(previous.get("vy", 0.0))
                            dvx = raw_vx - old_vx
                            dvy = raw_vy - old_vy
                            delta = math.hypot(dvx, dvy)
                            max_delta = self.obstacle_velocity_max_accel_mps2 * dt
                            if delta > max_delta > 0.0:
                                scale = max_delta / delta
                                raw_vx = old_vx + dvx * scale
                                raw_vy = old_vy + dvy * scale
                            alpha = self.obstacle_velocity_filter_alpha
                            vx = old_vx + alpha * (raw_vx - old_vx)
                            vy = old_vy + alpha * (raw_vy - old_vy)
                        else:
                            vx = raw_vx
                            vy = raw_vy
                        valid_speed = math.isfinite(vx) and math.isfinite(vy)

            obstacle.id = best_id
            self._set_obstacle_speed(
                obstacle,
                math.hypot(vx, vy),
                valid_speed,
                vx,
                vy,
            )
            obstacle.is_static = not valid_speed or math.hypot(vx, vy) < 0.15
            self.rule_obstacle_tracks[best_id] = {
                "x": x,
                "y": y,
                "vx": vx if valid_speed else 0.0,
                "vy": vy if valid_speed else 0.0,
                "valid": valid_speed,
                "stamp": now_stamp,
                "type": type_key,
            }
            used_ids.add(best_id)
            associated.append(obstacle)

        stale_ids = [
            track_id
            for track_id, track in self.rule_obstacle_tracks.items()
            if track_id not in used_ids
            and now_stamp - float(track.get("stamp", now_stamp))
            > self.obstacle_velocity_track_timeout_s
        ]
        for track_id in stale_ids:
            del self.rule_obstacle_tracks[track_id]
        return associated

    def _associate_rule_obstacles(self, obstacles, measurement_stamp):
        """Return confirmed, stable map-frame obstacle tracks.

        Detection boxes are associated globally instead of in detector input
        order.  Low-confidence boxes require two hits, while confirmed tracks
        survive short detector dropouts using constant-velocity prediction.
        """
        now_stamp = measurement_stamp
        if now_stamp is None:
            now_stamp = time.monotonic()
        try:
            now_stamp = float(now_stamp)
        except (TypeError, ValueError):
            now_stamp = time.monotonic()
        if not math.isfinite(now_stamp):
            now_stamp = time.monotonic()

        detections = []
        for obstacle in obstacles or []:
            try:
                x = float(obstacle.x)
                y = float(obstacle.y)
            except (AttributeError, TypeError, ValueError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if (
                self.ego is not None
                and math.hypot(x - self.ego.x, y - self.ego.y) <= 1.8
            ):
                continue
            detections.append(
                {
                    "x": x,
                    "y": y,
                    "length": float(
                        getattr(obstacle, "length", 4.0) or 4.0
                    ),
                    "width": float(
                        getattr(obstacle, "width", 1.8) or 1.8
                    ),
                    "heading": float(
                        getattr(obstacle, "theta", 0.0) or 0.0
                    ),
                    "type": int(
                        getattr(obstacle, "obs_type", OBS_TYPE_VEHICLE)
                    ),
                    "score": float(getattr(obstacle, "score", 0.0) or 0.0),
                }
            )

        track_outputs = self.stable_obstacle_tracker.update(
            detections,
            now_stamp,
        )
        associated = []
        for track in track_outputs:
            obstacle = Box2d()
            obstacle.id = str(track["id"])
            obstacle.x = float(track["x"])
            obstacle.y = float(track["y"])
            obstacle.theta = float(track["heading"])
            obstacle.length = float(track["length"])
            obstacle.width = float(track["width"])
            obstacle.obs_type = int(track["type"])
            obstacle.roleType = obstacle_type_to_role_type(
                obstacle.obs_type
            )
            obstacle.score = float(track["score"])
            obstacle.detector_label = {
                OBS_TYPE_PEDESTRIAN: KITTI_LABEL_PEDESTRIAN,
                OBS_TYPE_CYCLIST: KITTI_LABEL_CYCLIST,
                OBS_TYPE_VEHICLE: KITTI_LABEL_VEHICLE,
            }.get(obstacle.obs_type, KITTI_LABEL_VEHICLE)
            obstacle.track_hits = int(track["hits"])
            obstacle.track_misses = int(track["misses"])
            obstacle.track_age = float(track["age"])
            obstacle.track_predicted = bool(track["predicted"])
            self._set_obstacle_speed(
                obstacle,
                float(track["speed"]),
                bool(track["speed_valid"]),
                float(track["vx"]),
                float(track["vy"]),
            )
            obstacle.is_static = (
                not bool(track["speed_valid"])
                or float(track["speed"]) < 0.25
            )
            obstacle.is_virtual = False
            associated.append(obstacle)

        if self.debug_obs or self.debug_hot_path:
            debug_now = time.time()
            if debug_now - self.last_tracking_debug_wall_time >= 0.5:
                self.last_tracking_debug_wall_time = debug_now
                stats = self.stable_obstacle_tracker.last_stats
                type_counts = {
                    OBS_TYPE_VEHICLE: 0,
                    OBS_TYPE_CYCLIST: 0,
                    OBS_TYPE_PEDESTRIAN: 0,
                }
                for obstacle in associated:
                    type_counts[obstacle.obs_type] = (
                        type_counts.get(obstacle.obs_type, 0) + 1
                    )
                print(
                    "[track-stats] "
                    f"detections={stats.get('detections', 0)} "
                    f"active={stats.get('active_tracks', 0)} "
                    f"output={stats.get('confirmed_outputs', 0)} "
                    f"matched={stats.get('matches', 0)} "
                    f"new={stats.get('new_tracks', 0)} "
                    f"coasted={stats.get('coasted_outputs', 0)} "
                    f"vehicle={type_counts[OBS_TYPE_VEHICLE]} "
                    f"cyclist={type_counts[OBS_TYPE_CYCLIST]} "
                    f"ped={type_counts[OBS_TYPE_PEDESTRIAN]}"
                )
                ego_x = float(getattr(self.ego, "x", 0.0) or 0.0)
                ego_y = float(getattr(self.ego, "y", 0.0) or 0.0)
                nearest = sorted(
                    associated,
                    key=lambda item: math.hypot(
                        float(item.x) - ego_x,
                        float(item.y) - ego_y,
                    ),
                )
                for obstacle in nearest[:12]:
                    print(
                        "[track] "
                        f"id={obstacle.id} "
                        f"type={obstacle.obs_type} "
                        f"score={obstacle.score:.2f} "
                        f"pos=({obstacle.x:.2f},{obstacle.y:.2f}) "
                        f"speed={obstacle.speed:.2f}m/s "
                        f"speed_valid={int(obstacle.speed_valid)} "
                        f"hits={obstacle.track_hits} "
                        f"misses={obstacle.track_misses} "
                        f"predicted={int(obstacle.track_predicted)}"
                    )
        return associated

    def perceive(self, pointclouds, map_name=None):
        """Convert detector point clouds into rule-planner obstacle objects.

        Missing point-cloud messages reuse the latest result for a bounded
        interval.  A fresh detector miss is handled by the stable tracker,
        which coasts confirmed tracks for its shorter configured interval.
        """
        if self.ego is None:
            return []
        map_file = str(map_name or getattr(self, "map_name", "") or "")
        map_type = "AI_town" if self._is_aitown_map(map_file) else "default"

        if not pointclouds:
            self.obs_gap += 1
            age = (
                time.time() - self.cached_obstacles_wall_time
                if self.cached_obstacles_wall_time > 0.0
                else float("inf")
            )
            if self.cached_obstacles and age <= self.obstacle_hold_max_sec:
                return self._predict_cached_obstacles(age)
            return []

        self.obs_gap = 0
        self.planning_step += 1
        raw_obstacles = self.process_pointcloud_msg(
            pointclouds,
            False,
            map_type,
            os.path.basename(map_file),
        )
        measurement_stamp = self.latest_obstacle_measurement_stamp_s
        structured = self._associate_rule_obstacles(raw_obstacles, measurement_stamp)
        if structured:
            self.cached_obstacles = [self._clone_box2d(obstacle) for obstacle in structured]
            self.cached_obstacles_wall_time = time.time()
        else:
            self.cached_obstacles = []
            self.cached_obstacles_wall_time = 0.0
        return structured

    def _rl_lateral_offset(self, main, action0):
        env = getattr(main, "env", None)
        for attr_name in (
            "rl_lateral_offset",
            "target_lateral_offset",
            "target_lane_d",
            "rl_target_d",
        ):
            value = getattr(env, attr_name, None) if env is not None else None
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

        try:
            if isinstance(action0, (list, tuple, np.ndarray)) and len(action0) >= 2:
                return float(action0[1])
        except Exception:
            pass
        return 0.0

    def _rl_target_speed(self, main, action0=None):
        try:
            if isinstance(action0, (list, tuple, np.ndarray)) and len(action0) >= 1:
                return float(action0[0])
        except Exception:
            pass
        env = getattr(main, "env", None)
        if env is None:
            return getattr(self.ego, "speed", 0.0)
        for attr_name in ("v_des", "v_max1", "target_v"):
            value = getattr(env, attr_name, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return getattr(self.ego, "speed", 0.0)

    def _reset_frenet_reward_state(self):
        self.frenet_reward_last_s = None
        self.frenet_reward_last_time = None
        self.frenet_reward_last_lon_acc = None
        self.frenet_reward_last_lat_action = None
        self.frenet_reward_info = {}
        self.action0 = [0.0, 0.0]

    def _finite_float(self, value, default=0.0):
        try:
            value = float(value)
        except Exception:
            return default
        return value if math.isfinite(value) else default

    def calculate_frenet_reward(self, main):
        env = getattr(main, "env", None)
        if env is None or self.ego is None:
            self._reset_frenet_reward_state()
            return 0.0

        projection = env._project_to_reference_path(self.ego.x, self.ego.y)
        ego_s = self._finite_float(projection.get("s", 0.0))
        path_length = max(1.0, self._finite_float(projection.get("path_length", 1.0), 1.0))
        terminal_s = path_length
        remaining_s = max(0.0, terminal_s - ego_s)
        completion_ratio = float(np.clip(ego_s / terminal_s, 0.0, 1.0))
        remaining_ratio = float(np.clip(remaining_s / terminal_s, 0.0, 1.0))

        delta_s = 0.0
        if self.frenet_reward_last_s is not None:
            delta_s = ego_s - self.frenet_reward_last_s
            if abs(delta_s) > 0.5 * terminal_s:
                delta_s = 0.0
        delta_s = float(np.clip(delta_s, -3.0, 5.0))
        road_reward = 4.0 * delta_s + 3.0 * completion_ratio - 2.0 * remaining_ratio

        speed_limit = self._finite_float(getattr(env, "scene_speed_limit", None), 0.0)
        if speed_limit <= 0.0:
            speed_limit = scene_speed_limit_for_map(getattr(env, "map_file", ""))
        speed_limit = max(0.1, self._finite_float(speed_limit, 10.0))
        ego_speed = max(0.0, self._finite_float(getattr(self.ego, "speed", 0.0)))
        speed_ratio = min(ego_speed, speed_limit) / speed_limit
        over_speed_ratio = max(0.0, ego_speed - speed_limit) / speed_limit
        speed_reward = 4.0 * speed_ratio - 4.0 * over_speed_ratio * over_speed_ratio

        now = time.time()
        dt = 0.1
        if self.frenet_reward_last_time is not None:
            dt = float(np.clip(now - self.frenet_reward_last_time, 0.02, 0.5))

        lon_acc = self._finite_float(getattr(self, "ego_lon_acc", getattr(self.ego, "acc", 0.0)))
        lon_jerk = 0.0
        if self.frenet_reward_last_lon_acc is not None:
            lon_jerk = (lon_acc - self.frenet_reward_last_lon_acc) / dt

        yaw_lat_acc = ego_speed * self._finite_float(getattr(self, "ego_ros_yawrate", 0.0))
        measured_lat_acc = self._finite_float(getattr(self, "ego_lat_acc", 0.0))
        lat_acc = yaw_lat_acc if abs(yaw_lat_acc) > abs(measured_lat_acc) else measured_lat_acc

        lat_action = self._rl_lateral_offset(main, self.action0)
        lat_action = self._finite_float(lat_action)
        lat_jump = 0.0
        if self.frenet_reward_last_lat_action is not None:
            lat_jump = lat_action - self.frenet_reward_last_lat_action

        lon_acc_cost = max(0.0, abs(lon_acc) - 0.5)
        lon_jerk_cost = max(0.0, abs(lon_jerk) - 0.5)
        lat_acc_cost = max(0.0, abs(lat_acc) - 0.5)
        lat_jump_cost = abs(lat_jump)
        comfort_reward = (
            -0.6 * lon_acc_cost
            -0.08 * lon_jerk_cost
            -0.5 * lat_acc_cost
            -0.4 * lat_jump_cost
        )

        self.frenet_reward_last_s = ego_s
        self.frenet_reward_last_time = now
        self.frenet_reward_last_lon_acc = lon_acc
        self.frenet_reward_last_lat_action = lat_action

        reward = road_reward + speed_reward + comfort_reward
        self.frenet_reward_info = {
            "reward": reward,
            "road_reward": road_reward,
            "speed_reward": speed_reward,
            "comfort_reward": comfort_reward,
            "ego_s": ego_s,
            "terminal_s": terminal_s,
            "remaining_s": remaining_s,
            "delta_s": delta_s,
            "completion_ratio": completion_ratio,
            "ego_speed": ego_speed,
            "speed_limit": speed_limit,
            "lon_acc": lon_acc,
            "lon_jerk": lon_jerk,
            "lat_acc": lat_acc,
            "lat_action": lat_action,
            "lat_jump": lat_jump,
        }
        return reward

    def _sync_global_path_to_env(self, main, spin=True):
        if spin and self.ros_bridge is not None:
            self.ros_bridge.spin_once(0.0)
        env = getattr(main, "env", None)
        if env is None or not hasattr(env, "set_global_path"):
            return
        path = self.ros_bridge.get_global_path() if self.ros_bridge is not None else None
        if path is not None:
            env.set_global_path(path)

    def calculate_reward(self, main, notify, done_out):
        start_time = time.perf_counter()
        r_goal = 0
        r_collision = 0
        r_out = 0
        r_time_out = 0
        done = 0
        collision_done = 0
        time_out_done = 0
        done_type = None

        random_number = random.random()

        if notify == 7 and time.time() - self.start_time > 5.5:
            print('pppppppppppp')
            if random_number < self.collision_xishu:
                if self.collision == 0 and self.collision_xishu >= 1.1:
                    self.collision_xishu -= 0.0006
                    print(self.collision_xishu)
                r_collision = -1000
                if self.collision == 0:
                    self.collision += 1
                collision_done = 1
            else:
                print('collision_hulue')
        if notify == 13:
            r_out = 0
        if notify == 18:
            r_goal = 1500
            done = 1
            done_type = 'goal'
        if notify == 6:
            done = 1
            time_out_done = 1
            r_time_out = -1500
            done_type = 'max_time'


        if self.start and (self.ego.x > main.env.x_max or self.ego.y > main.env.y_max or self.ego.x < main.env.x_min or self.ego.y < main.env.y_min) and self.step > 5000:

            r_out = 0
            done = 1
            done_out = True
            done_type = 'out'
        if main.step*main.last_step_time >=main.env.time_max*main.last_step_time and not self.time_out:
            r_time_out = -700 -1500*np.clip(((math.sqrt((self.ego.x - main.env.x_goal)**2 + (self.ego.y - main.env.y_goal)**2))/ main.env.dis_goal), 0.0, 1.0)
            # print((math.sqrt((self.ego.x - main.env.x_goal)**2 + (self.ego.y - main.env.y_goal)**2) / main.env.dis_goal), r_time_out)
            time_out_done = 1
            done = 1
            done_type = 'time_out'
            self.time_out = True

        r_sparse = r_goal + r_collision + r_out + r_time_out
        r_frenet = self.calculate_frenet_reward(main)
        r = (r_sparse + r_frenet) * 0.2
        return r, done, collision_done, time_out_done, done_out, done_type

    def infer(self, pointclouds, main, notify): #根据点云数据pointclouds 输出控制命令
        

        self.infer_count += 1
        if self.debug_sync:
            ego_x = None if self.ego is None else self.ego.x
            ego_y = None if self.ego is None else self.ego.y
            ego_speed = None if self.ego is None else self.ego.speed
            print(
                "[sync-debug] infer enter "
                f"wall_time={time.time():.3f} "
                f"infer_count={self.infer_count} "
                f"pointclouds={len(pointclouds)} "
                f"ego=({ego_x}, {ego_y}) "
                f"speed={ego_speed} "
                f"planning_step={self.planning_step}"
            )
        if self.start_time == 0:
            self.start_time = time.time()
        if (self.weather['rain'] >= 0.15 or self.weather['fog'] >= 0.15 or self.weather['snow'] >= 0.15):
            main.env.bad_weather = True
        else:
            main.env.bad_weather = False
        time_start = time.time()
        self.step += 1
        time1 = time.time()
        done_out = False
        if self.ego is None or self.vehicle_feedback is None:
            return None, done_out
        self._sync_global_path_to_env(main)
        ganzhi = 0
        use_cached_obstacles = False
        cache_age = 0.0
        # pointclouds = []
        if  len(pointclouds) == 0:# or self.obs_gap < 4:
            self.obs_gap += 1
            cache_age = (
                time.time() - self.cached_obstacles_wall_time
                if self.cached_obstacles_wall_time > 0.0 else float("inf")
            )
            if self.cached_obstacles and cache_age <= self.obstacle_hold_max_sec:
                obstacles = self._predict_cached_obstacles(cache_age)
                use_cached_obstacles = True
                if self.debug_obs and (self.obs_gap == 1 or self.debug_sync):
                    print(
                        "[obs-cache] pointcloud empty; reuse "
                        f"{len(obstacles)} obstacles gap={self.obs_gap} "
                        f"age={cache_age:.2f}s"
                    )
            else:
                obstacles = []
            
        else:
            self.obs_gap = 0
            obstacles = self.process_pointcloud_msg(pointclouds, main.env.big, main.env.map_type, main.env.map_file)
            self.planning_step += 1
            self.last_pointclouds = []
            ganzhi = 1
        if self.start == 0:
            return None, done_out
        # Do not modify self.ego.speed with zhenlv here.  zhenlv is an
        # environment/reward scale and must never change a physical timestamp
        # or an obstacle velocity estimate.
        measurement_stamp = self.latest_obstacle_measurement_stamp_s if pointclouds else None

        
        n = 0
        obs2 = []
        self.k += 1
        
        have_id_list = []
        last_list = []

        self.time1.append(time.time() - time1)
        if main.env.map_type == 'AI_town':
            v = 3  # AI Town 低速场景
            ddd = 3  # 匹配阈值
        else:
            v = max(17, self.ego.speed)  # 高速场景
            ddd =  max(11, self.ego.speed /2) # 匹配阈值
        
        if use_cached_obstacles:
            for i in obstacles:
                if getattr(i, "id", "-1") != "-1" and math.sqrt((self.ego.x-i.x)**2+(self.ego.y-i.y)**2) > 1.8:
                    obs2.append(i)
            obstacles = []

        for i in obstacles:
            m = 999
            good_id = None
            dis_last = 999
            have_id = False
            for j in last_list:
                lll = math.sqrt((i.x - j[0])**2 + (i.y - j[1])**2)
                if lll < dis_last:
                    dis_last = lll
            if dis_last > 1:
                last_list.append([i.x, i.y])
                for key, value in self.obstacles_id_dict.items():
                    if value != {} and key not in have_id_list and self.planning_step - value['step'] <= 3 and i.roleType == value['type']:
                        # 计算时间差（step_diff = 当前step - 记录step）
                        step_diff = self.planning_step - value['step']
                         # 计算预测位置（考虑航向角 theta）
                        # Do not project with the legacy ego-derived dt.
                        # This loop only assigns IDs; physical velocity is
                        # calculated later from source timestamps.
                        dt = 0.0
                        
                        # 预测位置 = 上一帧位置 + 速度 * 时间 * 方向向量
                        if i.roleType == RoleType.PEDESTRIAN:
                            pred_x = value['pre_x']
                            pred_y = value['pre_y']
                        else:
                            pred_x = value['pre_x']
                            pred_y = value['pre_y']
                        
                        # 计算当前障碍物与预测位置的距离
                        dis = math.sqrt((i.x - pred_x)**2 + (i.y - pred_y)**2)

                        # dis = math.sqrt((i.x - value['pre_x'])**2 + (i.y - value['pre_y'])**2)
                        if dis < m:
                            good_id = key
                            m = dis

                if m < ddd:
                    self.obstacles_id_dict[str(good_id)] = {'pre_x':i.x, 'pre_y':i.y, 'step':self.planning_step, 'speed':0, 'type':i.roleType}
                    i.id = good_id
                    have_id = True
                    have_id_list.append(good_id)
                else:
                    for key, value in self.obstacles_id_dict.items():
                        if (value == {} or abs(value['step'] - self.planning_step) > 3) and key not in have_id_list:
                            i.id = key
                            have_id = True
                            have_id_list.append(key)
                            self.obstacles_id_dict[key] = {'pre_x':i.x, 'pre_y':i.y, 'step':self.planning_step, 'speed':0, 'type':i.roleType}
                            break
                    if not have_id:
                        if self.debug_hot_path:
                            print('id_outtttttttttttttttt')
                        i.id = '-1'
            else:
                i.id = '-1'
                        
        
        for key, value in self.obstacles_id_dict.items():
            if value != {} and abs(self.planning_step - value['step']) > 3:
                self.obstacles_id_dict[key] = {}

        for i in obstacles:
            n = 0
            for j in self.pre_obstacles:
                if j.id == i.id:
                    n = 1
                    dis_other = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) 
                    i.speed = 0.0
                    if i.roleType == RoleType.PEDESTRIAN:
                        delta_x = i.x - j.x
                        delta_y = i.y - j.y

                        if delta_x == 0 and delta_y == 0:
                            i.theta = j.theta
                        else:
                            # 计算航向角
                            heading_angle = math.atan2(delta_y, delta_x)
                            i.theta = heading_angle

                    break
            
            if n == 0:
                for j in self.pre_pre_obstaclea:
                    if j.id == i.id:
                        n = 2
                        dis_other = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) 
                        i.speed = 0.0
                        if i.roleType == RoleType.PEDESTRIAN:
                            delta_x = i.x - j.x
                            delta_y = i.y - j.y

                            if delta_x == 0 and delta_y == 0:
                                i.theta = j.theta
                            else:
                                # 计算航向角
                                heading_angle = math.atan2(delta_y, delta_x)
                                i.theta = heading_angle

                        break

            if n == 0:
                for j in self.pre_pre_pre_obstaclea:
                    if j.id == i.id:
                        n = 3
                        dis_other = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) 
                        i.speed = 0.0
                        if i.roleType == RoleType.PEDESTRIAN:
                            delta_x = i.x - j.x
                            delta_y = i.y - j.y

                            if delta_x == 0 and delta_y == 0:
                                i.theta = j.theta
                            else:
                                # 计算航向角
                                heading_angle = math.atan2(delta_y, delta_x)
                                i.theta = heading_angle

                        break

            if n == 0 and i.roleType == RoleType.VEHICLE:
                i.speed = 0.0
                # i.theta = self.ego.theta
            if i.speed > 50:
                i.speed = 50
            if i.id != '-1':
                self.obstacles_id_dict[i.id]['speed'] = i.speed
            if i.roleType == RoleType.PEDESTRIAN:
                i.speed = 0.0
            if math.sqrt((self.ego.x-i.x)**2+(self.ego.y-i.y)**2) > 1.8:
                obs2.append(i)

        # All ID reassignment is complete.  Estimate physical velocity only
        # now, from the final stable ID and the source pointcloud timestamp.
        if len(pointclouds) > 0:
            self._update_obstacle_longitudinal_speeds(
                obstacles, measurement_stamp, main
            )

        if len(pointclouds) > 0:
            if obs2:
                self.cached_obstacles = [self._clone_box2d(ob) for ob in obs2]
                self.cached_obstacles_wall_time = time.time()
            else:
                if self.cached_obstacles and (self.debug_obs or self.debug_sync):
                    print("[obs-cache] clear cache; fresh pointcloud has no valid obs2")
                self.cached_obstacles = []
                self.cached_obstacles_wall_time = 0.0

       
           
        r, done, collision_done, time_out_done, done_out, done_type = self.calculate_reward(
            main, notify, done_out
        )

        time3 = time.time()
        cost = compute_safety_cost(collision_done)
        action, action0 = main.get_action(
            self.ego.x, self.ego.y, self.ego.speed, self.ego.acc,
            self.ego.theta, self.ego.length, self.ego.width, obs2,
            r, cost, done, collision_done, time_out_done, ganzhi,
        )
        rl_speed = self._rl_target_speed(main, action0)
        rl_lateral_offset = self._rl_lateral_offset(main, action0)
        self._ensure_ros_map_alignment(main)
        if self.debug_obs:
            obs2_debug = [
                (
                    getattr(ob, "id", None),
                    round(float(getattr(ob, "x", 0.0)), 2),
                    round(float(getattr(ob, "y", 0.0)), 2),
                    round(float(getattr(ob, "speed", 0.0)), 2),
                    getattr(ob, "roleType", None),
                )
                for ob in obs2
            ]
            print(
                "obs2",
                obs2_debug,
                "pointclouds",
                len(pointclouds),
                "gap",
                self.obs_gap,
                "cached",
                use_cached_obstacles,
                "cache_age",
                round(cache_age, 2),
            )
        ros_obstacles = [self._copy_for_ros_map_frame(ob) for ob in obs2]
        self.ros_bridge.publish_obstacles(ros_obstacles)
        self.ros_bridge.publish_rl_decision(rl_speed, rl_lateral_offset)
        self._sync_global_path_to_env(main)
        self.ego_pre_x = self.ego.x
        self.ego_pre_y = self.ego.y
        if len(pointclouds)>0:
            self.ego_pre2_x = self.ego.x
            self.ego_pre2_y = self.ego.y
        if len(obstacles) > 0:
            self.pre_pre_pre_obstaclea = self.pre_pre_obstaclea
            self.pre_pre_obstaclea = self.pre_obstacles
            self.pre_obstacles = obstacles
        self.action0 = action0
        self.time3.append(time.time() - time3)
        if done:
            if self.time_out and done_type != 'out' and done_type != 'termination':
                done_type = 'time_out'
            if len(self.time1) != 0 and len(self.time3) != 0 and main.step != 0 and len(self.timeall) != 0:
                print('time1', sum(self.time1)/len(self.time1), 'time3', sum(self.time3)/len(self.time3), 'timeall', sum(self.timeall)/len(self.timeall))
            main.finish(self.collision, done_type, True)
            self.obstacles_id_dict = {'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'10':{},'11':{},'12':{},'13':{},'14':{},'15':{},'16':{},'17':{},'18':{},'19':{},'20':{},'21':{},'22':{},'23':{},'24':{},'25':{},'26':{},'27':{},'28':{},'29':{},'30':{},'31':{},'32':{},'33':{},'34':{},'35':{},'36':{},'37':{},'38':{},'39':{},'40':{}}
            self.start = 0
            self.collision = 0
            self.ego_pre_x = 0
            self.ego_pre_y = 0
            self.ego_pre2_x = 0
            self.ego_pre2_y = 0
            self.step = 0
            self.obs_gap = 0
            self.last_pointclouds = []
            self.last_ego = None
            self.near_origin_ego_reject_count = 0
            self.last_near_origin_ego_reject_warn_wall_time = 0.0
            self.planning_step = 0
            self.ok = 0
            self.time1 = []
            self.time3 = []
            self.timeall = []
            self.pre_obstacles = []
            self.pre_pre_obstaclea = []
            self.pre_pre_pre_obstaclea = []
            self.cached_obstacles = []
            self.cached_obstacles_wall_time = 0.0
            self.obstacle_velocity_tracks = {}
            self.latest_obstacle_measurement_stamp_s = None
            self._reset_frenet_reward_state()
            self.ros_bridge.clear_control()
            self.ros_map_alignment_initialized = False
            self.ros_map_alignment_enabled = False
            self.ros_map_offset_x = 0.0
            self.ros_map_offset_y = 0.0
            self.ros_map_alignment_mode = "none"

            self.total_episode += 1
            self.time_out = False
            self.last_steer = 0
            if self.total_episode %30 == 0:
                torch.cuda.empty_cache()
                self.model = init_model(self.config, self.checkpoint, device=self.detector_device)
        

        
        acc = action[0]
        rot = action[1]

        if False:  # retained only for compatibility with the inactive legacy infer path
            max_laa = 6
            max_w = 0.5
            max_ha = 0.5
            max_haa = (1 - 0.01)
        

            if self.last_time is not None:
                t = time.time() - self.last_time
                l_aa = (acc - self.last_a)/t
                if l_aa >= max_laa:
                    acc = self.last_a + max_laa*t
                elif l_aa < -max_laa:
                    acc = self.last_a - max_laa*t
            

            if self.last_time is not None and self.ego.speed > 0:
                t = max(time.time() - self.last_time, 1e-6)  # 时间间隔（秒）
                delta_tan = math.tan(rot) - math.tan(self.last_rot)  # tanδ差值（无量纲）
                h_aa2 = (self.ego.speed**2 / 4.6) * (delta_tan / t)  # 横向加加速度（m/s³）
                
                if abs(h_aa2) > max_haa:
                    # 物理正确的最大tanδ变化量（无量纲）
                    max_delta_tan = (max_haa * 4.6 * t) / self.ego.speed**2
                    
                    # 计算最大允许转角变化（弧度制）
                    max_rot_gap = math.atan(math.tan(self.last_rot) + np.sign(delta_tan)*abs(max_delta_tan)) - self.last_rot
                    
                    # 应用限制（全弧度制操作）
                    if delta_tan > 0:  # 正向变化
                        rot = self.last_rot + min(abs(rot - self.last_rot), abs(max_rot_gap))
                    else:  # 负向变化
                        rot = self.last_rot - min(abs(rot - self.last_rot), abs(max_rot_gap))
                    

            if rot != 0:
                R = 4.6/(math.tan(rot))
                w = self.ego.speed * math.tan(rot) / 4.6
                h_a = self.ego.speed**2 / R
                if self.last_time is not None:
                    h_aa = (h_a - self.last_h_a)/t
                
                
                    # print('yuzhi', w, h_a, h_aa)
                    # if abs(w) > max_w:
                    #     print('wwwwwwwwww')
                    # if abs(h_a) > max_ha:
                    #     print('hhhhhhhhhhhhhhhh')
                    # if abs(h_aa) > max_haa:
                    #     print('aaaaaaaaaaaaaaaaaaaa')

            self.last_time = time.time()
            self.last_a = acc
            self.last_h_a = h_a
            self.last_rot = rot
        

        


        self.timeall.append(time.time() - time1)
        ros_control_cmd = self.ros_bridge.get_control_command(self.ego_ros_speed)

        control_cmd = ControlCommand()
        if ros_control_cmd is None:
            control_cmd.acc = 0
            control_cmd.speed = 0.0
            control_cmd.steer = 0.0
            if self.debug_hot_path:
                print("[ros-control] no command received; defaulting to zero control")
        else:
            control_cmd = ros_control_cmd

        scene_vmax = scene_speed_limit_for_map(getattr(main.env, "map_file", ""))
        control_cmd.speed = min(control_cmd.speed, scene_vmax)
        if self.ego.speed > scene_vmax + 0.3:
            control_cmd.acc = min(control_cmd.acc, -0.6)
            control_cmd.speed = 0.0

        # Keep the whole longitudinal-control chain visible without flooding
        # the console at the simulator loop rate.
        debug_now = time.time()
        if (self.debug_hot_path and
                debug_now - self.ros_control_debug_last_time >= 0.25):
            latest_ctrl = self.ros_bridge.latest_ctrl
            raw_speed = float(latest_ctrl["speed"]) if latest_ctrl is not None else float("nan")
            raw_brake = float(latest_ctrl["brake"]) if latest_ctrl is not None else float("nan")
            ctrl_age = (
                debug_now - self.ros_bridge.latest_ctrl_wall_time
                if self.ros_bridge.latest_ctrl_wall_time > 0.0
                else float("inf")
            )
            acc_source = "brake" if raw_brake > 0.0 else "speed_pid"
            print(
                "[ros-longitudinal] "
                f"raw_speed={raw_speed:.3f} raw_brake={raw_brake:.3f} "
                f"ego_ros_speed={self.ego_ros_speed:.3f} "
                f"ego_sim_speed={self.ego.speed:.3f} "
                f"acc_source={acc_source} final_acc={control_cmd.acc:.3f} "
                f"final_speed={control_cmd.speed:.3f} "
                f"ctrl_age={ctrl_age:.3f}s scene_vmax={scene_vmax:.3f}"
            )
            self.ros_control_debug_last_time = debug_now
            
        return control_cmd, done_out
