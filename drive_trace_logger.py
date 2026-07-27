"""Compact structured logging for perception, planning, and control analysis."""

import json
import math
import os
import threading
import time


SCHEMA_VERSION = 1


def _finite(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clean(value, depth=0):
    """Convert runtime values into strict, bounded JSON data."""
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): _clean(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _clean(value.tolist(), depth + 1)
        except Exception:
            pass
    numeric = _finite(value)
    return numeric if numeric is not None else str(value)


def _object_fields(obj, names):
    if obj is None:
        return {}
    result = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, bool):
            result[name] = value
        elif name in ("id", "roleType"):
            result[name] = str(value)
        else:
            result[name] = _finite(value)
    return result


class DriveTraceLogger(object):
    """Write one self-contained JSON object per analysis snapshot."""

    def __init__(
        self,
        output_dir,
        period_sec=0.10,
        max_trajectory_points=30,
        max_obstacles=64,
    ):
        output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.archive_path = os.path.join(
            output_dir,
            f"drive_trace_{timestamp}_{os.getpid()}.jsonl",
        )
        self.latest_path = os.path.join(
            output_dir,
            "latest_drive_trace.jsonl",
        )
        self._files = [
            open(self.archive_path, "w", encoding="utf-8", buffering=1),
            open(self.latest_path, "w", encoding="utf-8", buffering=1),
        ]
        self.period_sec = max(0.02, float(period_sec))
        self.max_trajectory_points = max(
            5, int(max_trajectory_points)
        )
        self.max_obstacles = max(1, int(max_obstacles))
        self._lock = threading.RLock()
        self._closed = False
        self._last_record_monotonic = float("-inf")
        self._last_behavior = None
        self._session_id = ""
        self._map_name = ""
        self._write(
            self._schema_record()
        )

    @staticmethod
    def _schema_record():
        return {
            "record_type": "schema",
            "schema_version": SCHEMA_VERSION,
            "wall_time": time.time(),
            "description": (
                "ego, tracked obstacles, planner decision, local "
                "trajectory, controller state, and sent command"
            ),
        }

    def _write(self, record):
        if self._closed:
            return
        cleaned = _clean(record)
        line = json.dumps(
            cleaned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            for file_obj in self._files:
                file_obj.write(line + "\n")

    def start_session(self, session_id, map_name, metadata=None):
        # ``latest`` always contains one case only.  The timestamped archive
        # keeps every case from the current process.
        with self._lock:
            latest_file = self._files[1]
            try:
                latest_file.close()
            except Exception:
                pass
            self._files[1] = open(
                self.latest_path,
                "w",
                encoding="utf-8",
                buffering=1,
            )
        self._session_id = str(session_id or "")
        self._map_name = str(map_name or "")
        self._last_record_monotonic = float("-inf")
        self._last_behavior = None
        self._write(self._schema_record())
        self.record_event("session_start", metadata=metadata)

    def end_session(self, reason="", metadata=None):
        self.record_event(
            "session_end",
            metadata={
                "reason": str(reason or ""),
                **(metadata or {}),
            },
        )

    def record_event(self, name, metadata=None):
        self._write(
            {
                "record_type": "event",
                "schema_version": SCHEMA_VERSION,
                "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "session_id": self._session_id,
                "map_name": self._map_name,
                "event": str(name),
                "metadata": metadata or {},
            }
        )

    def _serialize_ego(self, ego, extra):
        data = _object_fields(
            ego,
            (
                "x",
                "y",
                "theta",
                "speed",
                "length",
                "width",
                "acc",
            ),
        )
        theta = data.get("theta")
        data["heading_deg"] = (
            math.degrees(theta) if theta is not None else None
        )
        for key in (
            "ins_sequence",
            "ros_vy",
            "yaw_rate",
            "lon_acc",
            "lat_acc",
        ):
            if key in extra:
                data[key] = _clean(extra[key])
        return data

    def _serialize_obstacles(self, obstacles, ego, planner_debug):
        ego_x = _finite(getattr(ego, "x", None), 0.0)
        ego_y = _finite(getattr(ego, "y", None), 0.0)
        ego_yaw = _finite(getattr(ego, "theta", None), 0.0)
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        planner_by_id = {
            str(item.get("id")): item
            for item in (planner_debug.get("obstacles") or [])
            if isinstance(item, dict)
        }
        serialized = []
        for obstacle in obstacles or []:
            item = _object_fields(
                obstacle,
                (
                    "id",
                    "x",
                    "y",
                    "theta",
                    "speed",
                    "speed_valid",
                    "world_vx",
                    "world_vy",
                    "length",
                    "width",
                    "obs_type",
                    "score",
                    "is_static",
                    "is_virtual",
                    "track_hits",
                    "track_misses",
                    "track_age",
                    "track_predicted",
                ),
            )
            x = item.get("x")
            y = item.get("y")
            if x is None or y is None:
                continue
            dx = x - ego_x
            dy = y - ego_y
            item.update(
                {
                    "distance": math.hypot(dx, dy),
                    "ego_longitudinal": dx * cos_yaw + dy * sin_yaw,
                    "ego_lateral": -dx * sin_yaw + dy * cos_yaw,
                }
            )
            planner_item = planner_by_id.get(str(item.get("id")))
            if planner_item is not None:
                item["frenet"] = {
                    key: planner_item.get(key)
                    for key in ("s", "d", "gap")
                }
            serialized.append(item)
        serialized.sort(key=lambda item: item["distance"])
        return serialized[: self.max_obstacles]

    def _serialize_trajectory(self, plan_result):
        trajectory = getattr(plan_result, "trajectory", None)
        if trajectory is None:
            return None
        try:
            count = len(trajectory.t)
        except Exception:
            return None
        if count <= 0:
            return None
        if count <= self.max_trajectory_points:
            indices = list(range(count))
        else:
            step = (count - 1) / float(self.max_trajectory_points - 1)
            indices = sorted(
                {min(count - 1, int(round(index * step)))
                 for index in range(self.max_trajectory_points)}
            )

        def samples(name):
            values = getattr(trajectory, name, None)
            if values is None:
                return None
            try:
                return [_finite(values[index]) for index in indices]
            except Exception:
                return None

        data = {
            "original_count": count,
            "sample_count": len(indices),
        }
        for name in (
            "t",
            "s",
            "d",
            "x",
            "y",
            "yaw",
            "kappa",
            "speed",
            "accel",
            "d_speed",
            "d_accel",
        ):
            values = samples(name)
            if values is not None:
                data[name] = values
        return data

    def record_cycle(
        self,
        loop_count,
        ego,
        obstacles,
        plan_result,
        planner_debug,
        control_command,
        controller_debug,
        vehicle_feedback=None,
        extra=None,
        force=False,
    ):
        extra = dict(extra or {})
        now_monotonic = _finite(
            extra.get("monotonic_time"),
            time.monotonic(),
        )
        behavior = str(
            getattr(plan_result, "behavior", "") or ""
        )
        emergency = bool(
            getattr(plan_result, "emergency", False)
        )
        changed = behavior != self._last_behavior
        if (
            not force
            and not emergency
            and not changed
            and now_monotonic - self._last_record_monotonic
            < self.period_sec
        ):
            return False
        self._last_record_monotonic = now_monotonic
        self._last_behavior = behavior
        planner_debug = dict(planner_debug or {})
        controller_debug = dict(controller_debug or {})
        record = {
            "record_type": "cycle",
            "schema_version": SCHEMA_VERSION,
            "wall_time": _finite(extra.get("wall_time"), time.time()),
            "monotonic_time": now_monotonic,
            "session_id": self._session_id,
            "map_name": self._map_name,
            "loop_count": int(loop_count),
            "fresh_pointcloud": bool(extra.get("fresh_pointcloud", False)),
            "perception": {
                "source": str(extra.get("perception_source", "")),
                "fresh_gt": bool(extra.get("fresh_gt", False)),
                "gt_ready": bool(extra.get("gt_ready", False)),
                "gt_age": _finite(extra.get("gt_age")),
                "gt_failsafe_applied": bool(
                    extra.get("gt_failsafe_applied", False)
                ),
            },
            "replanned": bool(extra.get("replanned", False)),
            "timing": {
                key: extra.get(key)
                for key in (
                    "control_dt",
                    "lidar_get_elapsed",
                    "perception_elapsed",
                    "planning_elapsed",
                    "control_elapsed",
                )
            },
            "ego": self._serialize_ego(ego, extra),
            "obstacles": self._serialize_obstacles(
                obstacles, ego, planner_debug
            ),
            "plan": {
                "behavior": behavior,
                "target_speed": _finite(
                    getattr(plan_result, "target_speed", None)
                ),
                "emergency": emergency,
                "reason": str(
                    getattr(plan_result, "reason", "") or ""
                ),
                "debug": planner_debug,
                "trajectory": self._serialize_trajectory(plan_result),
            },
            "control": {
                "command": _object_fields(
                    control_command,
                    ("acc", "speed", "steer"),
                ),
                "debug": controller_debug,
            },
            "feedback": _object_fields(
                vehicle_feedback,
                (
                    "steering_wheel_angle",
                    "accelerator_pedal_position",
                    "brake_pedal_position",
                ),
            ),
            "perception_measurement_time": extra.get(
                "perception_measurement_time"
            ),
        }
        self._write(record)
        return True

    def close(self):
        if self._closed:
            return
        self.record_event("logger_close")
        with self._lock:
            self._closed = True
            for file_obj in self._files:
                try:
                    file_obj.flush()
                    file_obj.close()
                except Exception:
                    pass
