"""Deterministic roundabout path tracking and fixed-lead following.

This controller deliberately has no behaviour tree, obstacle prediction, or
trajectory-candidate sampling.  The launch mode owns the whole session:

* ``direct`` tracks the global path at a fixed maximum speed.
* ``follow`` locks the nearest same-lane lead once and controls a fixed
  bumper-to-bumper gap to that ID for the rest of the session.
"""

import math
import os
import time

from rule_based_planner import ControlOutput, PlanResult, ReferencePath


EPS = 1e-6


def _finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clip(value, low, high):
    return max(low, min(high, value))


def _wrap(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class RoundaboutController(object):
    """Session-exclusive global-path/direct-follow controller."""

    VALID_MODES = ("off", "follow", "direct")

    def __init__(
        self,
        mode="off",
        desired_gap=1.0,
        max_speed=30.0,
        max_accel=20.0,
        max_decel=15.5,
        lane_half_width=1.75,
        gap_gain=4.0,
        speed_gain=2.5,
        wheelbase=3.38,
        steering_ratio=1.65,
        max_steer_deg=28.0,
        steer_rate_deg_s=100.0,
    ):
        self.mode = str(mode or "off").strip().lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                "roundabout mode must be one of {}".format(
                    ", ".join(self.VALID_MODES)
                )
            )
        self.desired_gap = max(0.1, float(desired_gap))
        self.max_speed = max(0.1, float(max_speed))
        self.max_accel = max(0.1, float(max_accel))
        self.max_decel = max(0.1, float(max_decel))
        self.lane_half_width = max(0.5, float(lane_half_width))
        self.gap_gain = max(0.1, float(gap_gain))
        self.speed_gain = max(0.1, float(speed_gain))
        self.wheelbase = max(0.5, float(wheelbase))
        self.steering_ratio = max(0.1, float(steering_ratio))
        self.max_steer_deg = max(1.0, float(max_steer_deg))
        self.steer_rate_deg_s = max(
            1.0, float(steer_rate_deg_s)
        )
        self.reference = ReferencePath()
        self.map_name = ""
        self.locked_lead_id = None
        self.locked_lead_description = None
        self.last_steer = 0.0
        self.last_debug = {}
        self.last_plan = PlanResult(
            None, 0.0, "ROUNDABOUT_OFF", False, ""
        )
        self._last_status_print = 0.0

    @property
    def enabled(self):
        return self.mode != "off"

    def reset(self, map_name=""):
        self.reference = ReferencePath()
        self.map_name = str(map_name or "")
        self.locked_lead_id = None
        self.locked_lead_description = None
        self.last_steer = 0.0
        self.last_debug = {}
        self.last_plan = PlanResult(
            None, 0.0, "ROUNDABOUT_OFF", False, ""
        )

    def _ego_values(self, ego):
        return {
            "x": _finite(getattr(ego, "x", 0.0)),
            "y": _finite(getattr(ego, "y", 0.0)),
            "yaw": _finite(getattr(ego, "theta", 0.0)),
            "speed": max(
                0.0, _finite(getattr(ego, "speed", 0.0))
            ),
            "length": max(
                1.0, _finite(getattr(ego, "length", 4.6), 4.6)
            ),
            "width": max(
                0.5, _finite(getattr(ego, "width", 1.9), 1.9)
            ),
        }

    def _obstacle_values(self, obstacle):
        yaw = _finite(getattr(obstacle, "theta", 0.0))
        speed = max(
            0.0, _finite(getattr(obstacle, "speed", 0.0))
        )
        speed_valid = bool(
            getattr(obstacle, "speed_valid", False)
        )
        vx = _finite(
            getattr(obstacle, "world_vx", speed * math.cos(yaw))
        )
        vy = _finite(
            getattr(obstacle, "world_vy", speed * math.sin(yaw))
        )
        if not speed_valid and abs(vx) + abs(vy) < EPS:
            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)
        return {
            "raw": obstacle,
            "id": str(getattr(obstacle, "id", "-1")),
            "x": _finite(getattr(obstacle, "x", 0.0)),
            "y": _finite(getattr(obstacle, "y", 0.0)),
            "yaw": yaw,
            "speed": speed,
            "vx": vx,
            "vy": vy,
            "length": max(
                0.3,
                _finite(getattr(obstacle, "length", 4.5), 4.5),
            ),
            "width": max(
                0.3,
                _finite(getattr(obstacle, "width", 1.9), 1.9),
            ),
            "role": str(
                getattr(
                    obstacle,
                    "role_name",
                    getattr(obstacle, "roleType", ""),
                )
            ),
            "obs_type": int(
                _finite(getattr(obstacle, "obs_type", 1), 1)
            ),
        }

    @staticmethod
    def _path_half_extent(length, width, object_yaw, path_yaw):
        delta = _wrap(object_yaw - path_yaw)
        return 0.5 * (
            length * abs(math.cos(delta))
            + width * abs(math.sin(delta))
        )

    def _project_obstacles(self, ego_values, ego_projection, obstacles):
        result = []
        ego_extent = self._path_half_extent(
            ego_values["length"],
            ego_values["width"],
            ego_values["yaw"],
            ego_projection["yaw"],
        )
        for raw in obstacles or []:
            item = self._obstacle_values(raw)
            projection = self.reference.project(
                item["x"], item["y"]
            )
            if projection is None:
                continue
            relative_d = projection["d"] - ego_projection["d"]
            if abs(relative_d) > self.lane_half_width:
                continue
            path_extent = self._path_half_extent(
                item["length"],
                item["width"],
                item["yaw"],
                projection["yaw"],
            )
            centre_distance = projection["s"] - ego_projection["s"]
            if centre_distance <= 0.0:
                continue
            item.update(
                {
                    "s": projection["s"],
                    "d": projection["d"],
                    "relative_d": relative_d,
                    "path_yaw": projection["yaw"],
                    "longitudinal_speed": max(
                        0.0,
                        item["vx"] * math.cos(projection["yaw"])
                        + item["vy"] * math.sin(projection["yaw"]),
                    ),
                    "gap": centre_distance - ego_extent - path_extent,
                }
            )
            result.append(item)
        return result

    def _select_or_find_locked_lead(
        self, ego_values, ego_projection, obstacles
    ):
        projected = self._project_obstacles(
            ego_values, ego_projection, obstacles
        )
        if self.locked_lead_id is None:
            candidates = [
                item
                for item in projected
                if item["id"] not in ("", "-1", "None")
                and (
                    item["obs_type"] == 1
                    or "VEHICLE" in item["role"].upper()
                )
            ]
            if candidates:
                lead = min(candidates, key=lambda item: item["gap"])
                self.locked_lead_id = lead["id"]
                self.locked_lead_description = lead["role"]
                print(
                    "[roundabout][LOCK] lead_id={} role={} "
                    "gap={:.3f}m d_rel={:.3f}m".format(
                        lead["id"],
                        lead["role"] or "unknown",
                        lead["gap"],
                        lead["relative_d"],
                    )
                )
                return lead
            return None
        return next(
            (
                item
                for item in projected
                if item["id"] == self.locked_lead_id
            ),
            None,
        )

    def _longitudinal_target(
        self, ego_values, ego_projection, obstacles
    ):
        if self.mode == "direct":
            return self.max_speed, None, "ROUNDABOUT_DIRECT", ""

        lead = self._select_or_find_locked_lead(
            ego_values, ego_projection, obstacles
        )
        if lead is None:
            reason = (
                "waiting for initial same-lane lead"
                if self.locked_lead_id is None
                else "locked lead temporarily unavailable"
            )
            return 0.0, None, "ROUNDABOUT_FOLLOW_WAIT", reason

        gap_error = lead["gap"] - self.desired_gap
        target_speed = _clip(
            lead["longitudinal_speed"]
            + self.gap_gain * gap_error,
            0.0,
            self.max_speed,
        )
        return target_speed, lead, "ROUNDABOUT_FOLLOW", ""

    def plan(self, ego, obstacles, global_path, map_name=""):
        """Update the fixed lead/gap target without trajectory sampling."""
        self.map_name = str(map_name or self.map_name)
        if not self.enabled:
            self.last_plan = PlanResult(
                None, 0.0, "ROUNDABOUT_OFF", False, ""
            )
            self.last_debug = {"mode": self.mode}
            return self.last_plan
        if not self.reference.update(global_path):
            self.last_plan = PlanResult(
                None,
                0.0,
                "ROUNDABOUT_STOP",
                True,
                "global path unavailable",
            )
            self.last_debug = {
                "mode": self.mode,
                "reason": "global path unavailable",
            }
            return self.last_plan

        ego_values = self._ego_values(ego)
        projection = self.reference.project(
            ego_values["x"], ego_values["y"]
        )
        if projection is None:
            self.last_plan = PlanResult(
                None,
                0.0,
                "ROUNDABOUT_STOP",
                True,
                "ego projection unavailable",
            )
            self.last_debug = {
                "mode": self.mode,
                "reason": "ego projection unavailable",
            }
            return self.last_plan

        target_speed, lead, behavior, reason = (
            self._longitudinal_target(
                ego_values, projection, obstacles
            )
        )
        lead_debug = None
        if lead is not None:
            lead_debug = {
                key: lead[key]
                for key in (
                    "id",
                    "gap",
                    "s",
                    "d",
                    "relative_d",
                    "longitudinal_speed",
                    "role",
                )
            }
        self.last_debug = {
            "mode": self.mode,
            "projection": dict(projection),
            "heading_error": _wrap(
                projection["yaw"] - ego_values["yaw"]
            ),
            "desired_speed": target_speed,
            "desired_gap": self.desired_gap,
            "locked_lead_id": self.locked_lead_id,
            "locked_lead_role": self.locked_lead_description,
            "lead": lead_debug,
            "lead_available": lead is not None,
            "sampling_enabled": False,
            "prediction_enabled": False,
            "exclusive_control": True,
            "reason": reason,
        }
        self.last_plan = PlanResult(
            None,
            target_speed,
            behavior,
            False,
            reason,
        )
        now = time.monotonic()
        if now - self._last_status_print >= 0.5:
            self._last_status_print = now
            if lead is None:
                lead_text = "none"
            else:
                lead_text = "{} gap={:.2f}m v={:.2f}m/s".format(
                    lead["id"],
                    lead["gap"],
                    lead["longitudinal_speed"],
                )
            print(
                "[roundabout] mode={} behavior={} target_v={:.2f} "
                "lead={} d={:.2f}m".format(
                    self.mode,
                    behavior,
                    target_speed,
                    lead_text,
                    projection["d"],
                )
            )
        return self.last_plan

    def _lateral_control(self, ego_values, projection, dt):
        heading_error = _wrap(
            projection["yaw"] - ego_values["yaw"]
        )
        lateral_error = projection["d"]
        speed = ego_values["speed"]
        front_angle = (
            math.atan(self.wheelbase * projection["kappa"])
            + 0.85 * heading_error
            - math.atan2(1.4 * lateral_error, speed + 3.0)
        )
        raw_steer = math.degrees(front_angle) * self.steering_ratio
        raw_steer = _clip(
            raw_steer, -self.max_steer_deg, self.max_steer_deg
        )
        change = self.steer_rate_deg_s * max(0.005, float(dt))
        steer = _clip(
            raw_steer,
            self.last_steer - change,
            self.last_steer + change,
        )
        self.last_steer = steer
        return steer, heading_error, lateral_error, raw_steer

    def control(
        self,
        ego,
        obstacles,
        global_path,
        dt,
        map_name="",
        steering_feedback=None,
    ):
        """Compute one exclusive direct/follow control command."""
        plan = self.plan(ego, obstacles, global_path, map_name)
        ego_values = self._ego_values(ego)
        projection = self.last_debug.get("projection")
        if projection is None:
            acc = (
                -self.max_decel
                if ego_values["speed"] > 0.1
                else 0.0
            )
            self.last_debug.update(
                {
                    "desired_acc": acc,
                    "output_acc": acc,
                    "output_steer": 0.0,
                }
            )
            return ControlOutput(acc, 0.0, 0.0)

        steer, heading_error, lateral_error, raw_steer = (
            self._lateral_control(ego_values, projection, dt)
        )
        target_speed = plan.target_speed
        speed_error = target_speed - ego_values["speed"]
        acc = _clip(
            self.speed_gain * speed_error,
            -self.max_decel,
            self.max_accel,
        )
        lead = self.last_debug.get("lead")
        if (
            self.mode == "follow"
            and lead is not None
            and lead["gap"] <= max(0.15, 0.35 * self.desired_gap)
        ):
            acc = -self.max_decel
            target_speed = min(
                target_speed, lead["longitudinal_speed"]
            )
        self.last_debug.update(
            {
                "speed_error": target_speed - ego_values["speed"],
                "desired_acc": acc,
                "planned_accel": 0.0,
                "output_acc": acc,
                "output_steer": steer,
                "lookahead": 0.0,
                "lateral_error": lateral_error,
                "path_lateral_offset": lateral_error,
                "actual_path_lateral_offset": lateral_error,
                "heading_error": heading_error,
                "global_heading_error": heading_error,
                "reference_curvature": projection["kappa"],
                "raw_steer": raw_steer,
                "filtered_steer": steer,
                "steering_limit": self.max_steer_deg,
                "steering_rate_limit": self.steer_rate_deg_s,
                "steering_feedback": steering_feedback,
                "roundabout_mode": self.mode,
                "roundabout_exclusive": True,
                "centerline_control_active": True,
                "reason": plan.reason,
            }
        )
        return ControlOutput(acc, target_speed, steer)
