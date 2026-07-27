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
        gap_gain=1.5,
        speed_gain=1.8,
        follow_max_accel=8.0,
        catchup_speed=6.0,
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
        self.follow_max_accel = max(
            0.1, min(self.max_accel, float(follow_max_accel))
        )
        self.catchup_speed = max(0.1, float(catchup_speed))
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
            "model_name": str(
                getattr(obstacle, "model_name", "")
            ),
        }

    @staticmethod
    def _path_half_extent(length, width, object_yaw, path_yaw):
        delta = _wrap(object_yaw - path_yaw)
        return 0.5 * (
            length * abs(math.cos(delta))
            + width * abs(math.sin(delta))
        )

    def _lead_geometry(self, ego_values, ego_projection, raw):
        """Measure one obstacle without using lane gates.

        The initial lock is lane-gated, but a fixed target must remain
        observable while either vehicle is slightly off the reference path.
        """
        item = self._obstacle_values(raw)
        dx = item["x"] - ego_values["x"]
        dy = item["y"] - ego_values["y"]
        cos_ego = math.cos(ego_values["yaw"])
        sin_ego = math.sin(ego_values["yaw"])
        forward = dx * cos_ego + dy * sin_ego
        lateral = -dx * sin_ego + dy * cos_ego
        heading_delta = _wrap(item["yaw"] - ego_values["yaw"])
        ego_extent = 0.5 * ego_values["length"]
        obstacle_extent = self._path_half_extent(
            item["length"],
            item["width"],
            item["yaw"],
            ego_values["yaw"],
        )
        ego_gap = forward - ego_extent - obstacle_extent
        ego_longitudinal_speed = max(
            0.0, item["vx"] * cos_ego + item["vy"] * sin_ego
        )

        projection = self.reference.project(item["x"], item["y"])
        path_gap = None
        path_speed = None
        relative_d = None
        if projection is not None:
            relative_d = projection["d"] - ego_projection["d"]
            path_ego_extent = self._path_half_extent(
                ego_values["length"],
                ego_values["width"],
                ego_values["yaw"],
                ego_projection["yaw"],
            )
            path_obstacle_extent = self._path_half_extent(
                item["length"],
                item["width"],
                item["yaw"],
                projection["yaw"],
            )
            centre_distance = projection["s"] - ego_projection["s"]
            path_gap = (
                centre_distance
                - path_ego_extent
                - path_obstacle_extent
            )
            path_speed = max(
                0.0,
                item["vx"] * math.cos(projection["yaw"])
                + item["vy"] * math.sin(projection["yaw"]),
            )

        # Along a curve, Frenet distance is more useful than the ego x-axis.
        # If projection puts the target behind, retain the physical ego-frame
        # measurement so a fixed ID cannot disappear because of topology.
        use_path = (
            path_gap is not None
            and projection["s"] > ego_projection["s"]
            and forward > -0.5 * ego_values["length"]
        )
        item.update(
            {
                "s": (
                    projection["s"] if projection is not None else None
                ),
                "d": (
                    projection["d"] if projection is not None else None
                ),
                "relative_d": relative_d,
                "path_yaw": (
                    projection["yaw"]
                    if projection is not None
                    else ego_values["yaw"]
                ),
                "forward": forward,
                "lateral": lateral,
                "heading_delta": heading_delta,
                "ego_gap": ego_gap,
                "path_gap": path_gap,
                "gap_source": "path" if use_path else "ego",
                "gap": path_gap if use_path else ego_gap,
                "longitudinal_speed": (
                    path_speed if use_path else ego_longitudinal_speed
                ),
            }
        )
        return item

    def _select_or_find_locked_lead(
        self, ego_values, ego_projection, obstacles
    ):
        measured = [
            self._lead_geometry(
                ego_values, ego_projection, raw
            )
            for raw in (obstacles or [])
        ]
        if self.locked_lead_id is None:
            candidates = [
                item
                for item in measured
                if item["id"] not in ("", "-1", "None")
                and (
                    item["obs_type"] == 1
                    or "VEHICLE" in item["role"].upper()
                )
                and item["forward"] > 0.0
                and abs(item["lateral"])
                <= self.lane_half_width + 0.5 * item["width"]
                and abs(item["heading_delta"]) <= math.radians(55.0)
            ]
            if candidates:
                lead = min(
                    candidates, key=lambda item: item["forward"]
                )
                self.locked_lead_id = lead["id"]
                self.locked_lead_description = lead["role"]
                print(
                    "[roundabout][LOCK] lead_id={} role={} "
                    "model={} gap={:.3f}m source={} "
                    "forward={:.3f}m lateral={:.3f}m "
                    "heading_delta={:.1f}deg size={:.2f}x{:.2f}m".format(
                        lead["id"],
                        lead["role"] or "unknown",
                        lead["model_name"] or "unknown",
                        lead["gap"],
                        lead["gap_source"],
                        lead["forward"],
                        lead["lateral"],
                        math.degrees(lead["heading_delta"]),
                        lead["length"],
                        lead["width"],
                    )
                )
                return lead
            return None
        return next(
            (
                item
                for item in measured
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
        speed_correction = _clip(
            self.gap_gain * gap_error,
            -self.catchup_speed,
            self.catchup_speed,
        )
        target_speed = lead["longitudinal_speed"] + speed_correction
        if gap_error > 0.0:
            # Never request a closing speed that cannot be removed before the
            # configured one-metre gap, even if catch-up gain is increased.
            safe_closing_speed = math.sqrt(
                2.0 * self.max_decel * gap_error
            )
            target_speed = min(
                target_speed,
                lead["longitudinal_speed"] + safe_closing_speed,
            )
        target_speed = _clip(target_speed, 0.0, self.max_speed)
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
                    "forward",
                    "lateral",
                    "ego_gap",
                    "path_gap",
                    "gap_source",
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

    def _lateral_control(
        self, ego_values, projection, dt, target_speed
    ):
        heading_error = _wrap(
            projection["yaw"] - ego_values["yaw"]
        )
        lateral_error = projection["d"]
        speed = ego_values["speed"]
        if speed < 0.25 and target_speed < 0.1:
            self.last_steer = 0.0
            return 0.0, heading_error, lateral_error, 0.0
        feedforward = math.atan(
            self.wheelbase * projection["kappa"]
        )
        feedback = (
            0.55 * heading_error
            - math.atan2(
                0.55 * lateral_error, max(speed, 5.0)
            )
        )
        feedback_steer = _clip(
            math.degrees(feedback) * self.steering_ratio,
            -8.0,
            8.0,
        )
        raw_steer = (
            math.degrees(feedforward) * self.steering_ratio
            + feedback_steer
        )
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

        target_speed = plan.target_speed
        steer, heading_error, lateral_error, raw_steer = (
            self._lateral_control(
                ego_values, projection, dt, target_speed
            )
        )
        speed_error = target_speed - ego_values["speed"]
        accel_ceiling = (
            self.follow_max_accel
            if self.mode == "follow"
            else self.max_accel
        )
        acc = _clip(
            self.speed_gain * speed_error,
            -self.max_decel,
            accel_ceiling,
        )
        lead = self.last_debug.get("lead")
        if self.mode == "follow" and lead is not None:
            lead_speed = lead["longitudinal_speed"]
            closing_speed = ego_values["speed"] - lead_speed
            gap_margin = lead["gap"] - self.desired_gap
            # Relative-speed damping prevents the old 20 m/s² launch followed
            # by full braking. It is intentionally independent of sampling.
            acc += 0.9 * (lead_speed - ego_values["speed"])
            acc = _clip(acc, -self.max_decel, accel_ceiling)
            stopping_distance = (
                closing_speed * closing_speed
                / (2.0 * self.max_decel)
                if closing_speed > 0.0
                else 0.0
            )
            if (
                closing_speed > 0.0
                and gap_margin
                <= stopping_distance + max(0.15, 0.15 * closing_speed)
            ):
                acc = -self.max_decel
            elif gap_margin <= 0.0:
                target_speed = min(target_speed, lead_speed)
                if closing_speed > 0.0:
                    acc = -self.max_decel
                else:
                    acc = min(0.0, acc)
            self.last_debug.update(
                {
                    "closing_speed": closing_speed,
                    "gap_margin": gap_margin,
                    "stopping_distance": stopping_distance,
                    "follow_accel_limit": self.follow_max_accel,
                    "catchup_speed_limit": self.catchup_speed,
                }
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
