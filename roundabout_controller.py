"""Exclusive global-path controller with IDM longitudinal following.

There is no behaviour tree, obstacle prediction, or trajectory sampling:

* lateral control follows the supplied global path;
* vehicles with ``abs(path_d) < lane_half_width`` are path traffic;
* a clear path uses the free-road IDM term;
* the nearest path vehicle uses full IDM car following;
* TTC is an emergency brake only;
* look-ahead path curvature caps the IDM desired speed.
"""

import math
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
    """Global-path lateral control plus cruise/follow IDM."""

    VALID_MODES = ("off", "follow", "direct")

    def __init__(
        self,
        mode="off",
        desired_gap=1.0,
        max_speed=30.0,
        max_accel=20.0,
        max_decel=15.5,
        lane_half_width=1.0,
        follow_max_accel=8.0,
        time_headway=0.8,
        idm_comfort_decel=6.0,
        idm_delta=4.0,
        ttc_emergency=1.0,
        curve_lateral_accel=4.0,
        curve_lookahead_time=2.0,
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
        self.lane_half_width = max(0.1, float(lane_half_width))
        self.follow_max_accel = max(
            0.1, min(self.max_accel, float(follow_max_accel))
        )
        self.time_headway = max(0.1, float(time_headway))
        self.idm_comfort_decel = max(
            0.1, float(idm_comfort_decel)
        )
        self.idm_delta = max(1.0, float(idm_delta))
        self.ttc_emergency = max(0.1, float(ttc_emergency))
        self.curve_lateral_accel = max(
            0.1, float(curve_lateral_accel)
        )
        self.curve_lookahead_time = max(
            0.1, float(curve_lookahead_time)
        )
        self.wheelbase = max(0.5, float(wheelbase))
        self.steering_ratio = max(0.1, float(steering_ratio))
        self.max_steer_deg = max(1.0, float(max_steer_deg))
        self.steer_rate_deg_s = max(1.0, float(steer_rate_deg_s))

        self.reference = ReferencePath()
        self.map_name = ""
        # Kept under the old name because run.py and existing debug tooling
        # expose it. It now means current nearest path lead, not a permanent
        # session lock.
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

    @staticmethod
    def _path_half_extent(length, width, object_yaw, path_yaw):
        delta = _wrap(object_yaw - path_yaw)
        return 0.5 * (
            length * abs(math.cos(delta))
            + width * abs(math.sin(delta))
        )

    @staticmethod
    def _ego_values(ego):
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

    @staticmethod
    def _obstacle_values(obstacle):
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

    def _project_vehicle(self, ego_values, ego_projection, obstacle):
        item = self._obstacle_values(obstacle)
        projection = self.reference.project(item["x"], item["y"])
        if projection is None:
            return None

        ego_extent = self._path_half_extent(
            ego_values["length"],
            ego_values["width"],
            ego_values["yaw"],
            ego_projection["yaw"],
        )
        obstacle_extent = self._path_half_extent(
            item["length"],
            item["width"],
            item["yaw"],
            projection["yaw"],
        )
        centre_distance = projection["s"] - ego_projection["s"]
        path_speed = max(
            0.0,
            item["vx"] * math.cos(projection["yaw"])
            + item["vy"] * math.sin(projection["yaw"]),
        )
        item.update(
            {
                "s": projection["s"],
                "d": projection["d"],
                "relative_d": (
                    projection["d"] - ego_projection["d"]
                ),
                "path_yaw": projection["yaw"],
                "path_heading_delta": _wrap(
                    item["yaw"] - projection["yaw"]
                ),
                "centre_distance": centre_distance,
                "gap": centre_distance - ego_extent - obstacle_extent,
                "longitudinal_speed": path_speed,
            }
        )
        return item

    def _select_path_lead(
        self, ego_values, ego_projection, obstacles
    ):
        candidates = []
        for obstacle in obstacles or []:
            item = self._project_vehicle(
                ego_values, ego_projection, obstacle
            )
            if item is None:
                continue
            is_vehicle = (
                item["obs_type"] == 1
                or "VEHICLE" in item["role"].upper()
            )
            if (
                is_vehicle
                and item["id"] not in ("", "-1", "None")
                and abs(item["d"]) < self.lane_half_width
                and item["centre_distance"] > 0.0
                and abs(item["path_heading_delta"])
                <= math.radians(55.0)
            ):
                candidates.append(item)

        lead = (
            min(candidates, key=lambda item: item["centre_distance"])
            if candidates
            else None
        )
        previous_id = self.locked_lead_id
        self.locked_lead_id = lead["id"] if lead else None
        self.locked_lead_description = lead["role"] if lead else None
        if lead is not None and lead["id"] != previous_id:
            print(
                "[roundabout][LEAD] id={} role={} model={} "
                "gap={:.3f}m path_d={:.3f}m path_ds={:.3f}m "
                "size={:.2f}x{:.2f}m".format(
                    lead["id"],
                    lead["role"] or "unknown",
                    lead["model_name"] or "unknown",
                    lead["gap"],
                    lead["d"],
                    lead["centre_distance"],
                    lead["length"],
                    lead["width"],
                )
            )
        return lead, len(candidates)

    def _curve_speed_limit(self, projection, speed):
        lookahead = _clip(
            8.0 + speed * self.curve_lookahead_time,
            15.0,
            45.0,
        )
        sample_count = max(2, int(math.ceil(lookahead)) + 1)
        stations = [
            projection["s"]
            + lookahead * index / float(sample_count - 1)
            for index in range(sample_count)
        ]
        _, _, _, kappas = self.reference.sample(stations)
        max_curvature = max(
            [abs(_finite(value)) for value in kappas] + [0.0]
        )
        if max_curvature <= 1e-4:
            curve_speed = self.max_speed
        else:
            curve_speed = math.sqrt(
                self.curve_lateral_accel / max_curvature
            )
        return (
            _clip(curve_speed, 0.5, self.max_speed),
            max_curvature,
            lookahead,
        )

    def plan(self, ego, obstacles, global_path, map_name=""):
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

        curve_speed, max_curvature, curve_lookahead = (
            self._curve_speed_limit(
                projection, ego_values["speed"]
            )
        )
        lead = None
        path_vehicle_count = 0
        if self.mode == "direct":
            behavior = "ROUNDABOUT_DIRECT"
            reason = ""
            target_speed = curve_speed
            self.locked_lead_id = None
            self.locked_lead_description = None
        else:
            lead, path_vehicle_count = self._select_path_lead(
                ego_values, projection, obstacles
            )
            if lead is None:
                behavior = "ROUNDABOUT_CRUISE"
                reason = "global path clear"
                target_speed = curve_speed
            else:
                behavior = "ROUNDABOUT_FOLLOW"
                reason = ""
                follow_speed = lead["longitudinal_speed"] + max(
                    0.0,
                    (lead["gap"] - self.desired_gap)
                    / self.time_headway,
                )
                target_speed = _clip(
                    min(curve_speed, follow_speed),
                    0.0,
                    self.max_speed,
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
                    "centre_distance",
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
            "time_headway": self.time_headway,
            "curve_speed_limit": curve_speed,
            "lookahead_max_curvature": max_curvature,
            "curve_lookahead": curve_lookahead,
            "path_d_threshold": self.lane_half_width,
            "path_vehicle_count": path_vehicle_count,
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
            None, target_speed, behavior, False, reason
        )

        now = time.monotonic()
        if now - self._last_status_print >= 0.5:
            self._last_status_print = now
            lead_text = (
                "none"
                if lead is None
                else "{} gap={:.2f}m v={:.2f}m/s d={:.2f}m".format(
                    lead["id"],
                    lead["gap"],
                    lead["longitudinal_speed"],
                    lead["d"],
                )
            )
            print(
                "[roundabout] mode={} behavior={} target_v={:.2f} "
                "curve_v={:.2f} lead={} ego_d={:.2f}m".format(
                    self.mode,
                    behavior,
                    target_speed,
                    curve_speed,
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
        raw_steer = _clip(
            math.degrees(feedforward) * self.steering_ratio
            + feedback_steer,
            -self.max_steer_deg,
            self.max_steer_deg,
        )
        change = self.steer_rate_deg_s * max(0.005, float(dt))
        steer = _clip(
            raw_steer,
            self.last_steer - change,
            self.last_steer + change,
        )
        self.last_steer = steer
        return steer, heading_error, lateral_error, raw_steer

    def _idm_acceleration(self, ego_speed, curve_speed, lead):
        acceleration = (
            self.max_accel
            if self.mode == "direct"
            else self.follow_max_accel
        )
        desired_speed = max(0.1, curve_speed)
        free_term = (
            max(0.0, ego_speed) / desired_speed
        ) ** self.idm_delta
        interaction_term = 0.0
        dynamic_gap = self.desired_gap
        closing_speed = 0.0
        ttc = float("inf")
        emergency = False

        if self.mode == "follow" and lead is not None:
            lead_speed = lead["longitudinal_speed"]
            closing_speed = ego_speed - lead_speed
            dynamic_gap = self.desired_gap + max(
                0.0,
                ego_speed * self.time_headway
                + ego_speed
                * closing_speed
                / (
                    2.0
                    * math.sqrt(
                        acceleration * self.idm_comfort_decel
                    )
                ),
            )
            effective_gap = max(0.05, lead["gap"])
            interaction_term = (
                dynamic_gap / effective_gap
            ) ** 2
            if closing_speed > 0.05 and lead["gap"] > 0.0:
                ttc = lead["gap"] / closing_speed
            emergency = (
                lead["gap"] <= 0.05
                or ttc <= self.ttc_emergency
            )

        command = acceleration * (
            1.0 - free_term - interaction_term
        )
        if emergency:
            command = -self.max_decel
        command = _clip(
            command, -self.max_decel, acceleration
        )
        return {
            "acc": command,
            "desired_dynamic_gap": dynamic_gap,
            "closing_speed": closing_speed,
            "ttc": ttc,
            "ttc_emergency_active": emergency,
            "free_road_term": free_term,
            "interaction_term": interaction_term,
        }

    def control(
        self,
        ego,
        obstacles,
        global_path,
        dt,
        map_name="",
        steering_feedback=None,
    ):
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
        lead = self.last_debug.get("lead")
        idm = self._idm_acceleration(
            ego_values["speed"],
            self.last_debug["curve_speed_limit"],
            lead,
        )
        if idm["ttc_emergency_active"]:
            target_speed = 0.0
            self.last_plan = PlanResult(
                None,
                0.0,
                "ROUNDABOUT_TTC_BRAKE",
                True,
                "TTC emergency brake",
            )
            self.last_debug["reason"] = "TTC emergency brake"

        self.last_debug.update(
            {
                "speed_error": target_speed - ego_values["speed"],
                "desired_acc": idm["acc"],
                "planned_accel": idm["acc"],
                "output_acc": idm["acc"],
                "output_steer": steer,
                "desired_dynamic_gap": idm["desired_dynamic_gap"],
                "closing_speed": idm["closing_speed"],
                "ttc": idm["ttc"],
                "ttc_threshold": self.ttc_emergency,
                "ttc_emergency_active": idm[
                    "ttc_emergency_active"
                ],
                "idm_free_road_term": idm["free_road_term"],
                "idm_interaction_term": idm["interaction_term"],
                "idm_acceleration": idm["acc"],
                "lookahead": self.last_debug["curve_lookahead"],
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
            }
        )
        return ControlOutput(idm["acc"], target_speed, steer)
