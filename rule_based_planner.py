"""Pure-Python rule-based local planning and stable vehicle control.

The module deliberately has no ROS, detector, reinforcement-learning, or
simulator imports.  ``run.py`` supplies an ego object, the structured obstacle
objects produced by :class:`predictor.Predictor`, and the latest global path.

Planning follows the useful part of the former C++ polynomial planner:

* project ego/obstacles into the global-path Frenet frame;
* sample quartic longitudinal and quintic lateral trajectories;
* reject trajectories violating dynamics, road-boundary, or collision rules;
* rank the remaining candidates by safety, comfort, progress, and continuity;
* track the selected Cartesian trajectory with curvature feed-forward plus a
  Stanley/Pure-Pursuit blend, followed by steering/rate/jerk limits.

The implementation favours predictable simulator behaviour over aggressive
lane changes.  Without explicit map road boundaries it stays inside a
configurable corridor around the global path and stops if no safe path exists.
"""

import fnmatch
import json
import math
import os
import time

import numpy as np

try:
    from speed_limits import scene_speed_limit_for_map
except ImportError:  # pragma: no cover - package-style import
    from .speed_limits import scene_speed_limit_for_map


EPS = 1e-6


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _wrap(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _unique(values, tolerance=1e-4):
    result = []
    for value in values:
        value = float(value)
        if not any(abs(value - old) <= tolerance for old in result):
            result.append(value)
    return result


class PlannerConfig(object):
    """Central tuning values for the rule planner and controller."""

    def __init__(self):
        # A safe default is one lane around the global path.  Wider maps can
        # opt in to adjacent-lane sampling through environment variables.
        half_width = float(os.environ.get("RULE_ROAD_HALF_WIDTH", "1.55"))
        self.road_left_bound = float(
            os.environ.get("RULE_ROAD_LEFT_BOUND", str(half_width))
        )
        self.road_right_bound = float(
            os.environ.get("RULE_ROAD_RIGHT_BOUND", str(-half_width))
        )
        self.road_margin = float(
            os.environ.get("RULE_ROAD_MARGIN", "0.10")
        )
        self.lateral_sample_step = max(
            0.15,
            abs(float(os.environ.get("RULE_LATERAL_SAMPLE_STEP", "0.35"))),
        )
        horizon_text = os.environ.get(
            "RULE_PLANNING_HORIZONS",
            "2.5,3.5,4.5,6.0",
        )
        try:
            horizons = tuple(sorted(
                float(value.strip())
                for value in horizon_text.split(",")
                if value.strip() and float(value.strip()) >= 1.0
            ))
        except ValueError:
            horizons = ()
        self.horizons = horizons or (2.5, 3.5, 4.5, 6.0)
        self.sample_dt = _clip(
            float(os.environ.get("RULE_SAMPLE_DT", "0.10")),
            0.05,
            0.20,
        )

        self.max_speed = float(
            os.environ.get("RULE_MAX_SPEED", "inf")
        )
        self.expected_speed_mps = None
        self.expected_speed_source = "map-category-fallback"
        self.override_map_speed_limit = (
            os.environ.get("RULE_OVERRIDE_MAP_SPEED_LIMIT", "0") == "1"
        )
        # The direct route planner may encode its own curvature-derived speed
        # in every path point.  Applying that cap and then applying
        # ``_curve_speed_limit`` again double-limits the vehicle.  The rule
        # planner therefore owns curve speed by default; enable this only
        # when the incoming path contains an authoritative external limit.
        self.respect_path_speed_limit = (
            os.environ.get("RULE_RESPECT_PATH_SPEED_LIMIT", "0") == "1"
        )
        self.max_accel = float(
            os.environ.get("RULE_MAX_ACCEL", "10.0")
        )
        self.ignore_obstacles = (
            os.environ.get("RULE_IGNORE_OBSTACLES", "0") == "1"
        )
        self.scenario_overrides_path = os.environ.get(
            "RULE_SCENARIO_OVERRIDES",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "scenario_overrides.json",
            ),
        )
        self.max_decel = float(
            os.environ.get("RULE_MAX_DECEL", "15.5")
        )
        self.max_lon_jerk = float(
            os.environ.get("RULE_MAX_LON_JERK", "100.0")
        )
        self.max_lat_speed = float(
            os.environ.get("RULE_MAX_LAT_SPEED", "6.0")
        )
        self.max_lat_accel = float(
            os.environ.get("RULE_MAX_LAT_ACCEL", "10.0")
        )
        self.max_lat_jerk = float(
            os.environ.get("RULE_MAX_LAT_JERK", "100.0")
        )
        # Updated below after the steering geometry is known.  The old value
        # (0.30 1/m) described a front-wheel angle of roughly 45 degrees and
        # allowed trajectories that a 42-degree steering-wheel command could
        # never track.
        self.max_curvature = 0.03

        self.time_headway = max(
            0.30,
            float(os.environ.get("RULE_FOLLOW_TIME_HEADWAY", "1.0")),
        )
        self.minimum_gap = max(
            1.0,
            float(os.environ.get("RULE_MINIMUM_GAP", "2.5")),
        )
        # Collision is a hard constraint.  Padding absorbs detector box-size
        # error and path-tracking error; prediction growth absorbs velocity
        # error over the planning horizon.
        self.collision_margin = max(
            0.40,
            float(os.environ.get("RULE_COLLISION_MARGIN", "0.65")),
        )
        self.pedestrian_extra_margin = float(
            os.environ.get("RULE_PEDESTRIAN_EXTRA_MARGIN", "0.85")
        )
        self.prediction_growth = float(
            os.environ.get("RULE_PREDICTION_GROWTH", "0.20")
        )
        self.lateral_prediction_growth = float(
            os.environ.get("RULE_LATERAL_PREDICTION_GROWTH", "0.05")
        )
        self.unknown_speed_growth = float(
            os.environ.get("RULE_UNKNOWN_SPEED_GROWTH", "1.50")
        )
        self.ego_length_padding = float(
            os.environ.get("RULE_EGO_LENGTH_PADDING", "0.40")
        )
        self.ego_width_padding = float(
            os.environ.get("RULE_EGO_WIDTH_PADDING", "0.25")
        )
        self.obstacle_length_padding = float(
            os.environ.get("RULE_OBSTACLE_LENGTH_PADDING", "0.40")
        )
        self.obstacle_width_padding = float(
            os.environ.get("RULE_OBSTACLE_WIDTH_PADDING", "0.30")
        )
        self.collision_check_dt = _clip(
            float(os.environ.get("RULE_COLLISION_CHECK_DT", "0.05")),
            0.02,
            0.10,
        )
        self.rear_obstacle_distance = float(
            os.environ.get("RULE_REAR_OBSTACLE_DISTANCE", "18.0")
        )
        # The simulator's background traffic is a fixed replay and will not
        # react to the ego vehicle.  A rear follower must therefore not make
        # every forward candidate collide and force the ego to stop.  Rear
        # vehicles in an adjacent lane remain hard constraints so an avoidance
        # manoeuvre cannot merge into them.
        self.non_yielding_replay_traffic = (
            os.environ.get(
                "RULE_NON_YIELDING_REPLAY_TRAFFIC", "1"
            ) == "1"
        )
        self.rear_follow_lateral_tolerance = max(
            0.50,
            float(
                os.environ.get(
                    "RULE_REAR_FOLLOW_LATERAL_TOLERANCE", "1.55"
                )
            ),
        )
        self.rear_pressure_distance = max(
            3.0,
            float(
                os.environ.get(
                    "RULE_REAR_PRESSURE_DISTANCE", "12.0"
                )
            ),
        )
        self.rear_pressure_closing_speed = max(
            0.0,
            float(
                os.environ.get(
                    "RULE_REAR_PRESSURE_CLOSING_SPEED", "0.5"
                )
            ),
        )
        self.emergency_ttc = 1.2
        self.follow_distance = 45.0
        self.goal_stop_buffer = 1.0
        self.stop_at_goal = (
            os.environ.get("RULE_STOP_AT_GOAL", "0") == "1"
        )
        self.goal_decel = max(
            0.5,
            float(os.environ.get("RULE_GOAL_DECEL", "8.0")),
        )
        self.static_obstacle_speed = 0.50
        # A lead vehicle is a centre-line decision object, not every box whose
        # padded edge touches the ego envelope.  The full oriented-box
        # collision checker still protects against side-swipe collisions.
        self.lead_lateral_tolerance = max(
            0.50,
            float(
                os.environ.get(
                    "RULE_LEAD_LATERAL_TOLERANCE", "1.00"
                )
            ),
        )
        self.static_avoidance_min_hits = max(
            1,
            int(
                os.environ.get(
                    "RULE_STATIC_AVOIDANCE_MIN_HITS", "2"
                )
            ),
        )
        self.static_avoidance_trigger_distance = max(
            6.0,
            float(
                os.environ.get(
                    "RULE_STATIC_AVOIDANCE_TRIGGER_DISTANCE",
                    "18.0",
                )
            ),
        )
        self.direct_lane_side_clearance = max(
            0.10,
            float(
                os.environ.get(
                    "RULE_DIRECT_LANE_SIDE_CLEARANCE", "0.35"
                )
            ),
        )
        self.launch_priority_speed = max(
            0.0,
            float(
                os.environ.get(
                    "RULE_LAUNCH_PRIORITY_SPEED", "2.00"
                )
            ),
        )
        self.static_avoidance_speed = float(
            os.environ.get("RULE_AVOIDANCE_SPEED", "10.0")
        )
        self.avoidance_half_width = float(
            os.environ.get("RULE_AVOIDANCE_HALF_WIDTH", "3.60")
        )
        # A trajectory is not an avoidance manoeuvre merely because its
        # absolute Frenet d is on one side of the reference line.  Require an
        # actual lateral displacement from where avoidance started.
        self.minimum_bypass_shift = max(
            0.0,
            float(
                os.environ.get(
                    "RULE_MINIMUM_BYPASS_SHIFT", "0.05"
                )
            ),
        )
        self.static_side_clearance = max(
            0.20,
            float(
                os.environ.get(
                    "RULE_STATIC_SIDE_CLEARANCE", "0.30"
                )
            ),
        )

        self.max_lateral_accel = float(
            os.environ.get("RULE_MAX_CARTESIAN_LAT_ACCEL", "10.0")
        )
        # Comfort scoring uses 0.5 m/s^2, but treating that score threshold as
        # an unconditional steering-authority limit makes a vehicle already
        # above the curve speed physically unable to follow the road.  This
        # higher limit is used only for curve feedforward and active
        # centreline recovery; ordinary tracking remains at
        # ``max_lateral_accel``.
        self.max_tracking_lateral_accel = max(
            0.5,
            float(
                os.environ.get(
                    "RULE_MAX_TRACKING_LAT_ACCEL", "2.5"
                )
            ),
        )
        # ``comfort_mode`` is deliberately opt-in at the library level so
        # existing users of PlannerConfig keep their tuning.  run.py enables
        # it by default for the scored DriveSim entrypoint.
        self.comfort_mode = False
        self.max_yaw_rate = float("inf")
        self.curve_speed_factor = max(
            0.1,
            float(
                os.environ.get(
                    "RULE_CURVE_SPEED_FACTOR", "1.50"
                )
            ),
        )
        self.centerline_feedback_gain = max(
            0.0,
            float(
                os.environ.get(
                    "RULE_CENTERLINE_FEEDBACK_GAIN", "1.25"
                )
            ),
        )
        # For ordinary lane keeping, control global Frenet d directly with a
        # critically damped second-order law.  Reusing the repeatedly rebuilt
        # local RECOVER trajectory here made the steering reverse too late:
        # the ego crossed the centreline, generated a bend in the opposite
        # direction, and then repeated the same motion.
        self.centerline_natural_frequency = _clip(
            float(
                os.environ.get(
                    "RULE_CENTERLINE_NATURAL_FREQUENCY", "0.45"
                )
            ),
            0.10,
            2.00,
        )
        self.centerline_damping_ratio = _clip(
            float(
                os.environ.get(
                    "RULE_CENTERLINE_DAMPING_RATIO", "1.05"
                )
            ),
            0.40,
            3.00,
        )
        self.centerline_safety_stop_enabled = (
            os.environ.get(
                "RULE_CENTERLINE_SAFETY_STOP", "0"
            )
            == "1"
        )
        self.strict_alignment_speed_guard = (
            os.environ.get(
                "RULE_STRICT_ALIGNMENT_SPEED_GUARD", "0"
            )
            == "1"
        )
        self.controller_wheelbase = float(
            os.environ.get("RULE_WHEELBASE_M", "3.38")
        )
        self.steering_ratio = float(
            # Cam6's DriveSim kinematic chassis does not consume
            # ``steering_wheel_angle`` as a 1:1 front-wheel angle.  Replaying
            # steady portions of INS yaw, speed, and steering feedback gives
            # an effective ratio of about 1.6--1.8.  Using 1.0 made the
            # comfort limiter believe it commanded 0.5 m/s^2 while the
            # chassis produced only about 0.3 m/s^2, so the ego drifted
            # monotonically outside a bend.
            os.environ.get("RULE_STEERING_RATIO", "1.65")
        )
        self.steering_command_sign = -1.0 if float(
            # DriveSim's kinematic chassis uses the same sign as the model
            # front-wheel angle: positive turns left, negative turns right.
            # Keep the environment override for other chassis backends.
            os.environ.get("RULE_STEER_COMMAND_SIGN", "1.0")
        ) < 0.0 else 1.0
        self.max_steering_wheel_deg = float(
            os.environ.get("RULE_MAX_STEER_DEG", "42.0")
        )
        self.steering_rate_low = float(
            os.environ.get("RULE_STEER_RATE_LOW", "90.0")
        )
        self.steering_rate_mid = float(
            os.environ.get("RULE_STEER_RATE_MID", "70.0")
        )
        self.steering_rate_high = float(
            os.environ.get("RULE_STEER_RATE_HIGH", "50.0")
        )
        physical_curvature = math.tan(
            math.radians(self.max_steering_wheel_deg / self.steering_ratio)
        ) / max(self.controller_wheelbase, EPS)
        self.max_curvature = float(
            os.environ.get(
                "RULE_MAX_CURVATURE",
                str(
                    min(
                        0.25,
                        max(0.03, 0.95 * physical_curvature),
                    )
                ),
            )
        )

        # Cost weights.  Hard collision/boundary checks are performed before
        # these soft costs, so weights tune behaviour rather than safety.
        self.w_center = 0.35
        self.w_obstacle = 45.0
        self.w_continuity = 0.8
        self.w_comfort = 0.15
        self.w_speed = 15.0
        self.w_progress = 3.0
        self.w_lateral_change = 0.30
        self.w_terminal_center = 1.0
        self.w_direction_switch = 80.0

    def enable_comfort_mode(self):
        """Apply the evaluator's five comfort thresholds.

        Emergency collision braking remains allowed to bypass longitudinal
        jerk limiting in StableController. Ordinary longitudinal commands use
        a margin below the evaluator boundary because the score is computed
        from differentiated vehicle motion, not from the ideal command.
        """
        self.comfort_mode = True
        self.max_accel = 2.8
        self.max_decel = 2.6
        self.max_lon_jerk = 4.5
        self.max_lat_accel = 0.5
        self.max_lat_jerk = 1.0
        self.max_lateral_accel = 0.5
        self.max_yaw_rate = 0.5
        # A value above one knowingly exceeds the Cartesian lateral
        # acceleration threshold on a steady curve.
        self.curve_speed_factor = 1.0


class ReferencePath(object):
    """Validated and interpolatable global reference path."""

    def __init__(self):
        self.x = np.empty(0, dtype=float)
        self.y = np.empty(0, dtype=float)
        self.yaw = np.empty(0, dtype=float)
        self.kappa = np.empty(0, dtype=float)
        self.speed_limit = np.empty(0, dtype=float)
        self.s = np.empty(0, dtype=float)
        self.frame_id = ""
        self.stamp = None
        self._signature = None

    @property
    def valid(self):
        return self.x.size >= 2 and self.s.size == self.x.size

    @property
    def length(self):
        return float(self.s[-1]) if self.valid else 0.0

    def update(self, path):
        if path is None:
            return False
        try:
            xs = np.asarray(path.get("x", []), dtype=float).reshape(-1)
            ys = np.asarray(path.get("y", []), dtype=float).reshape(-1)
        except Exception:
            return False
        count = min(xs.size, ys.size)
        if count < 2:
            return False
        xs = xs[:count]
        ys = ys[:count]
        try:
            speed_limits = np.asarray(path.get("speed_limit", []), dtype=float).reshape(-1)
        except Exception:
            speed_limits = np.empty(0, dtype=float)
        if speed_limits.size != count:
            speed_limits = np.zeros(count, dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        speed_limits = speed_limits[valid]
        if xs.size < 2:
            return False

        keep = np.ones(xs.size, dtype=bool)
        keep[1:] = np.hypot(np.diff(xs), np.diff(ys)) > 1e-4
        xs = xs[keep]
        ys = ys[keep]
        speed_limits = speed_limits[keep]
        if xs.size < 2:
            return False

        signature = (
            xs.size,
            round(float(xs[0]), 3),
            round(float(ys[0]), 3),
            round(float(xs[-1]), 3),
            round(float(ys[-1]), 3),
            path.get("stamp"),
        )
        if signature == self._signature:
            return True

        ds = np.hypot(np.diff(xs), np.diff(ys))
        stations = np.concatenate(([0.0], np.cumsum(ds)))
        dx = np.gradient(xs, stations, edge_order=1)
        dy = np.gradient(ys, stations, edge_order=1)
        yaws = np.unwrap(np.arctan2(dy, dx))

        supplied_kappa = path.get("kappa")
        try:
            kappas = np.asarray(supplied_kappa, dtype=float).reshape(-1)
        except Exception:
            kappas = np.empty(0, dtype=float)
        if kappas.size != xs.size or not np.all(np.isfinite(kappas)):
            kappas = np.gradient(yaws, stations, edge_order=1)
        elif np.max(np.abs(kappas)) < 1e-9:
            # nav_msgs/Path normally carries no explicit curvature.
            kappas = np.gradient(yaws, stations, edge_order=1)
        kappas = np.clip(kappas, -0.5, 0.5)

        self.x = xs
        self.y = ys
        self.s = stations
        self.yaw = yaws
        self.kappa = kappas
        self.speed_limit = speed_limits
        self.frame_id = str(path.get("frame_id", "") or "")
        self.stamp = path.get("stamp")
        self._signature = signature
        return True

    def project(self, x, y):
        """Project a Cartesian point onto the closest path segment."""
        if not self.valid:
            return None
        px = float(x)
        py = float(y)
        x0 = self.x[:-1]
        y0 = self.y[:-1]
        vx = self.x[1:] - x0
        vy = self.y[1:] - y0
        length2 = np.maximum(vx * vx + vy * vy, EPS)
        ratios = np.clip(((px - x0) * vx + (py - y0) * vy) / length2, 0.0, 1.0)
        proj_x = x0 + ratios * vx
        proj_y = y0 + ratios * vy
        distance2 = (px - proj_x) ** 2 + (py - proj_y) ** 2
        index = int(np.argmin(distance2))
        seg_len = math.sqrt(float(length2[index]))
        # Treat the final tangent as an open ray.  This keeps projection and
        # control valid after crossing a non-stopping route endpoint instead
        # of snapping the ego to the last point and eventually declaring it
        # too far from the path.
        if index == self.x.size - 2 and ratios[index] >= 1.0 - 1e-9:
            end_yaw = float(self.yaw[-1])
            tangent_x = math.cos(end_yaw)
            tangent_y = math.sin(end_yaw)
            beyond = (
                (px - float(self.x[-1])) * tangent_x
                + (py - float(self.y[-1])) * tangent_y
            )
            if beyond > 0.0:
                ray_x = float(self.x[-1]) + beyond * tangent_x
                ray_y = float(self.y[-1]) + beyond * tangent_y
                ray_d = (
                    tangent_x * (py - ray_y)
                    - tangent_y * (px - ray_x)
                )
                return {
                    "s": self.length + beyond,
                    "d": float(ray_d),
                    "yaw": _wrap(end_yaw),
                    "kappa": 0.0,
                    "index": index,
                    "distance": abs(float(ray_d)),
                }
        cross = vx[index] * (py - proj_y[index]) - vy[index] * (px - proj_x[index])
        d = float(cross / max(seg_len, EPS))
        s = float(self.s[index] + ratios[index] * seg_len)
        yaw = float(self.yaw[index] + ratios[index] * (self.yaw[index + 1] - self.yaw[index]))
        kappa = float(
            self.kappa[index]
            + ratios[index] * (self.kappa[index + 1] - self.kappa[index])
        )
        return {
            "s": s,
            "d": d,
            "yaw": _wrap(yaw),
            "kappa": kappa,
            "index": index,
            "distance": math.sqrt(float(distance2[index])),
        }

    def sample(self, stations):
        stations = np.asarray(stations, dtype=float)
        clipped = np.clip(stations, self.s[0], self.s[-1])
        xs = np.interp(clipped, self.s, self.x)
        ys = np.interp(clipped, self.s, self.y)
        yaws = np.interp(clipped, self.s, self.yaw)
        kappas = np.interp(clipped, self.s, self.kappa)
        beyond = np.maximum(0.0, stations - self.s[-1])
        if np.any(beyond > 0.0):
            end_yaw = float(self.yaw[-1])
            xs = xs + beyond * math.cos(end_yaw)
            ys = ys + beyond * math.sin(end_yaw)
            yaws = np.where(beyond > 0.0, end_yaw, yaws)
            kappas = np.where(beyond > 0.0, 0.0, kappas)
        return xs, ys, yaws, kappas

    def frenet_to_cartesian(self, stations, offsets):
        ref_x, ref_y, ref_yaw, ref_kappa = self.sample(stations)
        offsets = np.asarray(offsets, dtype=float)
        xs = ref_x - offsets * np.sin(ref_yaw)
        ys = ref_y + offsets * np.cos(ref_yaw)
        if xs.size >= 3:
            dx = np.gradient(xs)
            dy = np.gradient(ys)
            yaw = np.unwrap(np.arctan2(dy, dx))
            arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))))
            if arc[-1] > 1e-3:
                kappa = np.gradient(yaw, np.maximum.accumulate(arc + np.arange(arc.size) * EPS))
            else:
                kappa = np.zeros_like(xs)
        else:
            yaw = ref_yaw
            kappa = ref_kappa
        return xs, ys, yaw, np.clip(kappa, -0.5, 0.5)


class Trajectory(object):
    def __init__(self, t, s, d, speed, accel, d_speed, d_accel, d_jerk):
        self.t = np.asarray(t, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.speed = np.asarray(speed, dtype=float)
        self.accel = np.asarray(accel, dtype=float)
        self.d_speed = np.asarray(d_speed, dtype=float)
        self.d_accel = np.asarray(d_accel, dtype=float)
        self.d_jerk = np.asarray(d_jerk, dtype=float)
        self.x = np.empty(0, dtype=float)
        self.y = np.empty(0, dtype=float)
        self.yaw = np.empty(0, dtype=float)
        self.kappa = np.empty(0, dtype=float)
        self.cost = float("inf")
        self.cost_components = {}
        self.minimum_clearance = float("inf")
        self.closest_obstacle_id = None
        self.closest_collision_time = None
        self.ignored_rear_ids = []


class PlanResult(object):
    def __init__(self, trajectory=None, target_speed=0.0, behavior="STOP", emergency=False, reason=""):
        self.trajectory = trajectory
        self.target_speed = max(0.0, _finite(target_speed))
        self.behavior = str(behavior)
        self.emergency = bool(emergency)
        self.reason = str(reason)


class ControlOutput(object):
    def __init__(self, acc=0.0, speed=0.0, steer=0.0):
        self.acc = float(acc)
        self.speed = max(0.0, float(speed))
        self.steer = float(steer)


class RuleBasedPlanner(object):
    """Frenet sampling planner driven only by explicit rules."""

    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.reference = ReferencePath()
        self.previous_trajectory = None
        self.previous_target_d = 0.0
        self.behavior = "STOP"
        self.map_name = ""
        self.last_debug_wall_time = 0.0
        self.last_debug = {}
        self.avoidance_side = 0
        self.avoidance_obstacle_id = None
        self.avoidance_origin_d = None
        self._last_speed_limit_debug = {}
        self._scenario_override_mtime_ns = None
        self._scenario_override_rules = []
        self._scenario_override_error = ""
        self._active_manual_override = None
        self._active_manual_override_key = None
        self._load_scenario_overrides(force=True)

    def reset(self, map_name=""):
        self.reference = ReferencePath()
        self.previous_trajectory = None
        self.previous_target_d = 0.0
        self.behavior = "STOP"
        self.map_name = str(map_name or "")
        self.last_debug = {}
        self.avoidance_side = 0
        self.avoidance_obstacle_id = None
        self.avoidance_origin_d = None
        self._last_speed_limit_debug = {}
        self._active_manual_override = None
        self._active_manual_override_key = None

    def _load_scenario_overrides(self, force=False):
        """Load enabled manual-control rules and support live JSON edits."""
        path = os.path.abspath(
            os.path.expanduser(
                str(self.config.scenario_overrides_path or "")
            )
        )
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError as exc:
            if force:
                self._scenario_override_rules = []
                self._scenario_override_error = str(exc)
            return
        if (
            not force
            and mtime_ns == self._scenario_override_mtime_ns
        ):
            return

        try:
            with open(path, "r", encoding="utf-8") as stream:
                document = json.load(stream)
            raw_rules = (
                document.get("scenarios", [])
                if isinstance(document, dict)
                else document
            )
            if not isinstance(raw_rules, list):
                raise ValueError("'scenarios' must be a list")
            rules = []
            for index, raw_rule in enumerate(raw_rules):
                if (
                    not isinstance(raw_rule, dict)
                    or not bool(raw_rule.get("enabled", False))
                ):
                    continue
                target_d = _finite(
                    raw_rule.get("target_d"), float("nan")
                )
                if "target_speed_mps" in raw_rule:
                    target_speed = _finite(
                        raw_rule.get("target_speed_mps"),
                        float("nan"),
                    )
                else:
                    target_speed = (
                        _finite(
                            raw_rule.get("target_speed_kmh"),
                            float("nan"),
                        )
                        / 3.6
                    )
                if (
                    not math.isfinite(target_d)
                    or not math.isfinite(target_speed)
                    or target_speed < 0.0
                ):
                    raise ValueError(
                        "enabled rule {} requires finite target_d and "
                        "non-negative target_speed_mps or "
                        "target_speed_kmh".format(index)
                    )
                s_start = _finite(
                    raw_rule.get("s_start", -float("inf")),
                    -float("inf"),
                )
                s_end = _finite(
                    raw_rule.get("s_end", float("inf")),
                    float("inf"),
                )
                if s_end < s_start:
                    raise ValueError(
                        "rule {} has s_end below s_start".format(index)
                    )
                rules.append(
                    {
                        "index": index,
                        "name": str(
                            raw_rule.get(
                                "name", "manual-rule-{}".format(index)
                            )
                        ),
                        "map": str(
                            raw_rule.get("map", "*") or "*"
                        ),
                        "s_start": s_start,
                        "s_end": s_end,
                        "target_d": target_d,
                        "target_speed_mps": target_speed,
                        # Manual mode is intentionally unconditional per the
                        # operator request: perceived obstacles remain
                        # visible/logged, but cannot affect planning.
                        "ignore_collisions": True,
                    }
                )
            self._scenario_override_rules = rules
            self._scenario_override_mtime_ns = mtime_ns
            self._scenario_override_error = ""
            print(
                "[scenario-override] loaded "
                "path={} enabled_rules={}".format(path, len(rules))
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._scenario_override_rules = []
            self._scenario_override_mtime_ns = mtime_ns
            self._scenario_override_error = str(exc)
            print(
                "[scenario-override][WARN] disabled path={} error={}".format(
                    path, exc
                )
            )

    def _match_scenario_override(self, projection):
        self._load_scenario_overrides()
        map_basename = os.path.basename(str(self.map_name or ""))
        station = float(projection["s"])
        matched = next(
            (
                dict(rule)
                for rule in self._scenario_override_rules
                if (
                    fnmatch.fnmatchcase(map_basename, rule["map"])
                    and rule["s_start"] <= station < rule["s_end"]
                )
            ),
            None,
        )
        key = (
            None
            if matched is None
            else (
                matched["index"],
                matched["name"],
                map_basename,
            )
        )
        if key != self._active_manual_override_key:
            self._active_manual_override_key = key
            if matched is None:
                print("[scenario-override] inactive")
            else:
                print(
                    "[scenario-override][UNSAFE] active "
                    "name={} map={} s=[{:.3f},{:.3f}) "
                    "target_d={:.3f}m target_v={:.3f}m/s "
                    "collision_detection=BYPASSED".format(
                        matched["name"],
                        map_basename,
                        matched["s_start"],
                        matched["s_end"],
                        matched["target_d"],
                        matched["target_speed_mps"],
                    )
                )
        self._active_manual_override = matched
        return matched

    def _ego_values(self, ego, ego_lateral_speed):
        return {
            "x": _finite(getattr(ego, "x", 0.0)),
            "y": _finite(getattr(ego, "y", 0.0)),
            "yaw": _finite(getattr(ego, "theta", 0.0)),
            "speed": max(0.0, _finite(getattr(ego, "speed", 0.0))),
            "lat_speed": _finite(ego_lateral_speed),
            "length": max(3.8, _finite(getattr(ego, "length", 4.6), 4.6)),
            "width": max(1.7, _finite(getattr(ego, "width", 1.9), 1.9)),
        }

    def _obstacle_values(self, obstacle):
        speed_valid = bool(getattr(obstacle, "speed_valid", False))
        speed = max(0.0, _finite(getattr(obstacle, "speed", 0.0)))
        yaw = _finite(getattr(obstacle, "theta", 0.0))
        if speed_valid:
            vx = _finite(getattr(obstacle, "world_vx", speed * math.cos(yaw)))
            vy = _finite(getattr(obstacle, "world_vy", speed * math.sin(yaw)))
        else:
            vx = 0.0
            vy = 0.0
            speed = 0.0
        role = str(getattr(obstacle, "roleType", ""))
        obs_type = int(_finite(getattr(obstacle, "obs_type", 1), 1))
        pedestrian = "PEDESTRIAN" in role or obs_type == 4
        return {
            "id": str(getattr(obstacle, "id", "-1")),
            "x": _finite(getattr(obstacle, "x", 0.0)),
            "y": _finite(getattr(obstacle, "y", 0.0)),
            "yaw": yaw,
            "speed": speed,
            "vx": vx,
            "vy": vy,
            "length": max(0.6, _finite(getattr(obstacle, "length", 4.0), 4.0)),
            "width": max(0.5, _finite(getattr(obstacle, "width", 1.8), 1.8)),
            "pedestrian": pedestrian,
            "obs_type": obs_type,
            "score": max(
                0.0,
                min(1.0, _finite(getattr(obstacle, "score", 0.0))),
            ),
            "speed_valid": speed_valid,
            "track_hits": int(
                _finite(getattr(obstacle, "track_hits", 0), 0)
            ),
            "track_misses": int(
                _finite(getattr(obstacle, "track_misses", 0), 0)
            ),
            "track_predicted": bool(
                getattr(obstacle, "track_predicted", False)
            ),
        }

    def _prepare_obstacles(self, obstacles):
        prepared = []
        for raw in obstacles or []:
            obstacle = self._obstacle_values(raw)
            projection = self.reference.project(obstacle["x"], obstacle["y"])
            if projection is None:
                continue
            obstacle.update({
                "s": projection["s"],
                "d": projection["d"],
                "path_yaw": projection["yaw"],
            })
            obstacle["longitudinal_speed"] = max(
                0.0,
                obstacle["vx"] * math.cos(projection["yaw"])
                + obstacle["vy"] * math.sin(projection["yaw"]),
            )
            prepared.append(obstacle)
        return prepared

    def _planning_obstacles(self, ego, projection, obstacles):
        """Discard detections that cannot affect a forward local trajectory.

        PointPillars occasionally projects objects behind the beginning of the
        global path onto ``s=0``.  Feeding those objects into every candidate's
        clearance cost made the selected lateral side flip from frame to
        frame.  Keep a small rear overlap for safety, but ignore objects that
        are behind the ego or far outside the drivable corridor.
        """
        corridor = max(
            abs(self.config.road_left_bound),
            abs(self.config.road_right_bound),
            abs(self.config.avoidance_half_width),
        )
        cos_yaw = math.cos(ego["yaw"])
        sin_yaw = math.sin(ego["yaw"])
        relevant = []
        for obstacle in obstacles:
            dx = obstacle["x"] - ego["x"]
            dy = obstacle["y"] - ego["y"]
            ego_longitudinal = dx * cos_yaw + dy * sin_yaw
            ego_lateral = -dx * sin_yaw + dy * cos_yaw
            gap = obstacle["s"] - projection["s"]
            # PointPillars can return parts of the ego vehicle as a vehicle
            # centred almost exactly on the current INS pose.  Such a box
            # would make every forward trajectory collide at t=0.
            if (
                abs(ego_longitudinal) < 0.55 * ego["length"]
                and abs(ego_lateral) < 0.60 * ego["width"]
            ):
                continue
            lateral_limit = (
                corridor
                + 0.5 * obstacle["width"]
                + self.config.collision_margin
                + 0.5
            )
            if abs(obstacle["d"]) > lateral_limit:
                continue
            # Retain rear-quarter and adjacent-lane vehicles.  A lane-change
            # trajectory may decelerate while a vehicle behind continues
            # forward, so the old ~3 m rear cutoff could remove the exact
            # vehicle that the ego was about to merge into.
            obstacle_longitudinal_speed = (
                obstacle["vx"] * cos_yaw + obstacle["vy"] * sin_yaw
            )
            rear_limit = abs(self.config.rear_obstacle_distance)
            if obstacle.get("speed_valid", False):
                rear_limit += (
                    max(0.0, obstacle_longitudinal_speed)
                    * max(self.config.horizons)
                )
            else:
                rear_limit = max(
                    rear_limit,
                    self.config.follow_distance + 15.0,
                )
            if (
                ego_longitudinal
                < -rear_limit
            ):
                continue
            if gap > self.config.follow_distance + 15.0:
                continue
            item = dict(obstacle)
            item["ego_longitudinal"] = ego_longitudinal
            item["ego_lateral"] = ego_lateral
            relevant.append(item)
        return relevant

    def _find_lead(
        self, ego_s, ego_d, ego_width, obstacles, ego_speed=0.0
    ):
        del ego_width
        intervention_distance = self._intervention_distance(
            ego_speed
        )
        lead = None
        for obstacle in obstacles:
            gap = obstacle["s"] - ego_s
            is_slow = (
                obstacle["longitudinal_speed"]
                <= self.config.static_obstacle_speed
            )
            # Use the obstacle centre for longitudinal decisions.  The former
            # box-overlap threshold added both half widths and 0.65 m, so a
            # parked vehicle 1.5--2.3 m beside the ego became a "lead" and
            # triggered a full adjacent-lane avoidance manoeuvre.
            ego_lateral = obstacle.get("ego_lateral")
            ego_frame_close = (
                ego_lateral is None
                or abs(float(ego_lateral))
                <= self.config.lead_lateral_tolerance
            )
            frenet_close = (
                abs(obstacle["d"] - ego_d)
                <= self.config.lead_lateral_tolerance + 0.35
            )
            if (
                gap <= 0.0
                or gap > self.config.follow_distance
                or not ego_frame_close
                or not frenet_close
                or (is_slow and gap > intervention_distance)
            ):
                continue
            if lead is None or gap < lead["gap"]:
                lead = dict(obstacle)
                lead["gap"] = gap
        return lead

    def _intervention_distance(self, ego_speed):
        speed = max(0.0, float(ego_speed))
        braking_distance = (
            speed * speed
            / max(2.0 * self.config.max_decel, EPS)
        )
        return max(
            self.config.static_avoidance_trigger_distance,
            2.0 * speed + braking_distance + 5.0,
        )

    def _is_confirmed_static_lead(self, lead):
        return bool(
            lead is not None
            and lead["longitudinal_speed"]
            <= self.config.static_obstacle_speed
            and int(lead.get("track_hits", 0))
            >= self.config.static_avoidance_min_hits
        )

    def _direct_lane_obstacles(
        self, ego, projection, trajectory, obstacles
    ):
        """Return objects whose predicted centres can affect direct driving.

        This is only a relevance filter for the fast global-path trajectory.
        Every retained object is still checked with the full continuous SAT
        envelope. Sampled avoidance trajectories continue to check all nearby
        objects, including vehicles in the adjacent lane.
        """
        if trajectory is None or not obstacles:
            return []
        horizon = min(5.0, max(0.0, float(trajectory.t[-1])))
        intervention_distance = self._intervention_distance(
            ego["speed"]
        )
        relevant = []
        for obstacle in obstacles:
            forward_gap = obstacle["s"] - projection["s"]
            if (
                forward_gap > intervention_distance
                and obstacle["longitudinal_speed"]
                <= self.config.static_obstacle_speed
            ):
                continue
            relative_yaw = _wrap(
                obstacle.get("yaw", 0.0)
                - obstacle.get("path_yaw", projection["yaw"])
            )
            lateral_extent = (
                0.5
                * (
                    obstacle["length"]
                    + self.config.obstacle_length_padding
                )
                * abs(math.sin(relative_yaw))
                + 0.5
                * (
                    obstacle["width"]
                    + self.config.obstacle_width_padding
                )
                * abs(math.cos(relative_yaw))
            )
            corridor_half_width = (
                0.5
                * (
                    ego["width"]
                    + self.config.ego_width_padding
                )
                + lateral_extent
                + self.config.direct_lane_side_clearance
                + abs(projection["d"])
            )
            path_yaw = obstacle.get(
                "path_yaw", projection["yaw"]
            )
            lateral_speed = (
                -obstacle["vx"] * math.sin(path_yaw)
                + obstacle["vy"] * math.cos(path_yaw)
            )
            start_d = float(obstacle["d"])
            end_d = start_d + lateral_speed * horizon
            predicted_abs_d = min(abs(start_d), abs(end_d))
            if start_d * end_d <= 0.0:
                predicted_abs_d = 0.0
            if predicted_abs_d <= corridor_half_width:
                relevant.append(obstacle)
        return relevant

    def _curve_speed_limit(self, ego_s, base_limit, ego_speed):
        if self.config.comfort_mode:
            # Look far enough ahead to enter every visible curve at a speed
            # that is reachable with the ordinary (comfortable) deceleration
            # limit.  The old 55 m preview was much shorter than the roughly
            # 230 m needed to slow from 40 to 14 m/s at 3 m/s^2.
            finite_base = (
                float(base_limit)
                if math.isfinite(base_limit)
                else max(ego_speed, 40.0)
            )
            braking_preview = (
                finite_base * finite_base
                / (2.0 * max(self.config.max_decel, EPS))
            )
            preview = _clip(braking_preview + 35.0, 55.0, 350.0)
            mask = (
                (self.reference.s >= ego_s)
                & (self.reference.s <= ego_s + preview)
            )
            stations = self.reference.s[mask]
            signed_curvature = self.reference.kappa[mask]
            valid = (
                np.isfinite(stations)
                & np.isfinite(signed_curvature)
                & (np.abs(signed_curvature) <= 0.30)
            )
            stations = stations[valid]
            signed_curvature = signed_curvature[valid]
            # ``reference.s`` may be spaced roughly one metre apart.  If the
            # ego lies between two samples, the first retained curve point is
            # in the future and the braking back-propagation incorrectly
            # grants extra speed even though the ego is already in the bend.
            # Insert the interpolated curvature at exactly ego_s so the
            # steady-curve comfort cap is applied without a sampling gap.
            current_curvature = float(
                np.interp(
                    float(ego_s),
                    self.reference.s,
                    self.reference.kappa,
                )
            )
            if (
                math.isfinite(current_curvature)
                and abs(current_curvature) <= 0.30
            ):
                future = stations > float(ego_s) + 1e-6
                stations = np.concatenate(
                    ([float(ego_s)], stations[future])
                )
                signed_curvature = np.concatenate(
                    ([current_curvature], signed_curvature[future])
                )
            if signed_curvature.size == 0:
                return base_limit

            # Smooth only the derivative estimate.  The lateral-acceleration
            # and yaw-rate caps use the original curvature so a real local
            # maximum cannot be averaged away.
            derivative_curvature = signed_curvature
            if signed_curvature.size >= 5:
                window = min(7, signed_curvature.size)
                if window % 2 == 0:
                    window -= 1
                pad = window // 2
                padded = np.pad(
                    signed_curvature, (pad, pad), mode="edge"
                )
                derivative_curvature = np.convolve(
                    padded,
                    np.ones(window, dtype=float) / float(window),
                    mode="valid",
                )
            relative_s = np.maximum(0.0, stations - float(ego_s))
            safe_station = np.maximum.accumulate(
                stations + np.arange(stations.size) * EPS
            )
            if stations.size >= 2:
                curvature_rate = np.abs(
                    np.gradient(
                        derivative_curvature,
                        safe_station,
                    )
                )
            else:
                curvature_rate = np.zeros_like(signed_curvature)

            abs_curvature = np.maximum(
                np.abs(signed_curvature), EPS
            )
            local_caps = np.full(
                signed_curvature.shape,
                finite_base,
                dtype=float,
            )
            local_caps = np.minimum(
                local_caps,
                np.sqrt(
                    self.config.max_lateral_accel
                    / abs_curvature
                ),
            )
            if math.isfinite(self.config.max_yaw_rate):
                local_caps = np.minimum(
                    local_caps,
                    self.config.max_yaw_rate / abs_curvature,
                )
            changing = curvature_rate > EPS
            local_caps[changing] = np.minimum(
                local_caps[changing],
                np.cbrt(
                    self.config.max_lat_jerk
                    / curvature_rate[changing]
                ),
            )

            # A future low curve cap does not need to be applied immediately:
            # back-propagate it with v0^2 = v1^2 + 2*a*distance.
            reachable_now = np.sqrt(
                np.maximum(
                    0.0,
                    local_caps * local_caps
                    + 2.0
                    * self.config.max_decel
                    * relative_s,
                )
            )
            curve_limit = float(np.min(reachable_now))
            return _clip(curve_limit, 1.0, base_limit)

        preview = max(18.0, min(55.0, 12.0 + 2.5 * ego_speed))
        mask = (self.reference.s >= ego_s) & (self.reference.s <= ego_s + preview)
        curvature = np.abs(self.reference.kappa[mask])
        curvature = curvature[np.isfinite(curvature) & (curvature <= 0.30)]
        if curvature.size == 0:
            return base_limit
        effective = max(float(np.percentile(curvature, 85)), 0.8 * float(np.max(curvature)), EPS)
        curve_limit = (
            self.config.curve_speed_factor
            * math.sqrt(
                self.config.max_lateral_accel / effective
            )
        )
        return _clip(curve_limit, 1.0, base_limit)

    def _path_speed_limit(self, ego_s, base_limit, ego_speed):
        """Apply positive speed caps encoded in ``Path.pose.position.z``."""
        if self.reference.speed_limit.size != self.reference.s.size:
            return base_limit
        preview = max(12.0, min(45.0, 8.0 + 2.0 * ego_speed))
        mask = (self.reference.s >= ego_s) & (self.reference.s <= ego_s + preview)
        caps = self.reference.speed_limit[mask]
        caps = caps[np.isfinite(caps) & (caps > 0.05)]
        if caps.size == 0:
            return base_limit
        return min(base_limit, float(np.min(caps)))

    def _rule_target(self, ego, projection, obstacles, map_name):
        resolved_expected_speed = self.config.expected_speed_mps
        if (
            resolved_expected_speed is None
            or not math.isfinite(float(resolved_expected_speed))
            or float(resolved_expected_speed) <= 0.0
        ):
            scene_limit = max(
                0.0,
                _finite(scene_speed_limit_for_map(map_name), 8.0),
            )
            expected_speed_source = "map-category-fallback"
        else:
            scene_limit = float(resolved_expected_speed)
            expected_speed_source = str(
                self.config.expected_speed_source
                or "session-resolver"
            )
        configured_limit = max(0.0, self.config.max_speed)
        override_map_limit = bool(
            self.config.override_map_speed_limit
            and math.isfinite(configured_limit)
        )
        if override_map_limit:
            configured_base_limit = configured_limit
            path_limit = configured_base_limit
        else:
            configured_base_limit = scene_limit
            if math.isfinite(configured_limit):
                configured_base_limit = min(
                    configured_base_limit, configured_limit
                )
            if self.config.respect_path_speed_limit:
                path_limit = self._path_speed_limit(
                    projection["s"],
                    configured_base_limit,
                    ego["speed"],
                )
            else:
                path_limit = configured_base_limit
        curve_limit = self._curve_speed_limit(
            projection["s"], path_limit, ego["speed"]
        )
        target = curve_limit

        remaining = max(0.0, self.reference.length - projection["s"] - self.config.goal_stop_buffer)
        goal_limit = configured_base_limit
        if self.config.stop_at_goal:
            goal_limit = math.sqrt(
                max(
                    0.0,
                    2.0 * self.config.goal_decel * remaining,
                )
            )
            target = min(target, goal_limit)
            if remaining < 0.8:
                target = 0.0

        self._last_speed_limit_debug = {
            "scene_limit_mps": scene_limit,
            "expected_speed_source": expected_speed_source,
            "configured_limit_mps": (
                configured_limit
                if math.isfinite(configured_limit)
                else None
            ),
            "configured_base_limit_mps": configured_base_limit,
            "path_limit_mps": path_limit,
            "curve_limit_mps": curve_limit,
            "goal_limit_mps": goal_limit,
            "override_map_speed_limit": override_map_limit,
            "respect_path_speed_limit": (
                self.config.respect_path_speed_limit
            ),
            "stop_at_goal": self.config.stop_at_goal,
            "goal_decel_mps2": self.config.goal_decel,
        }

        lead = self._find_lead(
            projection["s"],
            projection["d"],
            ego["width"],
            obstacles,
            ego_speed=ego["speed"],
        )
        emergency = False
        behavior = "KEEP_LANE"
        if lead is not None:
            desired_gap = self.config.minimum_gap + self.config.time_headway * ego["speed"]
            closing = max(0.0, ego["speed"] - lead["longitudinal_speed"])
            ttc = lead["gap"] / max(closing, EPS) if closing > 0.05 else float("inf")
            static_lead = self._is_confirmed_static_lead(lead)
            if static_lead:
                # Preserve enough forward motion to execute a lane change.
                # Stop-only candidates remain available if neither side is
                # collision-free.
                target = min(
                    target, self.config.static_avoidance_speed
                )
                behavior = "AVOID"
            else:
                gap_speed = (
                    lead["longitudinal_speed"]
                    + 0.35 * (lead["gap"] - desired_gap)
                )
                target = min(target, max(0.0, gap_speed))
                behavior = "FOLLOW"
            emergency = ttc < self.config.emergency_ttc or lead["gap"] < max(
                2.0, 0.45 * ego["length"]
            )
        return target, lead, behavior, emergency

    @staticmethod
    def _quartic_lon(s0, v0, target_v, horizon):
        # s = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4, a0=s0,
        # a1=v0, a2=0; terminal velocity=target_v, terminal accel=0.
        matrix = np.array(
            [[3.0 * horizon ** 2, 4.0 * horizon ** 3],
             [6.0 * horizon, 12.0 * horizon ** 2]],
            dtype=float,
        )
        rhs = np.array([target_v - v0, 0.0], dtype=float)
        a3, a4 = np.linalg.solve(matrix, rhs)
        return np.array([s0, v0, 0.0, a3, a4], dtype=float)

    @staticmethod
    def _quintic(start, start_rate, start_accel, end, end_rate, end_accel, horizon):
        a0 = start
        a1 = start_rate
        a2 = 0.5 * start_accel
        matrix = np.array(
            [
                [horizon ** 3, horizon ** 4, horizon ** 5],
                [3 * horizon ** 2, 4 * horizon ** 3, 5 * horizon ** 4],
                [6 * horizon, 12 * horizon ** 2, 20 * horizon ** 3],
            ],
            dtype=float,
        )
        rhs = np.array(
            [
                end - (a0 + a1 * horizon + a2 * horizon ** 2),
                end_rate - (a1 + 2 * a2 * horizon),
                end_accel - 2 * a2,
            ],
            dtype=float,
        )
        a3, a4, a5 = np.linalg.solve(matrix, rhs)
        return np.array([a0, a1, a2, a3, a4, a5], dtype=float)

    @staticmethod
    def _poly_eval(coefficients, t, derivative=0):
        coeff = np.asarray(coefficients, dtype=float)
        for _ in range(int(derivative)):
            if coeff.size <= 1:
                return np.zeros_like(t, dtype=float)
            coeff = np.array([index * coeff[index] for index in range(1, coeff.size)])
        result = np.zeros_like(t, dtype=float)
        for coefficient in coeff[::-1]:
            result = result * t + coefficient
        return result

    def _lateral_targets(
        self, projection, lead, static_avoidance=False
    ):
        if static_avoidance:
            low = (
                -abs(self.config.avoidance_half_width)
                + self.config.road_margin
            )
            high = (
                abs(self.config.avoidance_half_width)
                - self.config.road_margin
            )
        else:
            low = self.config.road_right_bound + self.config.road_margin
            high = self.config.road_left_bound - self.config.road_margin
        # At standstill, forcing every candidate to move laterally while its
        # longitudinal station is almost unchanged creates an artificial
        # curvature spike. Include the current offset so the vehicle can first
        # accelerate along a feasible parallel path, then recover toward the
        # centre line on later planning cycles.
        current_d = _clip(projection["d"], low, high)
        values = [
            current_d,
            0.0,
            _clip(self.previous_target_d, low, high),
        ]
        step = self.config.lateral_sample_step
        offset = step
        while offset < max(abs(low), abs(high)) + 1e-4:
            if offset <= high + 1e-4:
                values.append(min(offset, high))
            if -offset >= low - 1e-4:
                values.append(max(-offset, low))
            offset += step
        values.extend([low, high])

        if lead is not None and lead["gap"] < 32.0:
            clearance = 0.5 * lead["width"] + 1.1
            values.extend([
                _clip(lead["d"] + clearance, low, high),
                _clip(lead["d"] - clearance, low, high),
            ])
        return _unique([_clip(value, low, high) for value in values])

    def _speed_targets(self, ego_speed, desired_speed, lead):
        values = [desired_speed, max(0.0, desired_speed - 1.5)]
        if desired_speed > ego_speed + 0.5:
            values.append(min(desired_speed, ego_speed + 1.5))
        if lead is not None:
            values.extend([lead["longitudinal_speed"], 0.0])
        if desired_speed < 0.2:
            values.append(0.0)
        return _unique([max(0.0, value) for value in values])

    def _make_trajectory(
        self,
        lon_coeff,
        lat_coeff,
        horizon,
        lateral_delay=0.0,
    ):
        t = np.arange(0.0, horizon + 0.5 * self.config.sample_dt, self.config.sample_dt)
        s = self._poly_eval(lon_coeff, t, 0)
        speed = self._poly_eval(lon_coeff, t, 1)
        accel = self._poly_eval(lon_coeff, t, 2)
        lateral_t = np.maximum(
            0.0, t - max(0.0, float(lateral_delay))
        )
        d = self._poly_eval(lat_coeff, lateral_t, 0)
        d_speed = self._poly_eval(lat_coeff, lateral_t, 1)
        d_accel = self._poly_eval(lat_coeff, lateral_t, 2)
        d_jerk = self._poly_eval(lat_coeff, lateral_t, 3)
        trajectory = Trajectory(t, s, d, speed, accel, d_speed, d_accel, d_jerk)
        trajectory.x, trajectory.y, trajectory.yaw, trajectory.kappa = (
            self.reference.frenet_to_cartesian(s, d)
        )
        return trajectory

    def _hard_feasibility_reason(
        self, trajectory, static_avoidance=False
    ):
        cfg = self.config
        if not all(
            np.all(np.isfinite(values))
            for values in (
                trajectory.s, trajectory.d, trajectory.speed, trajectory.accel,
                trajectory.d_speed, trajectory.d_accel, trajectory.d_jerk,
                trajectory.x, trajectory.y, trajectory.yaw, trajectory.kappa,
            )
        ):
            return "non_finite"
        if np.min(trajectory.speed) < -0.10:
            return "negative_speed"
        if np.max(trajectory.accel) > cfg.max_accel + 0.15:
            return "max_accel"
        if np.min(trajectory.accel) < -cfg.max_decel - 0.15:
            return "max_decel"
        lon_jerk = np.gradient(trajectory.accel, trajectory.t, edge_order=1)
        if np.max(np.abs(lon_jerk)) > cfg.max_lon_jerk + 0.2:
            return "lon_jerk"
        if np.max(np.abs(trajectory.d_speed)) > cfg.max_lat_speed:
            return "lat_speed"
        if np.max(np.abs(trajectory.d_accel)) > cfg.max_lat_accel:
            return "lat_accel"
        if np.max(np.abs(trajectory.d_jerk)) > cfg.max_lat_jerk:
            return "lat_jerk"
        # If the ego is already outside the nominal corridor, rejecting a
        # trajectory because its first sample is outside makes recovery
        # mathematically impossible.  Permit only monotonic inward recovery:
        # the endpoint must be inside and no sample may move farther out.
        if static_avoidance:
            right_bound = -abs(cfg.avoidance_half_width)
            left_bound = abs(cfg.avoidance_half_width)
        else:
            right_bound = cfg.road_right_bound
            left_bound = cfg.road_left_bound
        start_d = float(trajectory.d[0])
        end_d = float(trajectory.d[-1])
        if start_d > left_bound:
            if (
                end_d > left_bound + EPS
                or np.max(trajectory.d) > start_d + EPS
            ):
                return "left_bound"
        elif np.max(trajectory.d) > left_bound + EPS:
            return "left_bound"
        if start_d < right_bound:
            if (
                end_d < right_bound - EPS
                or np.min(trajectory.d) < start_d - EPS
            ):
                return "right_bound"
        elif np.min(trajectory.d) < right_bound - EPS:
            return "right_bound"
        if np.max(np.abs(trajectory.kappa)) > cfg.max_curvature:
            return "curvature"
        cartesian_lateral_accel = trajectory.speed ** 2 * np.abs(
            trajectory.kappa
        )
        if np.max(cartesian_lateral_accel) > cfg.max_lateral_accel + 0.20:
            return "cartesian_lat_accel"
        return None

    def _hard_feasible(self, trajectory, static_avoidance=False):
        return (
            self._hard_feasibility_reason(
                trajectory, static_avoidance=static_avoidance
            )
            is None
        )

    def _collision_samples(self, trajectory):
        """Interpolate a trajectory densely enough to prevent tunnelling."""
        start_time = float(trajectory.t[0])
        end_time = float(trajectory.t[-1])
        check_dt = max(
            0.02,
            min(0.10, float(self.config.collision_check_dt)),
        )
        times = np.arange(
            start_time,
            end_time + 0.5 * check_dt,
            check_dt,
        )
        if times.size == 0 or times[-1] < end_time - EPS:
            times = np.append(times, end_time)
        else:
            times[-1] = min(times[-1], end_time)
        x = np.interp(times, trajectory.t, trajectory.x)
        y = np.interp(times, trajectory.t, trajectory.y)
        yaw = np.interp(
            times,
            trajectory.t,
            np.unwrap(trajectory.yaw),
        )
        return times, x, y, yaw

    def _collision_free(
        self,
        trajectory,
        ego,
        obstacles,
        avoidance_obstacle_id=None,
    ):
        """Check continuous-time oriented vehicle envelopes using SAT.

        Both vehicle headings participate in the separating-axis test.
        Detector-size padding, tracking uncertainty, and unknown obstacle
        speed are hard-envelope expansions rather than soft cost terms.
        """
        minimum_clearance = float("inf")
        closest_obstacle_id = None
        closest_time = None
        times, ego_x, ego_y, ego_yaw = self._collision_samples(trajectory)
        ego_cos = np.cos(ego_yaw)
        ego_sin = np.sin(ego_yaw)
        ego_long_x = ego_cos
        ego_long_y = ego_sin
        ego_lat_x = -ego_sin
        ego_lat_y = ego_cos
        trajectory.ignored_rear_ids = []

        for obstacle in obstacles:
            if self._is_non_blocking_rear_follower(
                ego, obstacle, trajectory=trajectory
            ):
                trajectory.ignored_rear_ids.append(obstacle.get("id"))
                continue
            obs_x = obstacle["x"] + obstacle["vx"] * times
            obs_y = obstacle["y"] + obstacle["vy"] * times
            dx = obs_x - ego_x
            dy = obs_y - ego_y

            obstacle_yaw = _finite(obstacle.get("yaw", 0.0))
            obs_cos = math.cos(obstacle_yaw)
            obs_sin = math.sin(obstacle_yaw)
            obs_long_x = obs_cos
            obs_long_y = obs_sin
            obs_lat_x = -obs_sin
            obs_lat_y = obs_cos

            precise_static_bypass = bool(
                avoidance_obstacle_id is not None
                and obstacle["speed"]
                <= self.config.static_obstacle_speed
                and obstacle["longitudinal_speed"]
                <= self.config.static_obstacle_speed
                and int(obstacle.get("track_hits", 0))
                >= self.config.static_avoidance_min_hits
            )
            if precise_static_bypass:
                ego_half_length = 0.5 * ego["length"]
                ego_half_width = 0.5 * ego["width"]
                obstacle_length_padding = 0.0
                obstacle_width_padding = 0.0
            else:
                ego_half_length = 0.5 * (
                    ego["length"]
                    + self.config.ego_length_padding
                )
                ego_half_width = 0.5 * (
                    ego["width"]
                    + self.config.ego_width_padding
                )
                obstacle_length_padding = (
                    self.config.obstacle_length_padding
                )
                obstacle_width_padding = (
                    self.config.obstacle_width_padding
                )

            obs_half_length = np.full(
                times.shape,
                0.5
                * (
                    obstacle["length"]
                    + obstacle_length_padding
                ),
                dtype=float,
            )
            if not precise_static_bypass:
                obs_half_length += (
                    self.config.prediction_growth * times
                )
            if not bool(obstacle.get("speed_valid", False)):
                obs_half_length += (
                    self.config.unknown_speed_growth * times
                )
            obs_half_width = (
                0.5
                * (
                    obstacle["width"]
                    + obstacle_width_padding
                )
                + (
                    0.0
                    if precise_static_bypass
                    else self.config.lateral_prediction_growth
                )
                * times
            )

            extra = (
                self.config.static_side_clearance
                if precise_static_bypass
                else self.config.collision_margin
            )
            if obstacle["pedestrian"]:
                extra += self.config.pedestrian_extra_margin
            if bool(obstacle.get("track_predicted", False)):
                extra += 0.25
            margin = np.full(times.shape, extra, dtype=float)

            ratios = []
            axes = (
                (ego_long_x, ego_long_y),
                (ego_lat_x, ego_lat_y),
                (obs_long_x, obs_long_y),
                (obs_lat_x, obs_lat_y),
            )
            for axis_x, axis_y in axes:
                centre_projection = np.abs(dx * axis_x + dy * axis_y)
                ego_radius = (
                    ego_half_length
                    * np.abs(
                        ego_long_x * axis_x + ego_long_y * axis_y
                    )
                    + ego_half_width
                    * np.abs(ego_lat_x * axis_x + ego_lat_y * axis_y)
                )
                obstacle_radius = (
                    obs_half_length
                    * np.abs(obs_long_x * axis_x + obs_long_y * axis_y)
                    + obs_half_width
                    * np.abs(obs_lat_x * axis_x + obs_lat_y * axis_y)
                )
                ratios.append(
                    centre_projection
                    / np.maximum(
                        ego_radius + obstacle_radius + margin,
                        EPS,
                    )
                )

            # For oriented rectangles, one separating axis is sufficient.
            # Therefore max(axis ratios) <= 1 means all four axes overlap.
            separation_ratio = np.max(np.vstack(ratios), axis=0)
            local_index = int(np.argmin(separation_ratio))
            local_clearance = float(separation_ratio[local_index])
            if local_clearance < minimum_clearance:
                minimum_clearance = local_clearance
                closest_obstacle_id = obstacle.get("id")
                closest_time = float(times[local_index])
            if np.any(separation_ratio <= 1.0):
                trajectory.minimum_clearance = minimum_clearance
                trajectory.closest_obstacle_id = closest_obstacle_id
                trajectory.closest_collision_time = closest_time
                return False

        trajectory.minimum_clearance = minimum_clearance
        trajectory.closest_obstacle_id = closest_obstacle_id
        trajectory.closest_collision_time = closest_time
        return True

    def _is_non_blocking_rear_follower(
        self, ego, obstacle, trajectory=None
    ):
        """Return whether replay traffic is following in the ego's lane.

        A replayed rear vehicle cannot yield or brake for the ego.  Treating
        its predicted rear-end impact as an ego planning constraint creates a
        positive feedback loop: the rear car gets closer, every candidate is
        rejected, and the ego brakes harder.  Same-lane rear followers are
        always exempted.  An adjacent-lane rear vehicle is exempted only for a
        candidate which stays in place or moves away from it; a candidate
        merging toward that vehicle must still pass the full collision check.
        """
        if not self.config.non_yielding_replay_traffic:
            return False
        if bool(obstacle.get("pedestrian", False)):
            return False
        ego_yaw = _finite(ego.get("yaw", 0.0))
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        dx = obstacle["x"] - _finite(ego.get("x", 0.0))
        dy = obstacle["y"] - _finite(ego.get("y", 0.0))
        longitudinal = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw
        if longitudinal >= 0.0:
            return False
        if abs(lateral) <= self.config.rear_follow_lateral_tolerance:
            return True
        if trajectory is None or not hasattr(trajectory, "d"):
            return False
        candidate_d = np.asarray(trajectory.d, dtype=float)
        if candidate_d.size == 0 or not np.all(np.isfinite(candidate_d)):
            return False
        obstacle_d = _finite(obstacle.get("d"), float("nan"))
        if not math.isfinite(obstacle_d):
            return False
        start_d = float(candidate_d[0])
        obstacle_side = obstacle_d - start_d
        if abs(obstacle_side) <= self.config.rear_follow_lateral_tolerance:
            return False
        # Positive means that some point of the candidate moves toward the
        # adjacent rear car.  A small tolerance ignores replanning noise but
        # keeps genuine lane changes collision-constrained.
        toward_motion = float(
            np.max(
                math.copysign(1.0, obstacle_side)
                * (candidate_d - start_d)
            )
        )
        return toward_motion <= 0.25

    def _rear_pressure(self, ego, obstacles):
        """Describe the most urgent non-yielding rear follower, if any."""
        most_urgent = None
        for obstacle in obstacles:
            if not self._is_non_blocking_rear_follower(ego, obstacle):
                continue
            distance = max(
                0.0, -float(obstacle.get("ego_longitudinal", 0.0))
            )
            closing_speed = (
                float(obstacle.get("longitudinal_speed", 0.0))
                - ego["speed"]
            )
            pressured = (
                (
                    distance <= self.config.rear_pressure_distance
                    and closing_speed
                    >= self.config.rear_pressure_closing_speed
                )
                or distance <= 4.0
            )
            if not pressured:
                continue
            candidate = {
                "id": obstacle.get("id"),
                "distance": distance,
                "closing_speed": closing_speed,
            }
            if (
                most_urgent is None
                or distance < most_urgent["distance"]
            ):
                most_urgent = candidate
        return most_urgent

    def _trajectory_cost(self, trajectory, desired_speed):
        cfg = self.config
        center_cost = float(np.mean(trajectory.d ** 2))
        terminal_center_cost = float(trajectory.d[-1] ** 2)
        speed_cost = float((trajectory.speed[-1] - desired_speed) ** 2)
        # Compare progress as average speed. Using total distance unfairly
        # rewarded the 6 s horizon over identical 3 s trajectories, making
        # every lateral recovery unnecessarily sluggish.
        progress_cost = -float(
            (trajectory.s[-1] - trajectory.s[0])
            / max(float(trajectory.t[-1] - trajectory.t[0]), EPS)
        )
        lon_jerk = np.gradient(trajectory.accel, trajectory.t, edge_order=1)
        comfort_cost = float(
            np.mean(trajectory.accel ** 2)
            + 0.2 * np.mean(lon_jerk ** 2)
            + np.mean(trajectory.d_accel ** 2)
            + 0.2 * np.mean(trajectory.d_jerk ** 2)
        )
        lateral_change = float((trajectory.d[-1] - self.previous_target_d) ** 2)
        old_side = 1 if self.previous_target_d > 0.30 else (
            -1 if self.previous_target_d < -0.30 else 0
        )
        new_side = 1 if trajectory.d[-1] > 0.30 else (
            -1 if trajectory.d[-1] < -0.30 else 0
        )
        direction_switch = 1.0 if old_side and new_side and old_side != new_side else 0.0
        if math.isfinite(trajectory.minimum_clearance):
            obstacle_cost = math.exp(-1.2 * max(0.0, trajectory.minimum_clearance - 1.0))
        else:
            obstacle_cost = 0.0

        continuity = 0.0
        if self.previous_trajectory is not None and self.previous_trajectory.d.size:
            count = min(trajectory.d.size, self.previous_trajectory.d.size)
            continuity = float(
                np.mean((trajectory.d[:count] - self.previous_trajectory.d[:count]) ** 2)
            )
        components = {
            "center": cfg.w_center * center_cost,
            "terminal_center": cfg.w_terminal_center * terminal_center_cost,
            "obstacle": cfg.w_obstacle * obstacle_cost,
            "continuity": cfg.w_continuity * continuity,
            "comfort": cfg.w_comfort * comfort_cost,
            "speed": cfg.w_speed * speed_cost,
            "progress": cfg.w_progress * progress_cost,
            "lateral_change": cfg.w_lateral_change * lateral_change,
            "direction_switch": cfg.w_direction_switch * direction_switch,
        }
        trajectory.cost_components = components
        return float(sum(components.values()))

    def _stationary_stop_trajectory(self, ego, projection):
        t = np.array([0.0, 0.5, 1.0], dtype=float)
        # Keep a valid tangent for steering while the requested speed is zero.
        # Three identical points would otherwise yield an arbitrary path yaw.
        preview = max(2.0, min(5.0, ego["speed"]))
        s = projection["s"] + np.array([0.0, 0.5 * preview, preview], dtype=float)
        s = np.minimum(s, self.reference.length)
        d = np.full(t.shape, projection["d"], dtype=float)
        zeros = np.zeros_like(t)
        trajectory = Trajectory(t, s, d, zeros, zeros, zeros, zeros, zeros)
        trajectory.x, trajectory.y, trajectory.yaw, trajectory.kappa = (
            self.reference.frenet_to_cartesian(s, d)
        )
        return trajectory

    def _direct_clear_lane_trajectory(
        self,
        ego,
        projection,
        desired_speed,
        target_d=0.0,
        lateral_transition_distance=None,
    ):
        """Build a spatially smooth lane or bypass trajectory.

        Spatial lateral progress remains well-conditioned at standstill,
        unlike a time polynomial whose longitudinal derivative initially
        approaches zero.
        """
        remaining = max(0.0, self.reference.length - projection["s"])
        if self.config.stop_at_goal and remaining < 0.5:
            return None
        if desired_speed <= 0.05:
            return None

        requested_preview = max(
            20.0,
            min(
                60.0,
                18.0
                + 4.0 * desired_speed
                + 6.0 * abs(projection["d"]),
            ),
        )
        preview_distance = (
            min(remaining, requested_preview)
            if self.config.stop_at_goal
            else requested_preview
        )
        count = max(3, int(math.ceil(preview_distance / 0.5)) + 1)
        delta_s = np.linspace(0.0, preview_distance, count)
        stations = projection["s"] + delta_s

        # Recover to d=0 with a spatial fifth-order smoothstep.  It starts
        # from the current pose with zero lateral slope and does not depend on
        # the sampled time-polynomial feasibility checks.
        if lateral_transition_distance is None:
            recovery_distance = max(
                18.0,
                14.0
                + 8.0 * abs(
                    float(target_d) - projection["d"]
                )
                + 1.5 * ego["speed"],
            )
        else:
            recovery_distance = max(
                3.0, float(lateral_transition_distance)
            )
        recovery_distance = min(
            preview_distance, recovery_distance
        )
        lateral_error = projection["d"] - float(target_d)
        # A fifth-order smoothstep has zero slope at its first point. Since
        # this planner replans at 5 Hz, repeatedly starting that curve from
        # the latest ego pose resets recovery before meaningful lateral
        # motion occurs. An exponential spatial response has a non-zero,
        # error-proportional initial slope, so receding-horizon replanning
        # continues the correction instead of cancelling it.
        response_length = max(
            recovery_distance / 3.0,
            abs(lateral_error) / 0.25,
            1.0,
        )
        offsets = (
            float(target_d)
            + lateral_error
            * np.exp(-delta_s / response_length)
        )

        start_speed = max(0.0, ego["speed"])
        target_speed = max(0.0, desired_speed)
        if target_speed >= start_speed:
            # ``--max_accel`` is the actual requested acceleration ceiling.
            # The former hidden 1.2 m/s^2 cap made a configured 5 m/s^2 behave
            # like a comfort-oriented passenger car and allowed replay traffic
            # to catch the ego after every restart.
            acceleration = max(0.05, self.config.max_accel)
            speeds = np.minimum(
                target_speed,
                np.sqrt(
                    np.maximum(
                        0.0,
                        start_speed ** 2
                        + 2.0 * acceleration * delta_s,
                    )
                ),
            )
        else:
            speeds = np.maximum(
                target_speed,
                np.sqrt(
                    np.maximum(
                        target_speed ** 2,
                        start_speed ** 2
                        - 2.0
                        * max(0.05, self.config.max_decel)
                        * delta_s,
                    )
                ),
            )
        speeds[0] = start_speed

        segment_speed = np.maximum(
            0.5 * (speeds[:-1] + speeds[1:]), 0.25
        )
        times = np.concatenate(
            ([0.0], np.cumsum(np.diff(delta_s) / segment_speed))
        )
        accel = np.gradient(speeds, times, edge_order=1)
        d_speed = np.gradient(offsets, times, edge_order=1)
        d_accel = np.gradient(d_speed, times, edge_order=1)
        d_jerk = np.gradient(d_accel, times, edge_order=1)
        trajectory = Trajectory(
            times,
            stations,
            offsets,
            speeds,
            accel,
            d_speed,
            d_accel,
            d_jerk,
        )
        trajectory.x, trajectory.y, trajectory.yaw, trajectory.kappa = (
            self.reference.frenet_to_cartesian(stations, offsets)
        )
        return trajectory

    def _accept_direct_clear_lane(
        self, trajectory, projection, desired_speed
    ):
        self.avoidance_side = 0
        self.avoidance_obstacle_id = None
        self.avoidance_origin_d = None
        manual_override = self._active_manual_override
        behavior = (
            "MANUAL_OVERRIDE"
            if manual_override is not None
            else (
                "RECOVER"
                if abs(projection["d"]) > 0.25
                else "KEEP_LANE"
            )
        )
        trajectory.cost = 0.0
        trajectory.cost_components = {}
        self.previous_trajectory = trajectory
        self.previous_target_d = float(trajectory.d[-1])
        self.behavior = behavior
        self.last_debug.update(
            {
                "reason": "",
                "direct_clear_path": True,
                "blocking_obstacle": False,
                "generated_candidates": 1,
                "accepted_candidates": 1,
                "collision_rejected": 0,
                "hard_rejections": {},
                "lateral_targets": [
                    float(trajectory.d[-1])
                ],
                "speed_targets": [desired_speed],
                "selected_target_d": self.previous_target_d,
                "selected_cost": 0.0,
                "selected_clearance": float(
                    trajectory.minimum_clearance
                ),
                "rear_non_blocking_ids": list(
                    trajectory.ignored_rear_ids
                ),
                "selected_horizon": float(trajectory.t[-1]),
                "selected_end_speed": float(trajectory.speed[-1]),
                "selected_preview_speed": float(
                    trajectory.speed[
                        min(
                            trajectory.speed.size - 1,
                            max(
                                1,
                                int(
                                    np.searchsorted(
                                        trajectory.t,
                                        0.8,
                                        side="left",
                                    )
                                ),
                            ),
                        )
                    ]
                ),
                "selected_max_curvature": float(
                    np.max(np.abs(trajectory.kappa))
                ),
                "selected_cost_components": {},
                "top_candidates": [],
            }
        )
        now_wall = time.time()
        if now_wall - self.last_debug_wall_time >= 0.5:
            self.last_debug_wall_time = now_wall
            print(
                "[rule-plan] behavior={} mode=DIRECT_CLEAR "
                "target_v={:.2f} target_d={:.2f} clearance={:.2f}".format(
                    behavior,
                    desired_speed,
                    self.previous_target_d,
                    trajectory.minimum_clearance,
                )
            )
        return PlanResult(
            trajectory, desired_speed, behavior, False, ""
        )

    def plan(self, ego, obstacles, global_path, map_name="", ego_lateral_speed=0.0, now=None):
        """Return the safest sampled local trajectory for the current frame."""
        del now
        self.last_debug = {}
        if ego is None:
            self.last_debug = {"reason": "ego unavailable"}
            return PlanResult(None, 0.0, "STOP", True, "ego unavailable")
        if not self.reference.update(global_path):
            self.last_debug = {"reason": "global path unavailable"}
            return PlanResult(None, 0.0, "STOP", True, "global path unavailable")

        self.map_name = str(map_name or self.map_name)
        ego_values = self._ego_values(ego, ego_lateral_speed)
        projection = self.reference.project(ego_values["x"], ego_values["y"])
        if projection is None or projection["distance"] > 8.0:
            self.last_debug = {
                "reason": "ego is too far from global path",
                "projection": projection,
            }
            return PlanResult(None, 0.0, "STOP", True, "ego is too far from global path")

        manual_override = self._match_scenario_override(projection)
        detected_obstacle_count = len(obstacles or [])
        prepared_obstacles = (
            []
            if (
                self.config.ignore_obstacles
                or manual_override is not None
            )
            else self._prepare_obstacles(obstacles)
        )
        planning_obstacles = self._planning_obstacles(
            ego_values, projection, prepared_obstacles
        )
        rear_pressure = self._rear_pressure(
            ego_values, planning_obstacles
        )
        desired_speed, lead, behavior, immediate_emergency = self._rule_target(
            ego_values, projection, planning_obstacles, self.map_name
        )
        if manual_override is not None:
            desired_speed = float(
                manual_override["target_speed_mps"]
            )
            behavior = "MANUAL_OVERRIDE"
            immediate_emergency = False
            self._last_speed_limit_debug.update(
                {
                    "manual_override": True,
                    "manual_target_speed_mps": desired_speed,
                    "manual_target_d": float(
                        manual_override["target_d"]
                    ),
                }
            )
        self.last_debug = {
            "projection": dict(projection),
            "heading_error": _wrap(projection["yaw"] - ego_values["yaw"]),
            "ego_lateral_speed": ego_values["lat_speed"],
            "desired_speed": desired_speed,
            "speed_limits": dict(self._last_speed_limit_debug),
            "remaining": max(0.0, self.reference.length - projection["s"]),
            "intervention_distance": self._intervention_distance(
                ego_values["speed"]
            ),
            "detected_obstacle_count": detected_obstacle_count,
            "raw_obstacle_count": len(prepared_obstacles),
            "planning_obstacle_count": len(planning_obstacles),
            "ignore_obstacles": bool(self.config.ignore_obstacles),
            "manual_control_active": (
                manual_override is not None
            ),
            "manual_override_name": (
                None
                if manual_override is None
                else manual_override["name"]
            ),
            "manual_target_d": (
                None
                if manual_override is None
                else float(manual_override["target_d"])
            ),
            "manual_target_speed_mps": (
                None
                if manual_override is None
                else float(
                    manual_override["target_speed_mps"]
                )
            ),
            "manual_s_start": (
                None
                if manual_override is None
                else float(manual_override["s_start"])
            ),
            "manual_s_end": (
                None
                if manual_override is None
                else float(manual_override["s_end"])
            ),
            "manual_collision_bypass": (
                manual_override is not None
            ),
            "scenario_overrides_path": os.path.abspath(
                str(self.config.scenario_overrides_path)
            ),
            "scenario_override_error": (
                self._scenario_override_error
            ),
            "non_yielding_replay_traffic": bool(
                self.config.non_yielding_replay_traffic
            ),
            "rear_non_blocking_ids": [
                item.get("id")
                for item in planning_obstacles
                if self._is_non_blocking_rear_follower(
                    ego_values, item
                )
            ],
            "rear_pressure": (
                dict(rear_pressure)
                if rear_pressure is not None
                else None
            ),
            "lead": dict(lead) if lead is not None else None,
            "obstacles": [
                {
                    "id": item["id"],
                    "s": item["s"],
                    "d": item["d"],
                    "gap": item["s"] - projection["s"],
                    "speed": item["longitudinal_speed"],
                    "pedestrian": item["pedestrian"],
                    "obs_type": item["obs_type"],
                    "score": item["score"],
                    "speed_valid": item["speed_valid"],
                    "track_hits": item["track_hits"],
                    "track_misses": item["track_misses"],
                    "track_predicted": item["track_predicted"],
                    "ego_longitudinal": item.get(
                        "ego_longitudinal"
                    ),
                    "ego_lateral": item.get("ego_lateral"),
                }
                for item in planning_obstacles[:8]
            ],
        }
        if immediate_emergency:
            self.last_debug["reason"] = "lead obstacle TTC/gap"
            trajectory = self._stationary_stop_trajectory(ego_values, projection)
            self.behavior = "EMERGENCY"
            return PlanResult(trajectory, 0.0, self.behavior, True, "lead obstacle TTC/gap")

        static_lead = bool(
            self._is_confirmed_static_lead(lead)
            and lead["gap"] <= 32.0
        )

        # Normal case: if the current lane and the smooth recovery corridor are
        # clear, drive the global path directly.  Sampling is reserved for a
        # path actually blocked by an obstacle.
        direct_target_d = (
            0.0
            if manual_override is None
            else float(manual_override["target_d"])
        )
        direct_trajectory = self._direct_clear_lane_trajectory(
            ego_values,
            projection,
            desired_speed,
            target_d=direct_target_d,
        )
        direct_lane_obstacles = self._direct_lane_obstacles(
            ego_values,
            projection,
            direct_trajectory,
            planning_obstacles,
        )
        active_avoidance_obstacle = next(
            (
                item
                for item in planning_obstacles
                if (
                    self.avoidance_obstacle_id is not None
                    and str(item.get("id", ""))
                    == self.avoidance_obstacle_id
                    and item["s"] - projection["s"]
                    > -0.5
                    * (ego_values["length"] + item["length"])
                )
            ),
            None,
        )
        self.last_debug["direct_lane_obstacle_ids"] = [
            item.get("id") for item in direct_lane_obstacles
        ]
        if (
            lead is None
            and active_avoidance_obstacle is None
            and direct_trajectory is not None
            and self._collision_free(
                direct_trajectory, ego_values, direct_lane_obstacles
            )
        ):
            return self._accept_direct_clear_lane(
                direct_trajectory, projection, desired_speed
            )

        side_bypass_obstacle = None
        if lead is None:
            if (
                active_avoidance_obstacle is not None
                and active_avoidance_obstacle[
                    "longitudinal_speed"
                ] <= self.config.static_obstacle_speed
            ):
                static_direct_obstacles = [
                    active_avoidance_obstacle
                ]
            else:
                static_direct_obstacles = [
                    item
                    for item in direct_lane_obstacles
                    if item["s"] > projection["s"]
                    and item["longitudinal_speed"]
                    <= self.config.static_obstacle_speed
                    and int(item.get("track_hits", 0))
                    >= self.config.static_avoidance_min_hits
                ]
            if static_direct_obstacles:
                side_bypass_obstacle = dict(
                    min(
                        static_direct_obstacles,
                        key=lambda item: item["s"],
                    )
                )
                side_bypass_obstacle["gap"] = (
                    side_bypass_obstacle["s"] - projection["s"]
                )
        side_bypass = side_bypass_obstacle is not None
        wide_avoidance = static_lead or side_bypass
        avoidance_obstacle = (
            lead if static_lead else side_bypass_obstacle
        )
        if wide_avoidance:
            obstacle_id = str(
                avoidance_obstacle.get("id", "")
            )
            if obstacle_id != self.avoidance_obstacle_id:
                self.avoidance_side = 0
                self.avoidance_obstacle_id = obstacle_id
                self.avoidance_origin_d = float(projection["d"])
        else:
            self.avoidance_side = 0
            self.avoidance_obstacle_id = None
            self.avoidance_origin_d = None
        self.last_debug.update(
            {
                "side_bypass": side_bypass,
                "side_bypass_obstacle_id": (
                    side_bypass_obstacle.get("id")
                    if side_bypass_obstacle is not None
                    else None
                ),
            }
        )

        lateral_targets = self._lateral_targets(
            projection,
            avoidance_obstacle,
            static_avoidance=wide_avoidance,
        )
        avoidance_required_offset = None
        spatial_bypass_targets = []
        spatial_transition_distance = None
        if wide_avoidance:
            relative_yaw = _wrap(
                avoidance_obstacle.get(
                    "yaw", projection["yaw"]
                )
                - avoidance_obstacle.get(
                    "path_yaw", projection["yaw"]
                )
            )
            obstacle_lateral_extent = (
                0.5
                * avoidance_obstacle["length"]
                * abs(math.sin(relative_yaw))
                + 0.5
                * avoidance_obstacle["width"]
                * abs(math.cos(relative_yaw))
            )
            avoidance_required_offset = (
                0.5 * ego_values["width"]
                + obstacle_lateral_extent
                + self.config.static_side_clearance
            )
            bypass_bound = (
                abs(self.config.avoidance_half_width)
                - self.config.road_margin
            )
            avoidance_origin_d = float(
                self.avoidance_origin_d
                if self.avoidance_origin_d is not None
                else projection["d"]
            )
            left_edge_target = (
                avoidance_obstacle["d"]
                + avoidance_required_offset
            )
            right_edge_target = (
                avoidance_obstacle["d"]
                - avoidance_required_offset
            )
            left_target = _clip(
                max(
                    0.0,
                    left_edge_target,
                    avoidance_origin_d
                    + self.config.minimum_bypass_shift,
                ),
                -bypass_bound,
                bypass_bound,
            )
            right_target = _clip(
                min(
                    0.0,
                    right_edge_target,
                    avoidance_origin_d
                    - self.config.minimum_bypass_shift,
                ),
                -bypass_bound,
                bypass_bound,
            )
            # The exact edge solution is the most efficient target. Also
            # create a few progressively wider spatial candidates because
            # the ego box rotates during the lateral transition; its front
            # corner can require more clearance than the terminal
            # path-aligned footprint.
            spatial_bypass_targets = []
            for extra_index in range(4):
                extra = (
                    extra_index
                    * self.config.lateral_sample_step
                )
                spatial_bypass_targets.extend(
                    [
                        _clip(
                            left_target + extra,
                            -bypass_bound,
                            bypass_bound,
                        ),
                        _clip(
                            right_target - extra,
                            -bypass_bound,
                            bypass_bound,
                        ),
                    ]
                )
            spatial_bypass_targets = _unique(
                spatial_bypass_targets
            )
            # Finish most of the lateral transition before the two
            # longitudinal box intervals begin to overlap. A long generic
            # 12--18 m recovery can be collision-free at its endpoint but
            # still reach the parked vehicle before creating side clearance.
            spatial_transition_distance = _clip(
                avoidance_obstacle["gap"]
                - 0.5
                * (
                    ego_values["length"]
                    + avoidance_obstacle["length"]
                )
                - self.config.static_side_clearance,
                4.0,
                12.0,
            )
            lateral_targets = _unique(
                list(lateral_targets)
                + spatial_bypass_targets
            )
        speed_targets = self._speed_targets(ego_values["speed"], desired_speed, lead)
        candidates = []
        generated_count = 0
        collision_rejected = 0
        hard_rejections = {}
        if wide_avoidance:
            spatial_speed = min(
                desired_speed,
                self.config.static_avoidance_speed,
            )
            for target_d in spatial_bypass_targets:
                generated_count += 1
                trajectory = self._direct_clear_lane_trajectory(
                    ego_values,
                    projection,
                    spatial_speed,
                    target_d=target_d,
                    lateral_transition_distance=(
                        spatial_transition_distance
                    ),
                )
                hard_reason = (
                    self._hard_feasibility_reason(
                        trajectory,
                        static_avoidance=True,
                    )
                    if trajectory is not None
                    else "spatial_unavailable"
                )
                if hard_reason is not None:
                    hard_rejections[hard_reason] = (
                        hard_rejections.get(hard_reason, 0) + 1
                    )
                    continue
                if not self._collision_free(
                    trajectory,
                    ego_values,
                    planning_obstacles,
                    avoidance_obstacle_id=(
                        avoidance_obstacle.get("id")
                    ),
                ):
                    collision_rejected += 1
                    continue
                trajectory.cost = self._trajectory_cost(
                    trajectory, desired_speed
                )
                candidates.append(trajectory)
        for horizon in self.config.horizons:
            for target_speed in speed_targets:
                lon_coefficients = [
                    self._quartic_lon(
                        projection["s"], ego_values["speed"], target_speed, horizon
                    )
                ]
                if lead is not None and target_speed < 0.1:
                    stop_s = max(
                        projection["s"],
                        lead["s"]
                        - 0.5 * lead["length"]
                        - 0.5 * ego_values["length"]
                        - self.config.minimum_gap,
                    )
                    lon_coefficients.append(
                        self._quintic(
                            projection["s"], ego_values["speed"], 0.0,
                            stop_s, 0.0, 0.0, horizon,
                        )
                    )
                for lon_coeff in lon_coefficients:
                    for target_d in lateral_targets:
                        generated_count += 1
                        lateral_delay = 0.0
                        if (
                            wide_avoidance
                            and ego_values["speed"] < 0.75
                            and abs(
                                float(target_d)
                                - float(projection["d"])
                            ) >= 0.20
                        ):
                            # At standstill a time-polynomial that starts
                            # translating sideways immediately has near-zero
                            # longitudinal motion and therefore an artificial
                            # curvature spike. Move forward briefly, then
                            # begin the smooth bypass within the same plan.
                            lateral_delay = min(0.70, 0.20 * horizon)
                        lateral_horizon = max(
                            1.0, horizon - lateral_delay
                        )
                        lat_coeff = self._quintic(
                            projection["d"], ego_values["lat_speed"], 0.0,
                            target_d, 0.0, 0.0, lateral_horizon,
                        )
                        trajectory = self._make_trajectory(
                            lon_coeff,
                            lat_coeff,
                            horizon,
                            lateral_delay=lateral_delay,
                        )
                        hard_reason = self._hard_feasibility_reason(
                            trajectory,
                            static_avoidance=wide_avoidance,
                        )
                        if hard_reason is not None:
                            hard_rejections[hard_reason] = (
                                hard_rejections.get(hard_reason, 0) + 1
                            )
                            continue
                        if not self._collision_free(
                            trajectory,
                            ego_values,
                            planning_obstacles,
                            avoidance_obstacle_id=(
                                avoidance_obstacle.get("id")
                                if wide_avoidance
                                else None
                            ),
                        ):
                            collision_rejected += 1
                            continue
                        trajectory.cost = self._trajectory_cost(trajectory, desired_speed)
                        candidates.append(trajectory)
        self.last_debug.update(
            {
                "generated_candidates": generated_count,
                "accepted_candidates": len(candidates),
                "collision_rejected": collision_rejected,
                "hard_rejections": hard_rejections,
                "lateral_targets": list(lateral_targets),
                "speed_targets": list(speed_targets),
                "avoidance_required_offset": (
                    avoidance_required_offset
                ),
                "static_side_clearance": (
                    self.config.static_side_clearance
                ),
                "spatial_bypass_targets": list(
                    spatial_bypass_targets
                ),
                "spatial_transition_distance": (
                    spatial_transition_distance
                ),
            }
        )

        if not candidates:
            self.last_debug["reason"] = "no safe sampled trajectory"
            trajectory = self._stationary_stop_trajectory(ego_values, projection)
            self.previous_trajectory = None
            self.behavior = "EMERGENCY"
            return PlanResult(trajectory, 0.0, self.behavior, True, "no safe sampled trajectory")

        preferred_avoidance = []
        if wide_avoidance:
            avoidance_origin_d = float(
                self.avoidance_origin_d
                if self.avoidance_origin_d is not None
                else projection["d"]
            )
            minimum_shift = self.config.minimum_bypass_shift
            required_offset = avoidance_required_offset
            left_candidates = [
                item
                for item in candidates
                if (
                    float(item.d[-1])
                    - avoidance_obstacle["d"]
                    >= required_offset
                    and float(item.d[-1])
                    >= avoidance_origin_d + minimum_shift - EPS
                )
            ]
            right_candidates = [
                item
                for item in candidates
                if (
                    avoidance_obstacle["d"]
                    - float(item.d[-1])
                    >= required_offset
                    and float(item.d[-1])
                    <= avoidance_origin_d - minimum_shift + EPS
                )
            ]
            if self.avoidance_side > 0 and left_candidates:
                preferred_avoidance = left_candidates
            elif self.avoidance_side < 0 and right_candidates:
                preferred_avoidance = right_candidates
            elif left_candidates or right_candidates:
                left_best = min(
                    left_candidates,
                    key=lambda item: item.cost,
                    default=None,
                )
                right_best = min(
                    right_candidates,
                    key=lambda item: item.cost,
                    default=None,
                )
                if right_best is None or (
                    left_best is not None
                    and left_best.cost <= right_best.cost
                ):
                    self.avoidance_side = 1
                    preferred_avoidance = left_candidates
                else:
                    self.avoidance_side = -1
                    preferred_avoidance = right_candidates
            if preferred_avoidance:
                # If a moving bypass exists, do not let a cheaper stationary
                # trajectory win merely because staying near d=0 has lower
                # centre-line cost.
                moving = [
                    item
                    for item in preferred_avoidance
                    if float(item.speed[-1]) > 0.3
                ]
                candidates = moving or preferred_avoidance
            self.last_debug.update(
                {
                    "static_avoidance": wide_avoidance,
                    "wide_avoidance": True,
                    "avoidance_obstacle_id": (
                        avoidance_obstacle.get("id")
                    ),
                    "avoidance_side": self.avoidance_side,
                    "avoidance_required_offset": required_offset,
                    "static_side_clearance": (
                        self.config.static_side_clearance
                    ),
                    "avoidance_origin_d": avoidance_origin_d,
                    "minimum_bypass_shift": minimum_shift,
                    "avoidance_candidates": len(
                        preferred_avoidance
                    ),
                }
            )
        else:
            self.last_debug.update(
                {
                    "static_avoidance": False,
                    "wide_avoidance": False,
                    "avoidance_obstacle_id": None,
                    "avoidance_side": 0,
                    "avoidance_origin_d": None,
                    "minimum_bypass_shift": (
                        self.config.minimum_bypass_shift
                    ),
                    "avoidance_candidates": 0,
                }
            )

        if rear_pressure is not None:
            # Every item in this list has already passed the full continuous
            # collision envelope.  Under rear pressure, prefer the candidate
            # that creates forward separation soonest instead of allowing a
            # long, low-acceleration horizon to win on comfort cost.
            def rear_pressure_key(item):
                preview_index = min(
                    item.speed.size - 1,
                    max(
                        1,
                        int(
                            np.searchsorted(
                                item.t, 0.8, side="left"
                            )
                        ),
                    ),
                )
                return (
                    -float(item.speed[preview_index]),
                    item.cost,
                )

            candidates.sort(key=rear_pressure_key)
            self.last_debug["candidate_selection_mode"] = (
                "REAR_PRESSURE_FASTEST_SAFE"
            )
        elif (
            lead is None
            and ego_values["speed"]
            < self.config.launch_priority_speed
        ):
            # All candidates already satisfy dynamics, road bounds and the
            # full collision envelope.  At launch, prefer prompt forward
            # separation instead of a long comfort-oriented polynomial.
            def launch_key(item):
                preview_index = min(
                    item.speed.size - 1,
                    max(
                        1,
                        int(
                            np.searchsorted(
                                item.t, 0.8, side="left"
                            )
                        ),
                    ),
                )
                return (
                    -float(item.speed[preview_index]),
                    abs(float(item.d[-1])),
                    item.cost,
                )

            candidates.sort(key=launch_key)
            self.last_debug["candidate_selection_mode"] = (
                "LOW_SPEED_FASTEST_SAFE"
            )
        else:
            candidates.sort(key=lambda item: item.cost)
            self.last_debug["candidate_selection_mode"] = "MIN_COST"
        selected = candidates[0]
        self.previous_trajectory = selected
        self.previous_target_d = float(selected.d[-1])
        self.last_debug.update(
            {
                "reason": "",
                "selected_target_d": self.previous_target_d,
                "selected_cost": float(selected.cost),
                "selected_clearance": float(selected.minimum_clearance),
                "selected_closest_obstacle_id": selected.closest_obstacle_id,
                "selected_closest_time": selected.closest_collision_time,
                "rear_non_blocking_ids": list(
                    selected.ignored_rear_ids
                ),
                "selected_horizon": float(selected.t[-1]),
                "selected_end_speed": float(selected.speed[-1]),
                "selected_preview_speed": float(
                    selected.speed[
                        min(
                            selected.speed.size - 1,
                            max(
                                1,
                                int(
                                    np.searchsorted(
                                        selected.t,
                                        0.8,
                                        side="left",
                                    )
                                ),
                            ),
                        )
                    ]
                ),
                "selected_max_curvature": float(
                    np.max(np.abs(selected.kappa))
                ),
                "selected_cost_components": dict(selected.cost_components),
                "top_candidates": [
                    {
                        "rank": index + 1,
                        "target_d": float(item.d[-1]),
                        "horizon": float(item.t[-1]),
                        "end_speed": float(item.speed[-1]),
                        "cost": float(item.cost),
                        "clearance": float(item.minimum_clearance),
                        "max_curvature": float(np.max(np.abs(item.kappa))),
                    }
                    for index, item in enumerate(candidates[:5])
                ],
            }
        )
        blocking_obstacle = lead is not None or collision_rejected > 0
        self.last_debug["blocking_obstacle"] = blocking_obstacle
        actual_lateral_shift = (
            self.previous_target_d - float(projection["d"])
        )
        self.last_debug["selected_lateral_shift"] = (
            actual_lateral_shift
        )
        if wide_avoidance and preferred_avoidance:
            behavior = (
                "AVOID_LEFT"
                if self.avoidance_side > 0
                else "AVOID_RIGHT"
            )
        elif blocking_obstacle and abs(actual_lateral_shift) > 0.30:
            behavior = (
                "AVOID_LEFT"
                if actual_lateral_shift > 0.0
                else "AVOID_RIGHT"
            )
        elif abs(projection["d"]) > 0.25 or abs(self.previous_target_d) > 0.25:
            behavior = "RECOVER"
        elif lead is not None:
            behavior = "FOLLOW"
        else:
            behavior = "KEEP_LANE"
        self.behavior = behavior

        now_wall = time.time()
        if now_wall - self.last_debug_wall_time >= 0.5:
            self.last_debug_wall_time = now_wall
            print(
                "[rule-plan] behavior={} candidates={} target_v={:.2f} "
                "target_d={:.2f} cost={:.2f} clearance={:.2f}".format(
                    behavior, len(candidates), desired_speed, self.previous_target_d,
                    selected.cost, selected.minimum_clearance,
                )
            )
        return PlanResult(selected, desired_speed, behavior, False, "")


class StableController(object):
    """Stable trajectory tracker without LQR or a C++ control node."""

    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.speed_integral = 0.0
        self.last_speed_error = 0.0
        self.last_acc = 0.0
        self.last_steer = 0.0
        self.filtered_steer = 0.0
        self.last_abs_path_offset = None
        self.path_divergence_count = 0
        self.last_debug = {}

    def reset(self):
        self.speed_integral = 0.0
        self.last_speed_error = 0.0
        self.last_acc = 0.0
        self.last_steer = 0.0
        self.filtered_steer = 0.0
        self.last_abs_path_offset = None
        self.path_divergence_count = 0
        self.last_debug = {}

    def _steering_limit(self, speed, lateral_accel_limit=None):
        cfg = self.config
        if speed < 1.0:
            return cfg.max_steering_wheel_deg
        if lateral_accel_limit is None:
            lateral_accel_limit = cfg.max_lateral_accel
        front_limit = math.atan(
            max(0.05, float(lateral_accel_limit))
            * cfg.controller_wheelbase
            / max(speed * speed, EPS)
        )
        return min(
            cfg.max_steering_wheel_deg,
            abs(math.degrees(front_limit) * cfg.steering_ratio),
        )

    def _trajectory_comfort_speed_limit(
        self, trajectory, nearest, base_limit
    ):
        """Cap speed for the selected local path, including avoidance bends."""
        if (
            not self.config.comfort_mode
            or trajectory.kappa.size < 1
            or trajectory.x.size != trajectory.kappa.size
        ):
            return base_limit
        start = max(0, min(int(nearest), trajectory.kappa.size - 1))
        curvature = np.asarray(
            trajectory.kappa[start:], dtype=float
        )
        xs = np.asarray(trajectory.x[start:], dtype=float)
        ys = np.asarray(trajectory.y[start:], dtype=float)
        if curvature.size == 0:
            return base_limit
        distance = np.concatenate(
            (
                [0.0],
                np.cumsum(np.hypot(np.diff(xs), np.diff(ys))),
            )
        )
        finite_base = (
            float(base_limit)
            if math.isfinite(base_limit)
            else 40.0
        )
        abs_curvature = np.maximum(np.abs(curvature), EPS)
        caps = np.minimum(
            np.full(curvature.shape, finite_base, dtype=float),
            np.sqrt(
                self.config.max_lateral_accel / abs_curvature
            ),
        )
        if math.isfinite(self.config.max_yaw_rate):
            caps = np.minimum(
                caps,
                self.config.max_yaw_rate / abs_curvature,
            )
        if curvature.size >= 2 and distance[-1] > EPS:
            derivative_curvature = curvature
            if curvature.size >= 5:
                window = min(7, curvature.size)
                if window % 2 == 0:
                    window -= 1
                pad = window // 2
                derivative_curvature = np.convolve(
                    np.pad(curvature, (pad, pad), mode="edge"),
                    np.ones(window, dtype=float) / float(window),
                    mode="valid",
                )
            safe_distance = np.maximum.accumulate(
                distance + np.arange(distance.size) * EPS
            )
            curvature_rate = np.abs(
                np.gradient(derivative_curvature, safe_distance)
            )
            changing = curvature_rate > EPS
            caps[changing] = np.minimum(
                caps[changing],
                np.cbrt(
                    self.config.max_lat_jerk
                    / curvature_rate[changing]
                ),
            )
        reachable_now = np.sqrt(
            np.maximum(
                0.0,
                caps * caps
                + 2.0 * self.config.max_decel * distance,
            )
        )
        return min(base_limit, float(np.min(reachable_now)))

    @staticmethod
    def _closest_index(trajectory, x, y):
        distance2 = (trajectory.x - x) ** 2 + (trajectory.y - y) ** 2
        return int(np.argmin(distance2))

    def _target_index(self, trajectory, nearest, lookahead):
        index = int(nearest)
        distance = 0.0
        while index + 1 < trajectory.x.size and distance < lookahead:
            distance += math.hypot(
                trajectory.x[index + 1] - trajectory.x[index],
                trajectory.y[index + 1] - trajectory.y[index],
            )
            index += 1
        return index

    def _lateral_control(
        self,
        ego,
        trajectory,
        dt,
        steering_feedback=None,
        path_lateral_offset=None,
        path_reference_yaw=None,
        path_reference_curvature=None,
        centerline_feedback=False,
    ):
        cfg = self.config
        x = _finite(getattr(ego, "x", 0.0))
        y = _finite(getattr(ego, "y", 0.0))
        yaw = _finite(getattr(ego, "theta", 0.0))
        speed = max(0.0, _finite(getattr(ego, "speed", 0.0)))
        nearest = self._closest_index(trajectory, x, y)
        if cfg.comfort_mode:
            lookahead = _clip(5.0 + speed, 5.0, 35.0)
        else:
            lookahead = _clip(3.0 + 0.65 * speed, 3.0, 14.0)
        target = self._target_index(trajectory, nearest, lookahead)

        nearest_x = float(trajectory.x[nearest])
        nearest_y = float(trajectory.y[nearest])
        ref_x = float(trajectory.x[target])
        ref_y = float(trajectory.y[target])
        # Heading and cross-track feedback belong to the closest path point.
        # Using the look-ahead point for both injected the lane-change preview
        # angle into the immediate heading error and caused +/-42 degree
        # steering reversals.
        ref_yaw = float(trajectory.yaw[nearest])
        ref_kappa = float(trajectory.kappa[target])
        local_heading_error = _wrap(ref_yaw - yaw)
        local_lateral_error = (
            -math.sin(ref_yaw) * (x - nearest_x)
            + math.cos(ref_yaw) * (y - nearest_y)
        )

        front_ff = math.atan(cfg.controller_wheelbase * ref_kappa)
        stanley_gain = 0.75 if speed < 8.0 else 0.45
        front_stanley = (
            front_ff
            + 0.70 * local_heading_error
            - math.atan2(
                stanley_gain * local_lateral_error, speed + 1.5
            )
        )
        centerline_front_feedback = 0.0
        global_offset = _finite(
            path_lateral_offset, float("nan")
        )
        if (
            centerline_feedback
            and math.isfinite(global_offset)
            and cfg.centerline_feedback_gain > 0.0
        ):
            centerline_front_feedback = -math.atan2(
                cfg.centerline_feedback_gain * global_offset,
                speed + 2.5,
            )
            front_stanley += centerline_front_feedback
        alpha = _wrap(math.atan2(ref_y - y, ref_x - x) - yaw)
        front_pure_pursuit = math.atan2(
            2.0 * cfg.controller_wheelbase * math.sin(alpha), max(lookahead, 0.5)
        )
        front_angle = 0.55 * front_stanley + 0.45 * front_pure_pursuit
        global_reference_yaw = _finite(
            path_reference_yaw, float("nan")
        )
        global_reference_curvature = _finite(
            path_reference_curvature, float("nan")
        )
        centerline_control_active = bool(
            centerline_feedback
            and speed >= 2.0
            and math.isfinite(global_offset)
            and math.isfinite(global_reference_yaw)
            and math.isfinite(global_reference_curvature)
        )
        global_heading_error = float("nan")
        estimated_path_lateral_speed = float("nan")
        centerline_lat_accel_ff = float("nan")
        centerline_lat_accel_feedback = float("nan")
        centerline_lat_accel_command = float("nan")
        active_lateral_accel_limit = max(
            0.05, cfg.max_lateral_accel
        )
        curve_authority_active = False
        tracking_recovery_active = False
        heading_error = local_heading_error
        lateral_error = local_lateral_error
        if centerline_control_active:
            # d is positive to the left of the route.  With heading_error
            # defined as path_yaw - ego_yaw, d_dot is -v*sin(error).
            global_heading_error = _wrap(
                global_reference_yaw - yaw
            )
            estimated_path_lateral_speed = (
                -speed * math.sin(global_heading_error)
            )
            omega = cfg.centerline_natural_frequency
            damping = cfg.centerline_damping_ratio
            centerline_lat_accel_ff = (
                speed * speed * global_reference_curvature
            )
            centerline_lat_accel_feedback = (
                -cfg.centerline_feedback_gain
                * omega
                * omega
                * global_offset
                - 2.0
                * damping
                * omega
                * estimated_path_lateral_speed
            )
            lateral_accel_limit = max(
                0.05, cfg.max_lateral_accel
            )
            if math.isfinite(cfg.max_yaw_rate):
                lateral_accel_limit = min(
                    lateral_accel_limit,
                    max(0.05, speed * cfg.max_yaw_rate),
                )
            combined_lateral_accel = (
                centerline_lat_accel_ff
                + centerline_lat_accel_feedback
            )
            curve_authority_active = bool(
                cfg.comfort_mode
                and abs(centerline_lat_accel_ff)
                > lateral_accel_limit + 0.02
            )
            tracking_recovery_active = bool(
                cfg.comfort_mode
                and (
                    abs(global_offset) > 0.35
                    or abs(global_heading_error)
                    > math.radians(4.0)
                )
            )
            if curve_authority_active:
                # At least supply the road-curvature feedforward.  Otherwise
                # an overspeeding ego is guaranteed to run wide even with
                # zero tracking error.
                lateral_accel_limit = max(
                    lateral_accel_limit,
                    min(
                        cfg.max_tracking_lateral_accel,
                        abs(centerline_lat_accel_ff),
                    ),
                )
            if tracking_recovery_active:
                # Once d/heading starts diverging, temporarily allow the
                # complete damped feedback command.  The cap remains far
                # below the physical steering limit and drops back to the
                # comfort threshold as soon as alignment is recovered.
                lateral_accel_limit = max(
                    lateral_accel_limit,
                    min(
                        cfg.max_tracking_lateral_accel,
                        abs(combined_lateral_accel),
                    ),
                )
            if math.isfinite(cfg.max_yaw_rate):
                lateral_accel_limit = min(
                    lateral_accel_limit,
                    max(0.05, speed * cfg.max_yaw_rate),
                )
            active_lateral_accel_limit = lateral_accel_limit
            centerline_lat_accel_command = _clip(
                combined_lateral_accel,
                -lateral_accel_limit,
                lateral_accel_limit,
            )
            front_angle = math.atan2(
                centerline_lat_accel_command
                * cfg.controller_wheelbase,
                max(speed * speed, EPS),
            )
            centerline_front_feedback = (
                front_angle
                - math.atan(
                    cfg.controller_wheelbase
                    * global_reference_curvature
                )
            )
            heading_error = global_heading_error
            lateral_error = global_offset
        model_front_angle_deg = math.degrees(front_angle)
        raw_wheel = (
            cfg.steering_command_sign
            * model_front_angle_deg
            * cfg.steering_ratio
        )

        # A short filter removes sampling noise without retaining the wrong
        # steering sign for more than a metre when the heading error reverses.
        alpha_filter = _clip(dt / (0.08 + dt), 0.08, 1.0)
        self.filtered_steer += alpha_filter * (raw_wheel - self.filtered_steer)
        rate = (
            cfg.steering_rate_low
            if speed < 3.0
            else (
                cfg.steering_rate_mid
                if speed < 8.0
                else cfg.steering_rate_high
            )
        )
        if (
            cfg.comfort_mode
            and speed >= 1.0
            and not tracking_recovery_active
        ):
            # da_lat/dt ~= v^2/L * d(delta)/dt.  Limit steering slew so a
            # command transition itself cannot exceed lateral jerk 1 m/s^3.
            comfort_rate = math.degrees(
                cfg.max_lat_jerk
                * cfg.controller_wheelbase
                * cfg.steering_ratio
                / max(speed * speed, EPS)
            )
            rate = min(rate, comfort_rate)
        rate_limited = _clip(
            self.filtered_steer,
            self.last_steer - rate * dt,
            self.last_steer + rate * dt,
        )
        limit = self._steering_limit(
            speed,
            lateral_accel_limit=active_lateral_accel_limit,
        )
        steer = _clip(rate_limited, -limit, limit)
        previous_steer = self.last_steer
        steer_rate = (steer - previous_steer) / max(dt, EPS)
        commanded_front_angle = math.radians(
            cfg.steering_command_sign
            * steer
            / max(cfg.steering_ratio, EPS)
        )
        commanded_curvature = (
            math.tan(commanded_front_angle)
            / max(cfg.controller_wheelbase, EPS)
        )
        estimated_yaw_rate = speed * commanded_curvature
        estimated_lateral_accel = (
            speed * speed * commanded_curvature
        )
        estimated_lateral_jerk = (
            speed
            * speed
            * (
                1.0 / max(math.cos(commanded_front_angle) ** 2, EPS)
            )
            / max(
                cfg.controller_wheelbase
                * cfg.steering_ratio,
                EPS,
            )
            * math.radians(steer_rate)
        )
        self.last_steer = steer
        steering_tracking_error = (
            None
            if steering_feedback is None
            else steer - float(steering_feedback)
        )
        self.last_debug.update(
            {
                "nearest_index": nearest,
                "target_index": target,
                "lookahead": lookahead,
                "reference_yaw": ref_yaw,
                "reference_curvature": ref_kappa,
                "heading_error": heading_error,
                "lateral_error": lateral_error,
                "local_heading_error": local_heading_error,
                "local_lateral_error": local_lateral_error,
                "front_feedforward": front_ff,
                "front_stanley": front_stanley,
                "centerline_front_feedback": (
                    centerline_front_feedback
                ),
                "front_pure_pursuit": front_pure_pursuit,
                "centerline_control_active": (
                    centerline_control_active
                ),
                "global_reference_yaw": (
                    global_reference_yaw
                ),
                "global_reference_curvature": (
                    global_reference_curvature
                ),
                "global_heading_error": global_heading_error,
                "estimated_path_lateral_speed": (
                    estimated_path_lateral_speed
                ),
                "centerline_lat_accel_ff": (
                    centerline_lat_accel_ff
                ),
                "centerline_lat_accel_feedback": (
                    centerline_lat_accel_feedback
                ),
                "centerline_lat_accel_command": (
                    centerline_lat_accel_command
                ),
                "active_lateral_accel_limit": (
                    active_lateral_accel_limit
                ),
                "curve_authority_active": (
                    curve_authority_active
                ),
                "tracking_recovery_active": (
                    tracking_recovery_active
                ),
                "model_front_angle_deg": model_front_angle_deg,
                "steering_command_sign": cfg.steering_command_sign,
                "raw_steer": raw_wheel,
                "filtered_steer": self.filtered_steer,
                "steering_feedback": (
                    None if steering_feedback is None else float(steering_feedback)
                ),
                "steering_tracking_error": steering_tracking_error,
                "steering_limit": limit,
                "steering_rate_limit": rate,
                "steering_rate": steer_rate,
                "commanded_curvature": commanded_curvature,
                "estimated_yaw_rate": estimated_yaw_rate,
                "estimated_lateral_accel": (
                    estimated_lateral_accel
                ),
                "estimated_lateral_jerk": (
                    estimated_lateral_jerk
                ),
                "steer": steer,
            }
        )
        return steer

    def _longitudinal_control(
        self,
        ego_speed,
        target_speed,
        dt,
        emergency,
        planned_accel=0.0,
    ):
        speed_error = target_speed - ego_speed
        stationary_stale_brake_cleared = False
        if emergency:
            self.speed_integral = 0.0
            desired_acc = -self.config.max_decel if ego_speed > 1.0 else -2.0
        else:
            error = speed_error
            self.speed_integral = _clip(self.speed_integral + error * dt, -6.0, 6.0)
            # The detector makes the control-loop period vary between roughly
            # 20 and 200 ms. A raw derivative of the changing preview-speed
            # target produced full braking even when target_speed > ego_speed.
            # A bounded PI law is monotonic with the speed error and is much
            # less sensitive to replanning jitter.
            desired_acc = (
                _finite(planned_accel, 0.0)
                + 0.62 * error
                + 0.06 * self.speed_integral
            )
            desired_acc = _clip(
                desired_acc,
                -max(0.0, self.config.max_decel),
                max(0.0, self.config.max_accel),
            )
            self.last_speed_error = error
            # Clear stale braking only after the physical vehicle is already
            # stationary. While moving, a transition from braking to driving
            # must pass continuously through zero acceleration; resetting
            # ``last_acc`` there bypasses the configured jerk limit.
            if (
                ego_speed < 0.08
                and target_speed > 0.05
                and self.last_acc < 0.0
            ):
                self.last_acc = 0.0
                stationary_stale_brake_cleared = True

        # Acceleration command jerk limiting is bypassed only for emergency
        # braking; normal following and curve entry remain deliberately smooth.
        acceleration_before_limit = self.last_acc
        if emergency:
            acc = desired_acc
        else:
            jerk = max(0.5, self.config.max_lon_jerk)
            acc = _clip(
                desired_acc,
                self.last_acc - jerk * dt,
                self.last_acc + jerk * dt,
            )

        # A kinematic simulator integrates signed acceleration directly. Limit
        # one-cycle braking to 80% of the current speed so it cannot cross zero
        # and turn into an unintended reverse command before the next update.
        if acc < 0.0:
            non_reversing_floor = -0.8 * ego_speed / max(dt, 1e-3)
            acc = max(acc, non_reversing_floor)
        self.last_acc = acc
        actual_accel_command_jerk = (
            acc - acceleration_before_limit
        ) / max(dt, 1e-3)
        self.last_debug.update(
            {
                "ego_speed": ego_speed,
                "target_speed": target_speed,
                "speed_error": speed_error,
                "speed_integral": self.speed_integral,
                "desired_acc": desired_acc,
                "planned_accel": _finite(planned_accel, 0.0),
                "accel_command_jerk": (
                    None
                    if emergency
                    else max(0.5, self.config.max_lon_jerk)
                ),
                "actual_accel_command_jerk": (
                    actual_accel_command_jerk
                ),
                "stationary_stale_brake_cleared": (
                    stationary_stale_brake_cleared
                ),
                "acc": acc,
                "emergency": bool(emergency),
            }
        )
        return acc

    def _centerline_safety_guard(
        self, path_lateral_offset, behavior, ego_speed
    ):
        """Stop when normal lane keeping persistently moves away from d=0."""
        offset = _finite(path_lateral_offset, float("nan"))
        active = str(behavior) in (
            "KEEP_LANE",
            "RECOVER",
            "FOLLOW",
            "MANUAL_OVERRIDE",
        )
        if not math.isfinite(offset):
            self.last_debug.update(
                {
                    "path_lateral_offset": float("nan"),
                    "path_divergence_count": self.path_divergence_count,
                    "centerline_safety_stop": False,
                }
            )
            return False

        abs_offset = abs(offset)
        previous = self.last_abs_path_offset
        offset_change = (
            0.0 if previous is None else abs_offset - previous
        )
        moving_outward = previous is not None and offset_change > 0.02
        if not active or ego_speed < 0.15:
            # The guard is intended to arrest a worsening motion, not latch
            # the vehicle forever after it has stopped. A clear direct plan
            # must be able to start a fresh recovery attempt.
            self.path_divergence_count = 0
        elif previous is not None:
            if abs_offset >= 0.65 and moving_outward:
                self.path_divergence_count += 1
            elif offset_change < -0.015:
                self.path_divergence_count = max(
                    0, self.path_divergence_count - 2
                )
            # An unchanged projection is normally the same 5 Hz plan being
            # tracked by faster control frames. Do not decay the evidence.

        self.last_abs_path_offset = abs_offset
        safety_stop = bool(
            self.config.centerline_safety_stop_enabled
            and active
            and ego_speed >= 0.15
            and (
                (abs_offset >= 1.20 and moving_outward)
                or (
                    abs_offset >= 0.75
                    and self.path_divergence_count >= 2
                )
            )
        )
        self.last_debug.update(
            {
                "path_lateral_offset": offset,
                "path_offset_change": offset_change,
                "path_divergence_count": self.path_divergence_count,
                "centerline_safety_stop": safety_stop,
            }
        )
        return safety_stop

    def control(
        self,
        ego,
        plan_result,
        dt,
        steering_feedback=None,
        path_lateral_offset=None,
        path_reference_yaw=None,
        path_reference_curvature=None,
    ):
        self.last_debug = {}
        dt = _clip(_finite(dt, 0.05), 0.01, 0.25)
        self.last_debug["dt"] = dt
        if ego is None or plan_result is None:
            self.last_debug["reason"] = "ego or plan unavailable"
            return ControlOutput(0.0, 0.0, 0.0)
        ego_speed = max(0.0, _finite(getattr(ego, "speed", 0.0)))
        trajectory = plan_result.trajectory
        if trajectory is None or trajectory.x.size < 2:
            if ego_speed < 0.08:
                self.last_acc = 0.0
                self.speed_integral = 0.0
                self.last_debug.update(
                    {
                        "reason": "stationary hold without trajectory",
                        "ego_speed": ego_speed,
                        "target_speed": 0.0,
                        "acc": 0.0,
                    }
                )
                return ControlOutput(0.0, 0.0, self.last_steer)
            acc = self._longitudinal_control(ego_speed, 0.0, dt, True)
            self.last_debug["reason"] = "emergency stop without trajectory"
            return ControlOutput(acc, 0.0, self.last_steer)

        target_speed = max(0.0, plan_result.target_speed)
        planned_accel = 0.0
        if trajectory.speed.size:
            preview_index = min(
                trajectory.speed.size - 1,
                max(1, int(np.searchsorted(trajectory.t, 0.8, side="left"))),
            )
            target_speed = min(target_speed, max(0.0, float(trajectory.speed[preview_index])))
            if trajectory.accel.size:
                planned_accel = float(
                    trajectory.accel[
                        min(preview_index, trajectory.accel.size - 1)
                    ]
                )
        nearest = self._closest_index(
            trajectory,
            _finite(getattr(ego, "x", 0.0)),
            _finite(getattr(ego, "y", 0.0)),
        )
        behavior = str(getattr(plan_result, "behavior", ""))
        centerline_tracking = behavior in (
            "KEEP_LANE",
            "RECOVER",
            "FOLLOW",
            "MANUAL_OVERRIDE",
        )
        # RECOVER is rebuilt from the current Frenet state every plan cycle.
        # Its temporary S bend is a control reference, not road geometry.
        # Applying a comfort curvature limit to that bend caused unexplained
        # braking on a straight global route.  Global-route curve speed is
        # already applied by RuleBasedPlanner._curve_speed_limit().
        trajectory_comfort_cap_applied = not centerline_tracking
        if trajectory_comfort_cap_applied:
            trajectory_comfort_speed_cap = (
                self._trajectory_comfort_speed_limit(
                    trajectory, nearest, target_speed
                )
            )
        else:
            trajectory_comfort_speed_cap = target_speed
        target_speed = min(
            target_speed, trajectory_comfort_speed_cap
        )
        steer = self._lateral_control(
            ego,
            trajectory,
            dt,
            steering_feedback=steering_feedback,
            path_lateral_offset=path_lateral_offset,
            path_reference_yaw=path_reference_yaw,
            path_reference_curvature=path_reference_curvature,
            centerline_feedback=centerline_tracking,
        )
        # Do not continue accelerating while the vehicle is no longer aligned
        # with the selected trajectory.  This is the last-resort stability
        # guard that was missing from the original controller.
        abs_heading_error = abs(float(self.last_debug.get("heading_error", 0.0)))
        abs_lateral_error = abs(float(self.last_debug.get("lateral_error", 0.0)))
        alignment_speed_cap = float("inf")
        if self.config.strict_alignment_speed_guard:
            if abs_heading_error >= math.radians(25.0):
                alignment_speed_cap = 0.0
            elif abs_heading_error >= math.radians(18.0):
                alignment_speed_cap = 0.8
            elif abs_heading_error >= math.radians(12.0):
                alignment_speed_cap = 1.5
            elif abs_heading_error >= math.radians(8.0):
                alignment_speed_cap = 3.0
            if abs_lateral_error > 1.5:
                alignment_speed_cap = min(alignment_speed_cap, 1.0)
            elif abs_lateral_error > 0.9:
                alignment_speed_cap = min(alignment_speed_cap, 2.0)
        else:
            # DriveSim's kinematic chassis can correct moderate alignment
            # error without crawling.  Retain a progressive guard only for
            # genuinely large errors; the collision checker remains active.
            if abs_heading_error >= math.radians(32.0):
                alignment_speed_cap = 1.5
            elif abs_heading_error >= math.radians(24.0):
                alignment_speed_cap = 3.0
            elif abs_heading_error >= math.radians(18.0):
                alignment_speed_cap = 5.0
            elif abs_heading_error >= math.radians(14.0):
                alignment_speed_cap = 7.0
            if abs_lateral_error > 2.5:
                alignment_speed_cap = min(alignment_speed_cap, 3.0)
            elif abs_lateral_error > 1.8:
                alignment_speed_cap = min(alignment_speed_cap, 5.0)
            elif abs_lateral_error > 1.2:
                alignment_speed_cap = min(alignment_speed_cap, 7.0)
        target_speed = min(target_speed, alignment_speed_cap)
        if target_speed < ego_speed or math.isfinite(alignment_speed_cap):
            planned_accel = min(0.0, planned_accel)
        centerline_safety_stop = self._centerline_safety_guard(
            path_lateral_offset,
            getattr(plan_result, "behavior", ""),
            ego_speed,
        )
        if centerline_safety_stop:
            target_speed = 0.0
        acc = self._longitudinal_control(
            ego_speed,
            target_speed,
            dt,
            plan_result.emergency or centerline_safety_stop,
            planned_accel=planned_accel,
        )
        if ego_speed < 0.08 and target_speed < 0.05:
            acc = 0.0
            self.last_acc = 0.0
            self.speed_integral = 0.0
            steer = 0.0
            self.last_steer = 0.0
            self.filtered_steer = 0.0
        self.last_debug.update(
            {
                "reason": str(getattr(plan_result, "reason", "") or ""),
                "plan_behavior": str(
                    getattr(plan_result, "behavior", "")
                ),
                "plan_target_speed": float(plan_result.target_speed),
                "preview_target_speed": target_speed,
                "trajectory_comfort_speed_cap": (
                    trajectory_comfort_speed_cap
                ),
                "trajectory_comfort_cap_applied": (
                    trajectory_comfort_cap_applied
                ),
                "alignment_speed_cap": alignment_speed_cap,
                "strict_alignment_speed_guard": (
                    self.config.strict_alignment_speed_guard
                ),
                "centerline_safety_stop": centerline_safety_stop,
                "output_acc": acc,
                "output_steer": steer,
            }
        )
        return ControlOutput(acc, target_speed, steer)
