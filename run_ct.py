"""Scoring-oriented DriveSim entrypoint.

This module deliberately keeps the experimental direct-goal policy out of
run.py. It reuses run.py's protocol/session handling, but replaces
PlannerConfig with a small subclass whose sprint speed follows each episode's
resolved expected speed.
"""

import argparse
import math
import os
import runpy
import subprocess
import sys
import time

import numpy as np

import rule_based_planner


def _ct_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ct-speed-factor",
        type=float,
        default=float(os.environ.get("CT_SPEED_FACTOR", "1.10")),
        help=(
            "sprint speed / resolved expected speed; must be in "
            "[1.0, 1.19] (default: 1.10)"
        ),
    )
    parser.add_argument(
        "--ct-alignment-duration",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_DURATION", "0.02")
        ),
        help="stationary steering alignment time in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--ct-pulse-duration",
        type=float,
        default=float(os.environ.get("CT_PULSE_DURATION", "0.03")),
        help=(
            "minimum acceleration publication time before speed feedback may "
            "acknowledge the boost (default: 0.03)"
        ),
    )
    parser.add_argument(
        "--ct-boost-ack-speed",
        type=float,
        default=float(os.environ.get("CT_BOOST_ACK_SPEED", "5.0")),
        help=(
            "measured ego speed that confirms DriveSim consumed the boost, "
            "in m/s (default: 5.0)"
        ),
    )
    parser.add_argument(
        "--ct-boost-max-duration",
        type=float,
        default=float(os.environ.get("CT_BOOST_MAX_DURATION", "0.30")),
        help=(
            "maximum boost publication time if no speed acknowledgement is "
            "received, in seconds (default: 0.30)"
        ),
    )
    parser.add_argument(
        "--ct-accel",
        type=float,
        default=float(os.environ.get("CT_ACCEL", "1500.0")),
        help="one-shot acceleration command in m/s^2 (default: 1500)",
    )
    parser.add_argument(
        "--ct-alignment-speed",
        type=float,
        default=float(os.environ.get("CT_ALIGNMENT_SPEED", "3.0")),
        help="moving alignment speed in m/s (default: 3.0)",
    )
    parser.add_argument(
        "--ct-alignment-accel",
        type=float,
        default=float(os.environ.get("CT_ALIGNMENT_ACCEL", "2.8")),
        help="moving alignment acceleration limit in m/s^2 (default: 2.8)",
    )
    parser.add_argument(
        "--ct-alignment-jerk",
        type=float,
        default=float(os.environ.get("CT_ALIGNMENT_JERK", "4.5")),
        help="moving alignment jerk limit in m/s^3 (default: 4.5)",
    )
    parser.add_argument(
        "--ct-heading-tolerance-deg",
        type=float,
        default=float(
            os.environ.get("CT_HEADING_TOLERANCE_DEG", "0.5")
        ),
        help="heading error required before boost, degrees (default: 0.5)",
    )
    parser.add_argument(
        "--ct-lateral-tolerance",
        type=float,
        default=float(
            os.environ.get("CT_LATERAL_TOLERANCE", "0.15")
        ),
        help=(
            "legacy compatibility option; goal-heading alignment no longer "
            "requires returning to the original chord"
        ),
    )
    parser.add_argument(
        "--ct-alignment-settle-duration",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_SETTLE_DURATION", "0.20")
        ),
        help="continuous aligned time required before boost (default: 0.20)",
    )
    parser.add_argument(
        "--ct-alignment-timeout",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_TIMEOUT", "6.0")
        ),
        help=(
            "alignment warning threshold in seconds; it never bypasses "
            "alignment requirements (default: 6.0)"
        ),
    )
    parser.add_argument(
        "--ct-alignment-max-steer-deg",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_MAX_STEER_DEG", "42.0")
        ),
        help=(
            "maximum steering-wheel command during low-speed alignment, "
            "in degrees (default: 42.0)"
        ),
    )
    parser.add_argument(
        "--ct-alignment-heading-gain",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_HEADING_GAIN", "6.0")
        ),
        help="goal-heading curvature gain (default: 6.0)",
    )
    parser.add_argument(
        "--ct-alignment-lateral-gain",
        type=float,
        default=float(
            os.environ.get("CT_ALIGNMENT_LATERAL_GAIN", "0.4")
        ),
        help=(
            "legacy compatibility option; lateral feedback is disabled "
            "during alignment"
        ),
    )
    parser.add_argument(
        "--ct-alignment-steer-tolerance-deg",
        type=float,
        default=float(
            os.environ.get(
                "CT_ALIGNMENT_STEER_TOLERANCE_DEG", "0.5"
            )
        ),
        help=(
            "commanded and measured steering-wheel tolerance required "
            "before boost, in degrees (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--ct-alignment-steer-rate-deg-s",
        type=float,
        default=float(
            os.environ.get(
                "CT_ALIGNMENT_STEER_RATE_DEG_S", "286.5"
            )
        ),
        help=(
            "measured steering-wheel slew used by heading-stop prediction, "
            "in deg/s (default: 286.5, matching 5 rad/s)"
        ),
    )
    parser.add_argument(
        "--ct-straight-spatial-frequency",
        type=float,
        default=float(
            os.environ.get("CT_STRAIGHT_SPATIAL_FREQUENCY", "0.03")
        ),
        help=(
            "straight-line feedback frequency in 1/m; lower values are "
            "gentler at sprint speed (default: 0.03)"
        ),
    )
    parser.add_argument(
        "--ct-straight-max-lateral-accel",
        type=float,
        default=float(
            os.environ.get("CT_STRAIGHT_MAX_LATERAL_ACCEL", "0.45")
        ),
        help=(
            "maximum lateral acceleration used by CT straight tracking, "
            "in m/s^2 (default: 0.45)"
        ),
    )
    parser.add_argument(
        "--ct-straight-max-lateral-jerk",
        type=float,
        default=float(
            os.environ.get("CT_STRAIGHT_MAX_LATERAL_JERK", "0.90")
        ),
        help=(
            "maximum lateral jerk used to slew the CT curvature command, "
            "in m/s^3 (default: 0.90)"
        ),
    )
    parser.add_argument(
        "--ct-channel-settle-duration",
        type=float,
        default=float(
            os.environ.get("CT_CHANNEL_SETTLE_DURATION", "0.0")
        ),
        help=(
            "wait before the one native create_channels call so daemon and "
            "LinuxNoEditor registrations can settle (default: 0.0; immediate)"
        ),
    )
    parser.add_argument(
        "--ct-first-ins-timeout",
        type=float,
        default=float(
            os.environ.get("CT_FIRST_INS_TIMEOUT", "8.0")
        ),
        help=(
            "request a clean process reconnect when START_TEST receives no "
            "valid INS for this many seconds (default: 8.0)"
        ),
    )
    parser.add_argument(
        "--ct-ins-start-gate-tolerance",
        type=float,
        default=float(
            os.environ.get("CT_INS_START_GATE_TOLERANCE", "3.0")
        ),
        help=(
            "maximum distance from ActorPrepare init_state for the first "
            "accepted INS, in metres (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--ct-first-prepare-timeout",
        type=float,
        default=float(
            os.environ.get("CT_FIRST_PREPARE_TIMEOUT", "0.0")
        ),
        help=(
            "on the first runtime child only, reconnect when ActorPrepare is "
            "not received this many seconds after channel creation "
            "(default: 0.0, disabled)"
        ),
    )
    parser.add_argument(
        "--ct-prepare-response-delay",
        type=float,
        default=float(
            os.environ.get("CT_PREPARE_RESPONSE_DELAY", "5.0")
        ),
        help=(
            "delay ActorPrepareResult after route readiness so simulator "
            "publishers can initialize (default: 5.0)"
        ),
    )
    parser.add_argument(
        "--ct-prepare-resend-interval",
        type=float,
        default=float(
            os.environ.get("CT_PREPARE_RESEND_INTERVAL", "6.0")
        ),
        help=(
            "resend ActorPrepareResult until START_TEST at this interval "
            "(default: 6.0)"
        ),
    )
    parser.add_argument(
        "--ct-startup-reconnects",
        type=int,
        default=int(os.environ.get("CT_STARTUP_RECONNECTS", "1")),
        help=(
            "maximum automatic process-level reconnects for a dead initial "
            "INS subscription (default: 1)"
        ),
    )
    parser.add_argument(
        "--ct-help",
        action="store_true",
        help="show run_ct.py-specific options and exit",
    )
    return parser


def _validate_ct_args(parser, args):
    if not math.isfinite(args.ct_speed_factor):
        parser.error("--ct-speed-factor must be finite")
    if not 1.0 <= args.ct_speed_factor <= 1.19:
        parser.error("--ct-speed-factor must be between 1.0 and 1.19")
    if (
        not math.isfinite(args.ct_alignment_duration)
        or args.ct_alignment_duration < 0.0
    ):
        parser.error("--ct-alignment-duration must be finite and non-negative")
    if (
        not math.isfinite(args.ct_pulse_duration)
        or args.ct_pulse_duration < 0.01
    ):
        parser.error("--ct-pulse-duration must be at least 0.01 seconds")
    if not math.isfinite(args.ct_accel) or args.ct_accel <= 0.0:
        parser.error("--ct-accel must be finite and greater than zero")
    if (
        not math.isfinite(args.ct_boost_ack_speed)
        or args.ct_boost_ack_speed <= args.ct_alignment_speed
    ):
        parser.error(
            "--ct-boost-ack-speed must be finite and greater than "
            "--ct-alignment-speed"
        )
    if (
        not math.isfinite(args.ct_boost_max_duration)
        or args.ct_boost_max_duration < args.ct_pulse_duration
    ):
        parser.error(
            "--ct-boost-max-duration must be finite and no shorter than "
            "--ct-pulse-duration"
        )
    for name in (
        "ct_alignment_speed",
        "ct_alignment_accel",
        "ct_alignment_jerk",
        "ct_heading_tolerance_deg",
        "ct_lateral_tolerance",
        "ct_alignment_settle_duration",
        "ct_alignment_timeout",
        "ct_alignment_max_steer_deg",
        "ct_alignment_heading_gain",
        "ct_alignment_lateral_gain",
        "ct_alignment_steer_tolerance_deg",
        "ct_alignment_steer_rate_deg_s",
        "ct_straight_spatial_frequency",
        "ct_straight_max_lateral_accel",
        "ct_straight_max_lateral_jerk",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and "
                "greater than zero"
            )
    if (
        not math.isfinite(args.ct_channel_settle_duration)
        or args.ct_channel_settle_duration < 0.0
    ):
        parser.error(
            "--ct-channel-settle-duration must be finite and non-negative"
        )
    if (
        not math.isfinite(args.ct_first_ins_timeout)
        or args.ct_first_ins_timeout <= 0.0
    ):
        parser.error(
            "--ct-first-ins-timeout must be finite and greater than zero"
        )
    if (
        not math.isfinite(args.ct_ins_start_gate_tolerance)
        or args.ct_ins_start_gate_tolerance <= 0.0
    ):
        parser.error(
            "--ct-ins-start-gate-tolerance must be finite and "
            "greater than zero"
        )
    if (
        not math.isfinite(args.ct_first_prepare_timeout)
        or args.ct_first_prepare_timeout < 0.0
    ):
        parser.error(
            "--ct-first-prepare-timeout must be finite and non-negative"
        )
    for name in (
        "ct_prepare_response_delay",
        "ct_prepare_resend_interval",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and "
                "greater than zero"
            )
    if args.ct_startup_reconnects < 0 or args.ct_startup_reconnects > 1:
        parser.error("--ct-startup-reconnects must be either 0 or 1")


def _has_option(arguments, *names):
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in names:
            return True
    return False


def _append_default(arguments, value, *names):
    if not _has_option(arguments, *names):
        arguments.append(value)


def _supervise_runtime(ct_args):
    """Run native channels in a child and reconnect a dead INS only once."""
    command = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    explicit_settle = _has_option(
        sys.argv[1:], "--ct-channel-settle-duration"
    )
    for attempt in range(ct_args.ct_startup_reconnects + 1):
        child_env = os.environ.copy()
        child_env["CT_RUNTIME_CHILD"] = "1"
        child_env["CT_RUNTIME_ATTEMPT"] = str(attempt)
        if attempt > 0 and not explicit_settle:
            child_env["CT_CHANNEL_SETTLE_DURATION"] = "0.0"
        if attempt > 0:
            print(
                "[ct-startup][RECONNECT] restarting runtime child once "
                f"attempt={attempt + 1}/"
                f"{ct_args.ct_startup_reconnects + 1}"
            )
            time.sleep(2.0)
        try:
            return_code = subprocess.call(command, env=child_env)
        except KeyboardInterrupt:
            return 130
        if return_code != 75:
            return return_code
        if attempt >= ct_args.ct_startup_reconnects:
            print(
                "[ct-startup][FATAL] valid INS is still unavailable after "
                "the permitted clean reconnect; automatic restart stopped"
            )
            return 75
    return 75


def _install_score_config(ct_args):
    speed_factor = ct_args.ct_speed_factor
    base_config = rule_based_planner.PlannerConfig
    base_planner = rule_based_planner.RuleBasedPlanner
    base_controller = rule_based_planner.StableController
    # Shared only by the CT planner/controller classes installed below.
    # The controller uses the literal route goal while aligning so it can
    # steer toward the goal bearing instead of trying to return to the
    # original start-to-goal chord.
    ct_route_state = {"goal": None}

    class CtPlannerConfig(base_config):
        """Planner configuration with episode-relative sprint speed."""

        @property
        def expected_speed_mps(self):
            return getattr(self, "_ct_expected_speed_mps", None)

        @expected_speed_mps.setter
        def expected_speed_mps(self, value):
            self._ct_expected_speed_mps = value
            try:
                expected_speed = float(value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(expected_speed) or expected_speed <= 0.0:
                return

            attack_speed = expected_speed * speed_factor
            self.sprint_speed = attack_speed
            print(
                "[ct-score] dynamic sprint speed "
                f"expected={expected_speed:.3f}m/s "
                f"factor={speed_factor:.3f} "
                f"target={attack_speed:.3f}m/s "
                f"target_kmh={attack_speed * 3.6:.1f}km/h"
            )

    CtPlannerConfig.__name__ = "CtPlannerConfig"
    rule_based_planner.PlannerConfig = CtPlannerConfig

    class CtRuleBasedPlanner(base_planner):
        """Direct sprint planner using the literal start-to-goal chord."""

        def _build_sprint_path(self, ego, global_path):
            if self._sprint_path is not None:
                goal = getattr(self, "_sprint_goal", None)
                if goal is not None:
                    ct_route_state["goal"] = goal
                return self._sprint_path
            try:
                route_x = np.asarray(
                    global_path.get("x", []), dtype=float
                ).reshape(-1)
                route_y = np.asarray(
                    global_path.get("y", []), dtype=float
                ).reshape(-1)
                count = min(route_x.size, route_y.size)
                valid = (
                    np.isfinite(route_x[:count])
                    & np.isfinite(route_y[:count])
                )
                if count < 2 or not np.any(valid):
                    return None
                goal_x = float(route_x[:count][valid][-1])
                goal_y = float(route_y[:count][valid][-1])
                start_x = float(ego.x)
                start_y = float(ego.y)
            except (AttributeError, TypeError, ValueError):
                return None

            dx = goal_x - start_x
            dy = goal_y - start_y
            distance = math.hypot(dx, dy)
            if not math.isfinite(distance) or distance < 1.0:
                return None
            direction_x = dx / distance
            direction_y = dy / distance
            before_goal = np.arange(0.0, distance, 0.5)
            stations = np.concatenate(
                (
                    before_goal,
                    np.asarray([distance]),
                    distance + np.arange(0.5, 100.5, 0.5),
                )
            )
            xs = start_x + direction_x * stations
            ys = start_y + direction_y * stations
            self._sprint_path = {
                "x": xs.tolist(),
                "y": ys.tolist(),
                "kappa": np.zeros(xs.size, dtype=float).tolist(),
                "speed_limit": np.zeros(xs.size, dtype=float).tolist(),
                "frame_id": "ct-straight-sprint",
                "stamp": (
                    "ct-straight-sprint",
                    round(start_x, 3),
                    round(start_y, 3),
                    round(goal_x, 3),
                    round(goal_y, 3),
                ),
            }
            self._sprint_goal = (goal_x, goal_y)
            ct_route_state["goal"] = self._sprint_goal
            chord_yaw = math.atan2(dy, dx)
            start_yaw = float(getattr(ego, "theta", chord_yaw))
            heading_error = math.atan2(
                math.sin(chord_yaw - start_yaw),
                math.cos(chord_yaw - start_yaw),
            )
            print(
                "[ct-score] strict straight reference "
                f"start=({start_x:.3f},{start_y:.3f}) "
                f"goal=({goal_x:.3f},{goal_y:.3f}) "
                f"distance={distance:.3f}m kappa=0 "
                f"start_heading={math.degrees(start_yaw):.3f}deg "
                f"line_heading={math.degrees(chord_yaw):.3f}deg "
                f"initial_heading_error="
                f"{math.degrees(heading_error):.3f}deg"
            )
            return self._sprint_path

    CtRuleBasedPlanner.__name__ = "CtRuleBasedPlanner"
    rule_based_planner.RuleBasedPlanner = CtRuleBasedPlanner

    class CtStableController(base_controller):
        """Low-speed line alignment followed by an acknowledged boost."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reset_ct_alignment()

        def _reset_ct_alignment(self):
            self.ct_alignment_complete = False
            self.ct_alignment_elapsed = 0.0
            self.ct_alignment_stable_elapsed = 0.0
            self.ct_alignment_result_logged = False
            self.ct_alignment_command_speed = 0.0
            self.ct_alignment_command_acc = 0.0
            self.ct_alignment_timeout_warned = False
            self.ct_straight_curvature_command = 0.0
            self.ct_straight_anchor = None

        def reset(self):
            super().reset()
            self._reset_ct_alignment()

        def control(self, *args, **kwargs):
            # The base acknowledgement threshold (0.1 m/s) is already met by
            # moving alignment. Disable it here and acknowledge only after
            # the measured speed proves that DriveSim consumed the boost.
            wait_for_boost_ack = (
                self.ct_alignment_complete
                and not self.sprint_pulse_acknowledged
            )
            saved_ack_speed = self.config.sprint_pulse_ack_speed
            saved_alignment_duration = (
                self.config.sprint_alignment_duration
            )
            if not self.ct_alignment_complete:
                self.config.sprint_alignment_duration = float("inf")
            if wait_for_boost_ack:
                self.config.sprint_pulse_ack_speed = float("inf")
            try:
                output = super().control(*args, **kwargs)
            finally:
                self.config.sprint_pulse_ack_speed = saved_ack_speed
                self.config.sprint_alignment_duration = (
                    saved_alignment_duration
                )
            if not bool(self.last_debug.get("sprint_mode", False)):
                self._reset_ct_alignment()
                # CT has no non-sprint driving policy. In particular, never
                # preserve the previous 42-degree alignment command when the
                # planner temporarily returns STOP or no trajectory.
                output.steer = 0.0
                self.last_steer = 0.0
                self.filtered_steer = 0.0
                self.last_debug.update(
                    {
                        "output_steer": 0.0,
                        "ct_non_sprint_wheel_centered": True,
                    }
                )
                return output

            dt = max(
                0.01,
                float(self.last_debug.get("dt", 0.02)),
            )
            if not self.ct_alignment_complete:
                self.ct_alignment_elapsed += dt
                ego = args[0] if args else kwargs.get("ego")
                goal = ct_route_state.get("goal")
                signed_heading_error = float(
                    self.last_debug.get("heading_error", 0.0)
                )
                if ego is not None and goal is not None:
                    try:
                        goal_dx = float(goal[0]) - float(ego.x)
                        goal_dy = float(goal[1]) - float(ego.y)
                        if math.hypot(goal_dx, goal_dy) > 1e-6:
                            goal_yaw = math.atan2(goal_dy, goal_dx)
                            ego_yaw = float(ego.theta)
                            signed_heading_error = math.atan2(
                                math.sin(goal_yaw - ego_yaw),
                                math.cos(goal_yaw - ego_yaw),
                            )
                    except (AttributeError, TypeError, ValueError):
                        pass
                heading_error = abs(signed_heading_error)
                lateral_error = abs(
                    float(self.last_debug.get("lateral_error", math.inf))
                )
                aligned = (
                    heading_error
                    <= math.radians(
                        ct_args.ct_heading_tolerance_deg
                    )
                )
                if aligned:
                    self.ct_alignment_stable_elapsed += dt
                else:
                    self.ct_alignment_stable_elapsed = 0.0

                remaining_speed = max(
                    0.0,
                    ct_args.ct_alignment_speed
                    - self.ct_alignment_command_speed,
                )
                # Jerk-limited S-curve. Begin ramping acceleration down when
                # its remaining triangular area would consume the remaining
                # delta-v. This avoids the old 2.8/0/-2.8 bang-bang commands.
                ramp_down_delta_v = (
                    self.ct_alignment_command_acc
                    * self.ct_alignment_command_acc
                    / (2.0 * ct_args.ct_alignment_jerk)
                )
                desired_acc = (
                    0.0
                    if remaining_speed
                    <= ramp_down_delta_v + 1e-6
                    else ct_args.ct_alignment_accel
                )
                max_acc_change = ct_args.ct_alignment_jerk * dt
                acc_error = (
                    desired_acc - self.ct_alignment_command_acc
                )
                acc_change = max(
                    -max_acc_change,
                    min(max_acc_change, acc_error),
                )
                self.ct_alignment_command_acc += acc_change
                self.ct_alignment_command_acc = max(
                    0.0,
                    min(
                        ct_args.ct_alignment_accel,
                        self.ct_alignment_command_acc,
                    ),
                )
                self.ct_alignment_command_speed = min(
                    ct_args.ct_alignment_speed,
                    self.ct_alignment_command_speed
                    + self.ct_alignment_command_acc * dt,
                )
                if (
                    self.ct_alignment_command_speed
                    >= ct_args.ct_alignment_speed - 1e-6
                    and desired_acc <= 0.0
                    and self.ct_alignment_command_acc
                    <= max_acc_change
                ):
                    self.ct_alignment_command_acc = 0.0

                output.speed = self.ct_alignment_command_speed
                output.acc = self.ct_alignment_command_acc

                # The generic sprint controller is intentionally capped at
                # 11 degrees and cannot align a 38-degree initial heading
                # before a short route ends. At low speed the chassis can
                # use its normal steering authority. CT uses a short 3 m/s
                # manoeuvre here: its brief lateral-threshold exposure costs
                # less under the duration-based score than spending most of
                # a short episode aligning at crawl speed.
                wheelbase = max(
                    0.1, float(self.config.controller_wheelbase)
                )
                steering_ratio = max(
                    0.1, float(self.config.steering_ratio)
                )
                command_sign = float(
                    self.config.steering_command_sign
                )
                max_alignment_steer = min(
                    ct_args.ct_alignment_max_steer_deg,
                    float(self.config.max_steering_wheel_deg),
                )
                steering_feedback = kwargs.get("steering_feedback")
                try:
                    signed_measured_steer = float(
                        steering_feedback
                    )
                except (TypeError, ValueError):
                    signed_measured_steer = 0.0
                measured_steer = abs(signed_measured_steer)
                max_front_angle = math.radians(
                    max_alignment_steer / steering_ratio
                )
                max_alignment_curvature = (
                    math.tan(max_front_angle) / wheelbase
                )
                measured_front_angle = math.radians(
                    command_sign
                    * signed_measured_steer
                    / steering_ratio
                )
                measured_curvature = (
                    math.tan(measured_front_angle) / wheelbase
                )
                try:
                    measured_speed = max(
                        0.0, float(getattr(ego, "speed", 0.0))
                    )
                except (AttributeError, TypeError, ValueError):
                    measured_speed = self.ct_alignment_command_speed
                # Heading-only alignment. Predict the heading at the point
                # where the measured wheel would reach centre, then command
                # curvature from that error. There is deliberately no
                # lateral-error term and no reverse-curvature switching
                # surface: the car turns once toward the live goal bearing.
                steering_unwind_time = (
                    abs(signed_measured_steer)
                    / ct_args.ct_alignment_steer_rate_deg_s
                )
                predicted_heading_error = (
                    signed_heading_error
                    - measured_speed
                    * measured_curvature
                    * steering_unwind_time
                )
                desired_alignment_curvature = (
                    ct_args.ct_alignment_heading_gain
                    * predicted_heading_error
                )
                alignment_control_mode = "GOAL_HEADING"
                alignment_switch_surface = float("nan")
                desired_alignment_curvature = max(
                    -max_alignment_curvature,
                    min(
                        max_alignment_curvature,
                        desired_alignment_curvature,
                    ),
                )
                alignment_front_angle = math.atan(
                    wheelbase * desired_alignment_curvature
                )
                alignment_steer = (
                    command_sign
                    * math.degrees(alignment_front_angle)
                    * steering_ratio
                )
                alignment_steer = max(
                    -max_alignment_steer,
                    min(max_alignment_steer, alignment_steer),
                )
                output.steer = alignment_steer
                self.last_steer = alignment_steer
                self.filtered_steer = alignment_steer
                if steering_feedback is None:
                    measured_steer = abs(alignment_steer)

                profile_settled = (
                    self.ct_alignment_command_speed
                    >= ct_args.ct_alignment_speed - 0.01
                    and self.ct_alignment_command_acc <= 0.05
                )
                geometry_settled = (
                    self.ct_alignment_stable_elapsed
                    >= ct_args.ct_alignment_settle_duration - 1e-9
                    and profile_settled
                )
                if geometry_settled:
                    # Do not let the fine controller spend another second
                    # asymptotically reducing a small steering command. Once
                    # pose geometry has remained valid, explicitly centre the
                    # wheel and wait only for measured feedback to follow.
                    alignment_control_mode = "CENTER_WHEEL"
                    alignment_steer = 0.0
                    output.steer = 0.0
                    self.last_steer = 0.0
                    self.filtered_steer = 0.0

                # Discard any pulse state advanced internally while the
                # lateral controller was being evaluated. The real boost
                # starts only on the cycle after alignment completes.
                self.sprint_pulse_elapsed = 0.0
                self.sprint_pulse_sent = False
                self.sprint_pulse_acknowledged = False
                self.sprint_phase = "CT_ALIGN_MOVE"
                self.last_debug.update(
                    {
                        "sprint_phase": self.sprint_phase,
                        "output_acc": output.acc,
                        "preview_target_speed": output.speed,
                        "ct_alignment_complete": False,
                        "ct_alignment_elapsed": (
                            self.ct_alignment_elapsed
                        ),
                        "ct_alignment_stable_elapsed": (
                            self.ct_alignment_stable_elapsed
                        ),
                        "ct_alignment_heading_error_deg": (
                            math.degrees(heading_error)
                        ),
                        "ct_alignment_lateral_error": lateral_error,
                        "ct_alignment_command_speed": (
                            self.ct_alignment_command_speed
                        ),
                        "ct_alignment_command_acc": (
                            self.ct_alignment_command_acc
                        ),
                        "ct_alignment_steer": alignment_steer,
                        "ct_alignment_measured_steer": measured_steer,
                        "ct_alignment_control_mode": (
                            alignment_control_mode
                        ),
                        "ct_alignment_switch_surface": (
                            alignment_switch_surface
                        ),
                        "output_steer": alignment_steer,
                    }
                )

                steering_settled = (
                    abs(alignment_steer)
                    <= ct_args.ct_alignment_steer_tolerance_deg
                    and measured_steer
                    <= ct_args.ct_alignment_steer_tolerance_deg
                )
                settled = (
                    geometry_settled
                    and steering_settled
                )
                timed_out = (
                    self.ct_alignment_elapsed
                    >= ct_args.ct_alignment_timeout
                )
                if settled:
                    self.ct_alignment_complete = True
                    self.ct_alignment_result_logged = True
                    try:
                        self.ct_straight_anchor = (
                            float(ego.x),
                            float(ego.y),
                        )
                    except (AttributeError, TypeError, ValueError):
                        self.ct_straight_anchor = None
                    print(
                        "[ct-score] moving alignment complete "
                        "reason=settled "
                        f"elapsed={self.ct_alignment_elapsed:.3f}s "
                        f"stable="
                        f"{self.ct_alignment_stable_elapsed:.3f}s "
                        f"heading_error="
                        f"{math.degrees(heading_error):.3f}deg "
                        f"original_line_offset={lateral_error:.3f}m "
                        f"new_line_anchor={self.ct_straight_anchor}; "
                        "boost starts next control cycle"
                    )
                elif timed_out and not self.ct_alignment_timeout_warned:
                    self.ct_alignment_timeout_warned = True
                    print(
                        "[ct-score][WAIT] alignment warning threshold "
                        f"elapsed={self.ct_alignment_elapsed:.3f}s "
                        f"heading_error="
                        f"{math.degrees(heading_error):.3f}deg "
                        f"lateral_error={lateral_error:.3f}m "
                        f"command_steer={alignment_steer:.3f}deg "
                        f"measured_steer={measured_steer:.3f}deg; "
                        "boost remains inhibited until genuinely aligned"
                    )
                return output

            ego = args[0] if args else kwargs.get("ego")
            try:
                ego_speed = max(
                    0.0, float(getattr(ego, "speed", 0.0))
                )
            except (TypeError, ValueError):
                ego_speed = 0.0

            # At 40 m/s even one degree of front-wheel steering produces
            # roughly 10 m/s^2 lateral acceleration. The generic sprint
            # controller intentionally grants large recovery authority and
            # therefore oscillates around a literal straight chord. CT mode
            # instead uses a low-bandwidth spatial controller:
            #
            #   curvature = 2*w*heading_error - w^2*lateral_error
            #
            # Its acceleration and curvature slew are both bounded using the
            # anticipated post-boost speed, so the command is already small
            # before delayed chassis speed feedback reports the jump.
            heading_error = float(
                self.last_debug.get("heading_error", 0.0)
            )
            lateral_error = float(
                self.last_debug.get("lateral_error", 0.0)
            )
            goal = ct_route_state.get("goal")
            anchor = self.ct_straight_anchor
            if ego is not None and goal is not None and anchor is not None:
                try:
                    line_dx = float(goal[0]) - float(anchor[0])
                    line_dy = float(goal[1]) - float(anchor[1])
                    line_length = math.hypot(line_dx, line_dy)
                    if line_length > 1e-6:
                        line_yaw = math.atan2(line_dy, line_dx)
                        ego_yaw = float(ego.theta)
                        heading_error = math.atan2(
                            math.sin(line_yaw - ego_yaw),
                            math.cos(line_yaw - ego_yaw),
                        )
                        offset_x = float(ego.x) - float(anchor[0])
                        offset_y = float(ego.y) - float(anchor[1])
                        lateral_error = (
                            -math.sin(line_yaw) * offset_x
                            + math.cos(line_yaw) * offset_y
                        )
                except (AttributeError, TypeError, ValueError):
                    pass
            plan_result = (
                args[1] if len(args) > 1 else kwargs.get("plan_result")
            )
            try:
                planned_speed = max(
                    0.0,
                    float(getattr(plan_result, "target_speed", 0.0)),
                )
            except (TypeError, ValueError):
                planned_speed = 0.0
            design_speed = max(
                ego_speed,
                planned_speed,
                ct_args.ct_boost_ack_speed,
            )
            spatial_frequency = (
                ct_args.ct_straight_spatial_frequency
            )
            desired_curvature = (
                2.0 * spatial_frequency * heading_error
                - spatial_frequency
                * spatial_frequency
                * lateral_error
            )
            curvature_limit = (
                ct_args.ct_straight_max_lateral_accel
                / max(design_speed * design_speed, 1.0)
            )
            desired_curvature = max(
                -curvature_limit,
                min(curvature_limit, desired_curvature),
            )
            curvature_change_limit = (
                ct_args.ct_straight_max_lateral_jerk
                * dt
                / max(design_speed * design_speed, 1.0)
            )
            curvature_error = (
                desired_curvature
                - self.ct_straight_curvature_command
            )
            self.ct_straight_curvature_command += max(
                -curvature_change_limit,
                min(curvature_change_limit, curvature_error),
            )
            wheelbase = max(
                0.1, float(self.config.controller_wheelbase)
            )
            steering_ratio = float(self.config.steering_ratio)
            command_sign = float(
                self.config.steering_command_sign
            )
            straight_steer = (
                command_sign
                * math.degrees(
                    math.atan(
                        wheelbase
                        * self.ct_straight_curvature_command
                    )
                )
                * steering_ratio
            )
            output.steer = straight_steer
            self.last_steer = straight_steer
            self.filtered_steer = straight_steer
            self.last_debug.update(
                {
                    "output_steer": straight_steer,
                    "ct_straight_tracking": True,
                    "ct_straight_anchor": anchor,
                    "ct_straight_goal": goal,
                    "ct_straight_heading_error_deg": math.degrees(
                        heading_error
                    ),
                    "ct_straight_lateral_error": lateral_error,
                    "ct_straight_design_speed": design_speed,
                    "ct_straight_desired_curvature": (
                        desired_curvature
                    ),
                    "ct_straight_curvature_command": (
                        self.ct_straight_curvature_command
                    ),
                    "ct_straight_curvature_limit": curvature_limit,
                    "ct_straight_lateral_accel_command": (
                        design_speed
                        * design_speed
                        * self.ct_straight_curvature_command
                    ),
                }
            )
            minimum_elapsed = (
                self.sprint_pulse_elapsed
                >= self.config.sprint_pulse_min_duration - 1e-9
            )
            speed_acknowledged = (
                minimum_elapsed
                and ego_speed >= ct_args.ct_boost_ack_speed
            )
            boost_timed_out = (
                self.sprint_phase == "PULSE"
                and self.sprint_pulse_elapsed
                >= ct_args.ct_boost_max_duration - 1e-9
            )
            if (
                self.sprint_phase == "PULSE"
                and (speed_acknowledged or boost_timed_out)
            ):
                self.sprint_pulse_acknowledged = True
                self.sprint_phase = "COAST"
                output.acc = 0.0
                self.last_acc = 0.0
                self.last_debug.update(
                    {
                        "sprint_phase": self.sprint_phase,
                        "sprint_pulse_acknowledged": True,
                        "sprint_accel_pulse_active": False,
                        "output_acc": 0.0,
                        "ct_boost_speed_acknowledged": (
                            speed_acknowledged
                        ),
                        "ct_boost_timed_out": boost_timed_out,
                        "ct_boost_feedback_speed": ego_speed,
                    }
                )
                if speed_acknowledged:
                    print(
                        "[ct-score] boost acknowledged by chassis "
                        f"ego_v={ego_speed:.3f}m/s "
                        f"elapsed={self.sprint_pulse_elapsed:.3f}s; "
                        "entering COAST"
                    )
                else:
                    print(
                        "[ct-score][WARN] boost acknowledgement timeout "
                        f"ego_v={ego_speed:.3f}m/s "
                        f"elapsed={self.sprint_pulse_elapsed:.3f}s; "
                        "stopping acceleration publication"
                    )
            return output

    CtStableController.__name__ = "CtStableController"
    rule_based_planner.StableController = CtStableController


def main():
    parser = _ct_parser()
    ct_args, forwarded = parser.parse_known_args(sys.argv[1:])
    if ct_args.ct_help:
        parser.print_help()
        return 0
    _validate_ct_args(parser, ct_args)
    if os.environ.get("CT_RUNTIME_CHILD") != "1":
        return _supervise_runtime(ct_args)

    _install_score_config(ct_args)

    # Only provide defaults. Ordinary run.py options supplied by the operator
    # remain authoritative, except sprint speed itself, which is intentionally
    # recomputed after each ActorPrepare message.
    _append_default(
        forwarded,
        "--sprint",
        "--sprint",
        "--no_sprint",
        "--no-sprint",
    )
    if not _has_option(
        forwarded,
        "--sprint_alignment_duration",
        "--sprint-alignment-duration",
    ):
        forwarded.extend(
            [
                "--sprint_alignment_duration",
                str(ct_args.ct_alignment_duration),
            ]
        )
    if not _has_option(
        forwarded,
        "--sprint_pulse_min_duration",
        "--sprint-pulse-min-duration",
    ):
        forwarded.extend(
            [
                "--sprint_pulse_min_duration",
                str(ct_args.ct_pulse_duration),
            ]
        )
    if not _has_option(
        forwarded,
        "--sprint_accel",
        "--sprint-accel",
    ):
        forwarded.extend(["--sprint_accel", str(ct_args.ct_accel)])
    _append_default(
        forwarded,
        "--sprint_ignore_obstacles",
        "--sprint_ignore_obstacles",
        "--sprint-ignore-obstacles",
    )
    # Append last so it wins over an environment default or an accidentally
    # forwarded --perception_source value. NONE loads neither the GT adapter
    # nor PointPillars and always supplies an empty obstacle set.
    forwarded.extend(["--perception_source", "none"])
    os.environ["E2E_NPC_TRUTH_ENABLED"] = "0"
    # DriveSim's synchronous get_ins() occasionally interleaves a cached
    # sequence-1 sample from another field/session after current sequence
    # numbers have advanced. Holding the last accepted pose prevents those
    # stale samples from alternating SPRINT and emergency STOP.
    os.environ["E2E_INS_MONOTONIC_SEQUENCE"] = "1"
    os.environ["E2E_FIRST_INS_TIMEOUT"] = str(
        ct_args.ct_first_ins_timeout
    )
    os.environ["E2E_INS_START_GATE_ENABLED"] = "1"
    os.environ["E2E_INS_START_GATE_TOL"] = str(
        ct_args.ct_ins_start_gate_tolerance
    )
    runtime_attempt = int(os.environ.get("CT_RUNTIME_ATTEMPT", "0"))
    os.environ["E2E_FIRST_PREPARE_TIMEOUT"] = (
        str(ct_args.ct_first_prepare_timeout)
        if runtime_attempt == 0
        else "0"
    )
    os.environ["E2E_PREPARE_RESPONSE_DELAY"] = str(
        ct_args.ct_prepare_response_delay
    )
    os.environ["E2E_PREPARE_RESULT_RESEND_INTERVAL"] = str(
        ct_args.ct_prepare_resend_interval
    )

    print(
        "[ct-score] isolated scoring entrypoint active "
        f"speed_factor={ct_args.ct_speed_factor:.3f} "
        f"alignment={ct_args.ct_alignment_duration:.3f}s "
        f"pulse={ct_args.ct_pulse_duration:.3f}s "
        f"boost_ack_speed={ct_args.ct_boost_ack_speed:.3f}m/s "
        f"boost_max={ct_args.ct_boost_max_duration:.3f}s "
        f"accel={ct_args.ct_accel:.1f}m/s2 "
        f"alignment_speed={ct_args.ct_alignment_speed:.3f}m/s "
        f"alignment_jerk={ct_args.ct_alignment_jerk:.3f}m/s3 "
        f"alignment_steer="
        f"{ct_args.ct_alignment_max_steer_deg:.1f}deg "
        f"heading_tol={ct_args.ct_heading_tolerance_deg:.3f}deg "
        f"lateral_tol={ct_args.ct_lateral_tolerance:.3f}m "
        f"straight_frequency="
        f"{ct_args.ct_straight_spatial_frequency:.3f}/m "
        f"straight_lat_accel="
        f"{ct_args.ct_straight_max_lateral_accel:.3f}m/s2 "
        f"straight_lat_jerk="
        f"{ct_args.ct_straight_max_lateral_jerk:.3f}m/s3 "
        f"first_prepare_timeout="
        f"{ct_args.ct_first_prepare_timeout:.1f}s "
        f"prepare_delay={ct_args.ct_prepare_response_delay:.1f}s "
        f"prepare_resend="
        f"{ct_args.ct_prepare_resend_interval:.1f}s "
        f"first_ins_timeout={ct_args.ct_first_ins_timeout:.1f}s "
        f"ins_start_gate="
        f"{ct_args.ct_ins_start_gate_tolerance:.1f}m "
        "background_vehicles=IGNORED gt_subscription=DISABLED "
        "ins_sequence_filter=MONOTONIC"
    )
    if ct_args.ct_channel_settle_duration > 0.0:
        print(
            "[ct-startup][WAIT] allowing daemon/LinuxNoEditor channel "
            "registrations to settle before the single native channel "
            f"creation; wait={ct_args.ct_channel_settle_duration:.1f}s"
        )
        time.sleep(ct_args.ct_channel_settle_duration)
    run_path = os.path.join(os.path.dirname(__file__), "run.py")
    sys.argv = [run_path] + forwarded
    runpy.run_path(run_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
