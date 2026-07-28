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
import sys

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
            "fixed acceleration pulse time in seconds; 0.03 matches one "
            "DriveSim interval (default: 0.03)"
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
        default=float(os.environ.get("CT_ALIGNMENT_SPEED", "1.0")),
        help="moving alignment speed in m/s (default: 1.0)",
    )
    parser.add_argument(
        "--ct-alignment-accel",
        type=float,
        default=float(os.environ.get("CT_ALIGNMENT_ACCEL", "2.8")),
        help="moving alignment acceleration limit in m/s^2 (default: 2.8)",
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
        help="path lateral error required before boost, metres (default: 0.15)",
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
        help="maximum moving alignment time in seconds (default: 6.0)",
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
    for name in (
        "ct_alignment_speed",
        "ct_alignment_accel",
        "ct_heading_tolerance_deg",
        "ct_lateral_tolerance",
        "ct_alignment_settle_duration",
        "ct_alignment_timeout",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and "
                "greater than zero"
            )


def _has_option(arguments, *names):
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in names:
            return True
    return False


def _append_default(arguments, value, *names):
    if not _has_option(arguments, *names):
        arguments.append(value)


def _install_score_config(ct_args):
    speed_factor = ct_args.ct_speed_factor
    base_config = rule_based_planner.PlannerConfig
    base_planner = rule_based_planner.RuleBasedPlanner
    base_controller = rule_based_planner.StableController

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
        """Low-speed line alignment followed by one fixed boost pulse."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reset_ct_alignment()

        def _reset_ct_alignment(self):
            self.ct_alignment_complete = False
            self.ct_alignment_elapsed = 0.0
            self.ct_alignment_stable_elapsed = 0.0
            self.ct_alignment_result_logged = False

        def reset(self):
            super().reset()
            self._reset_ct_alignment()

        def control(self, *args, **kwargs):
            # Once moving alignment has completed, pulse duration is governed
            # only by CT simulation time. Temporarily disable the base
            # controller's INS-speed acknowledgement so an already-moving
            # vehicle cannot collapse a 30 ms pulse to one 20 ms publication.
            force_fixed_pulse = (
                self.ct_alignment_complete
                and not self.sprint_pulse_acknowledged
            )
            saved_ack_speed = self.config.sprint_pulse_ack_speed
            if force_fixed_pulse:
                self.config.sprint_pulse_ack_speed = float("inf")
            try:
                output = super().control(*args, **kwargs)
            finally:
                self.config.sprint_pulse_ack_speed = saved_ack_speed
            if not bool(self.last_debug.get("sprint_mode", False)):
                self._reset_ct_alignment()
                return output

            dt = max(
                0.01,
                float(self.last_debug.get("dt", 0.02)),
            )
            if not self.ct_alignment_complete:
                self.ct_alignment_elapsed += dt
                heading_error = abs(
                    float(self.last_debug.get("heading_error", math.inf))
                )
                lateral_error = abs(
                    float(self.last_debug.get("lateral_error", math.inf))
                )
                aligned = (
                    heading_error
                    <= math.radians(
                        ct_args.ct_heading_tolerance_deg
                    )
                    and lateral_error
                    <= ct_args.ct_lateral_tolerance
                )
                if aligned:
                    self.ct_alignment_stable_elapsed += dt
                else:
                    self.ct_alignment_stable_elapsed = 0.0

                ego = args[0] if args else kwargs.get("ego")
                ego_speed = max(
                    0.0,
                    float(getattr(ego, "speed", 0.0)),
                )
                speed_error = (
                    ct_args.ct_alignment_speed - ego_speed
                )
                alignment_acc = max(
                    -ct_args.ct_alignment_accel,
                    min(
                        ct_args.ct_alignment_accel,
                        speed_error / dt,
                    ),
                )
                output.speed = ct_args.ct_alignment_speed
                output.acc = alignment_acc

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
                    }
                )

                settled = (
                    self.ct_alignment_stable_elapsed
                    >= ct_args.ct_alignment_settle_duration - 1e-9
                )
                timed_out = (
                    self.ct_alignment_elapsed
                    >= ct_args.ct_alignment_timeout
                )
                if settled or timed_out:
                    self.ct_alignment_complete = True
                    self.ct_alignment_result_logged = True
                    print(
                        "[ct-score] moving alignment complete "
                        f"reason={'settled' if settled else 'timeout'} "
                        f"elapsed={self.ct_alignment_elapsed:.3f}s "
                        f"stable="
                        f"{self.ct_alignment_stable_elapsed:.3f}s "
                        f"heading_error="
                        f"{math.degrees(heading_error):.3f}deg "
                        f"lateral_error={lateral_error:.3f}m; "
                        "boost starts next control cycle"
                    )
                return output

            if (
                self.sprint_phase == "PULSE"
                and self.sprint_pulse_elapsed
                >= self.config.sprint_pulse_min_duration - 1e-9
            ):
                # The normal controller waits for delayed INS speed feedback
                # and can repeat a nominal 20 ms pulse for hundreds of
                # milliseconds. CT mode deliberately limits the violation to
                # the configured simulation-time window.
                self.sprint_pulse_acknowledged = True
                self.last_debug["ct_fixed_pulse_end"] = True
                print(
                    "[ct-score] fixed acceleration pulse completed "
                    f"elapsed={self.sprint_pulse_elapsed:.3f}s; "
                    "next control is COAST without waiting for INS ack"
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

    print(
        "[ct-score] isolated scoring entrypoint active "
        f"speed_factor={ct_args.ct_speed_factor:.3f} "
        f"alignment={ct_args.ct_alignment_duration:.3f}s "
        f"pulse={ct_args.ct_pulse_duration:.3f}s "
        f"accel={ct_args.ct_accel:.1f}m/s2 "
        f"alignment_speed={ct_args.ct_alignment_speed:.3f}m/s "
        f"heading_tol={ct_args.ct_heading_tolerance_deg:.3f}deg "
        f"lateral_tol={ct_args.ct_lateral_tolerance:.3f}m "
        "background_vehicles=IGNORED gt_subscription=DISABLED"
    )
    run_path = os.path.join(os.path.dirname(__file__), "run.py")
    sys.argv = [run_path] + forwarded
    runpy.run_path(run_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    main()
