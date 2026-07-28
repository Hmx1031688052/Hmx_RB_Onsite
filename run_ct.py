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


def _has_option(arguments, *names):
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in names:
            return True
    return False


def _append_default(arguments, value, *names):
    if not _has_option(arguments, *names):
        arguments.append(value)


def _install_score_config(speed_factor):
    base_config = rule_based_planner.PlannerConfig
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

    class CtStableController(base_controller):
        """Sprint controller with a fixed pulse independent of INS delay."""

        def control(self, *args, **kwargs):
            output = super().control(*args, **kwargs)
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

    _install_score_config(ct_args.ct_speed_factor)

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
    if not _has_option(
        forwarded,
        "--perception_source",
        "--perception-source",
    ):
        forwarded.extend(["--perception_source", "gt"])

    print(
        "[ct-score] isolated scoring entrypoint active "
        f"speed_factor={ct_args.ct_speed_factor:.3f} "
        f"alignment={ct_args.ct_alignment_duration:.3f}s "
        f"pulse={ct_args.ct_pulse_duration:.3f}s "
        f"accel={ct_args.ct_accel:.1f}m/s2"
    )
    run_path = os.path.join(os.path.dirname(__file__), "run.py")
    sys.argv = [run_path] + forwarded
    runpy.run_path(run_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    main()
