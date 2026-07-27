import json
import os
import sys
import tempfile
import unittest

import numpy as np


SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from drive_trace_logger import DriveTraceLogger  # noqa: E402


class Object(object):
    def __init__(self, **values):
        self.__dict__.update(values)


class DriveTraceLoggerTest(unittest.TestCase):
    def test_cycle_contains_ego_obstacle_plan_and_control(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = DriveTraceLogger(
                output_dir,
                period_sec=0.10,
                max_trajectory_points=10,
            )
            logger.start_session(
                "session-1",
                "map.xodr",
                {"start": [10.0, 2.0]},
            )
            count = 101
            trajectory = Object(
                t=np.linspace(0.0, 5.0, count),
                s=np.linspace(20.0, 40.0, count),
                d=np.linspace(0.0, 3.5, count),
                x=np.linspace(10.0, 30.0, count),
                y=np.linspace(2.0, 5.5, count),
                yaw=np.zeros(count),
                kappa=np.zeros(count),
                speed=np.full(count, 4.0),
                accel=np.zeros(count),
                d_speed=np.zeros(count),
                d_accel=np.zeros(count),
            )
            plan = Object(
                trajectory=trajectory,
                target_speed=4.5,
                behavior="AVOID_LEFT",
                emergency=False,
                reason="",
            )
            ego = Object(
                x=10.0,
                y=2.0,
                theta=0.0,
                speed=3.0,
                length=4.6,
                width=1.9,
            )
            nearby = Object(
                id="7",
                x=13.0,
                y=4.0,
                theta=0.0,
                speed=2.0,
                speed_valid=True,
                world_vx=2.0,
                world_vy=0.0,
                length=4.5,
                width=2.0,
                obs_type=1,
                score=0.82,
                is_static=False,
                is_virtual=False,
                track_hits=8,
                track_misses=0,
                track_age=1.2,
                track_predicted=False,
            )
            planner_debug = {
                "projection": {"s": 20.0, "d": 0.1},
                "selected_target_d": 3.5,
                "selected_clearance": 1.15,
                "obstacles": [
                    {"id": "7", "s": 23.0, "d": 2.0, "gap": 3.0}
                ],
            }
            command = Object(acc=1.2, speed=4.0, steer=8.0)
            controller_debug = {
                "lateral_error": 0.15,
                "heading_error": 0.02,
            }
            feedback = Object(
                steering_wheel_angle=7.5,
                accelerator_pedal_position=0.4,
                brake_pedal_position=0.0,
            )

            written = logger.record_cycle(
                loop_count=12,
                ego=ego,
                obstacles=[nearby],
                plan_result=plan,
                planner_debug=planner_debug,
                control_command=command,
                controller_debug=controller_debug,
                vehicle_feedback=feedback,
                extra={
                    "monotonic_time": 100.0,
                    "wall_time": 200.0,
                    "fresh_pointcloud": True,
                    "replanned": True,
                    "control_dt": 0.05,
                    "ins_sequence": 99,
                },
            )
            self.assertTrue(written)
            latest_path = logger.latest_path
            logger.close()

            with open(latest_path, encoding="utf-8") as file_obj:
                records = [json.loads(line) for line in file_obj]
            cycle = next(
                item for item in records
                if item["record_type"] == "cycle"
            )
            self.assertEqual("session-1", cycle["session_id"])
            self.assertEqual(99, cycle["ego"]["ins_sequence"])
            self.assertEqual("7", cycle["obstacles"][0]["id"])
            self.assertAlmostEqual(
                3.0, cycle["obstacles"][0]["ego_longitudinal"]
            )
            self.assertAlmostEqual(
                2.0, cycle["obstacles"][0]["ego_lateral"]
            )
            self.assertEqual(
                {"s": 23.0, "d": 2.0, "gap": 3.0},
                cycle["obstacles"][0]["frenet"],
            )
            self.assertEqual(
                "AVOID_LEFT", cycle["plan"]["behavior"]
            )
            self.assertLessEqual(
                cycle["plan"]["trajectory"]["sample_count"],
                10,
            )
            self.assertEqual(8.0, cycle["control"]["command"]["steer"])

    def test_rate_limit_is_bypassed_by_emergency(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = DriveTraceLogger(output_dir, period_sec=1.0)
            logger.start_session("session-2", "map.xodr")
            plan = Object(
                trajectory=None,
                target_speed=2.0,
                behavior="KEEP_LANE",
                emergency=False,
                reason="",
            )
            common = {
                "loop_count": 1,
                "ego": Object(x=0.0, y=0.0, theta=0.0, speed=1.0),
                "obstacles": [],
                "plan_result": plan,
                "planner_debug": {},
                "control_command": Object(acc=0.0, speed=1.0, steer=0.0),
                "controller_debug": {},
            }
            self.assertTrue(
                logger.record_cycle(
                    **common,
                    extra={"monotonic_time": 10.0},
                )
            )
            self.assertFalse(
                logger.record_cycle(
                    **common,
                    extra={"monotonic_time": 10.1},
                )
            )
            plan.emergency = True
            plan.behavior = "EMERGENCY"
            self.assertTrue(
                logger.record_cycle(
                    **common,
                    extra={"monotonic_time": 10.2},
                )
            )
            logger.close()

    def test_latest_file_contains_only_latest_session(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = DriveTraceLogger(output_dir)
            logger.start_session("old-session", "old.xodr")
            logger.record_event("old-event")
            logger.start_session("new-session", "new.xodr")
            logger.record_event("new-event")
            latest_path = logger.latest_path
            archive_path = logger.archive_path
            logger.close()

            with open(latest_path, encoding="utf-8") as file_obj:
                latest_text = file_obj.read()
            with open(archive_path, encoding="utf-8") as file_obj:
                archive_text = file_obj.read()
            self.assertNotIn("old-session", latest_text)
            self.assertIn("new-session", latest_text)
            self.assertIn("old-session", archive_text)
            self.assertIn("new-session", archive_text)


if __name__ == "__main__":
    unittest.main()
