import math
import os
import sys
import json
import tempfile
import unittest

import numpy as np


SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from rule_based_planner import (
    PlanResult,
    PlannerConfig,
    RuleBasedPlanner,
    StableController,
    Trajectory,
    build_direct_sprint_route,
)


class Object(object):
    def __init__(self, **values):
        self.__dict__.update(values)


def straight_path(stamp=1):
    return {
        "x": [float(index) for index in range(121)],
        "y": [0.0] * 121,
        "frame_id": "intersection.xodr",
        "stamp": stamp,
    }


def ego(x=0.0, y=0.0, speed=2.0):
    return Object(
        x=x,
        y=y,
        theta=0.0,
        speed=speed,
        length=4.6,
        width=1.9,
    )


def obstacle(x, y=0.0, speed=0.0):
    return Object(
        id="1",
        x=x,
        y=y,
        theta=0.0,
        speed=speed,
        speed_valid=True,
        world_vx=speed,
        world_vy=0.0,
        length=4.5,
        width=2.0,
        roleType="RoleType.VEHICLE",
        obs_type=1,
        track_hits=5,
        track_misses=0,
        track_predicted=False,
    )


class RulePlannerTest(unittest.TestCase):
    def test_direct_sprint_route_preserves_opposite_lane_goal(self):
        route = build_direct_sprint_route(
            {"x": 103.64, "y": 5.17},
            {"x": -1.74, "y": 3.5},
            "MT_19-non-motorized.xodr",
            speed_limit=30.0,
        )

        self.assertTrue(route["_sprint_direct_fallback"])
        self.assertAlmostEqual(route["x"][0], 103.64)
        self.assertAlmostEqual(route["y"][0], 5.17)
        self.assertAlmostEqual(route["x"][-1], -1.74)
        self.assertAlmostEqual(route["y"][-1], 3.5)
        self.assertGreater(len(route["x"]), 100)
        self.assertEqual(
            len(route["speed_limit"]), len(route["x"])
        )

    def test_enabled_scenario_override_forces_d_v_and_ignores_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            override_path = os.path.join(
                directory, "scenario_overrides.json"
            )
            with open(override_path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "scenarios": [
                            {
                                "name": "manual-test",
                                "enabled": True,
                                "map": "intersection.xodr",
                                "s_start": 0.0,
                                "s_end": 1000.0,
                                "target_d": 1.2,
                                "target_speed_kmh": 72.0,
                            }
                        ]
                    },
                    stream,
                )
            config = PlannerConfig()
            config.scenario_overrides_path = override_path
            planner = RuleBasedPlanner(config)
            blocking_vehicle = obstacle(
                3.0, y=0.0, speed=0.0
            )

            result = planner.plan(
                ego(speed=4.0),
                [blocking_vehicle],
                straight_path(),
                "intersection.xodr",
            )

            self.assertFalse(result.emergency)
            self.assertEqual("MANUAL_OVERRIDE", result.behavior)
            self.assertAlmostEqual(20.0, result.target_speed)
            self.assertAlmostEqual(
                1.2, float(result.trajectory.d[-1]), places=2
            )
            self.assertTrue(
                planner.last_debug["manual_control_active"]
            )
            self.assertTrue(
                planner.last_debug["manual_collision_bypass"]
            )
            self.assertEqual(
                0.0, planner.last_debug["manual_s_start"]
            )
            self.assertEqual(
                1000.0, planner.last_debug["manual_s_end"]
            )
            self.assertEqual(
                1, planner.last_debug["detected_obstacle_count"]
            )
            self.assertEqual(
                0, planner.last_debug["planning_obstacle_count"]
            )

    def test_disabled_scenario_override_leaves_collision_safety_active(self):
        config = PlannerConfig()
        config.scenario_overrides_path = os.path.join(
            SAMPLE_DIR, "scenario_overrides.json"
        )
        planner = RuleBasedPlanner(config)
        result = planner.plan(
            ego(speed=0.0),
            [obstacle(3.0, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertTrue(result.emergency)
        self.assertFalse(
            planner.last_debug["manual_control_active"]
        )

    def test_manual_behavior_tracks_nonzero_d_reference(self):
        controller = StableController()
        result = PlanResult(
            trajectory=self._straight_trajectory(speed=8.0),
            target_speed=8.0,
            behavior="MANUAL_OVERRIDE",
        )
        command = controller.control(
            ego(y=0.0, speed=5.0),
            result,
            0.1,
            path_lateral_offset=-1.0,
            path_reference_yaw=0.0,
            path_reference_curvature=0.0,
        )
        self.assertTrue(
            controller.last_debug["centerline_control_active"]
        )
        self.assertFalse(
            controller.last_debug[
                "trajectory_comfort_cap_applied"
            ]
        )
        self.assertGreater(command.steer, 0.0)

    def test_stationary_offset_ego_has_forward_candidate(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego=ego(y=-0.658, speed=0.0),
            obstacles=[],
            global_path=straight_path(),
            map_name="MT_48-urbanvillage.xodr",
            ego_lateral_speed=0.0,
        )
        self.assertFalse(result.emergency)
        self.assertGreater(result.target_speed, 0.0)
        self.assertIsNotNone(result.trajectory)
        self.assertLess(
            abs(float(result.trajectory.d[10])),
            abs(float(result.trajectory.d[0])),
        )
        self.assertGreater(
            float(result.trajectory.yaw[0]), 0.0
        )

    def test_free_road_produces_finite_keep_lane_trajectory(self):
        planner = RuleBasedPlanner()
        result = planner.plan(ego(), [], straight_path(), "intersection.xodr")
        self.assertFalse(result.emergency)
        self.assertEqual(result.behavior, "KEEP_LANE")
        self.assertIsNotNone(result.trajectory)
        self.assertGreater(len(result.trajectory.x), 10)
        self.assertAlmostEqual(float(result.trajectory.d[-1]), 0.0, places=3)
        self.assertGreater(result.target_speed, 2.0)
        self.assertTrue(planner.last_debug["direct_clear_path"])

    def test_clear_sharp_curve_at_standstill_uses_direct_path(self):
        radius = 13.4
        angles = np.linspace(0.0, 1.4, 160)
        path = {
            "x": (radius * np.sin(angles)).tolist(),
            "y": (radius * (1.0 - np.cos(angles))).tolist(),
            "frame_id": "sharp_curve.xodr",
            "stamp": 1,
        }
        planner = RuleBasedPlanner()
        current_ego = ego(x=0.0, y=-1.0, speed=0.0)
        current_ego.theta = 0.0
        result = planner.plan(
            current_ego, [], path, "sharp_curve.xodr"
        )
        self.assertFalse(result.emergency)
        self.assertEqual(result.behavior, "RECOVER")
        self.assertGreater(result.target_speed, 0.0)
        self.assertTrue(planner.last_debug["direct_clear_path"])
        command = StableController(planner.config).control(
            current_ego,
            result,
            0.1,
            path_lateral_offset=-1.0,
        )
        self.assertGreater(command.speed, 0.0)
        self.assertGreater(command.acc, 0.0)

    def test_curve_speed_factor_increases_curve_target_speed(self):
        radius = 20.0
        angles = np.linspace(0.0, 2.5, 240)
        path = {
            "x": (radius * np.sin(angles)).tolist(),
            "y": (
                radius * (1.0 - np.cos(angles))
            ).tolist(),
            "frame_id": "intersection.xodr",
            "stamp": 1,
        }
        slow_config = PlannerConfig()
        slow_config.curve_speed_factor = 0.75
        fast_config = PlannerConfig()
        fast_config.curve_speed_factor = 1.50
        slow = RuleBasedPlanner(slow_config).plan(
            ego(speed=2.0),
            [],
            path,
            "intersection.xodr",
        )
        fast = RuleBasedPlanner(fast_config).plan(
            ego(speed=2.0),
            [],
            path,
            "intersection.xodr",
        )
        self.assertGreater(
            fast.target_speed, slow.target_speed
        )

    def test_comfort_curve_cap_includes_current_sparse_path_position(self):
        radius = 25.0
        stations = np.linspace(0.0, 60.0, 31)
        angles = stations / radius
        path = {
            "x": (radius * np.sin(angles)).tolist(),
            "y": (
                radius * (1.0 - np.cos(angles))
            ).tolist(),
            "frame_id": "sparse_curve.xodr",
            "stamp": 1,
        }
        config = PlannerConfig()
        config.enable_comfort_mode()
        planner = RuleBasedPlanner(config)
        self.assertTrue(planner.reference.update(path))

        curve_limit = planner._curve_speed_limit(
            ego_s=11.0,
            base_limit=20.0,
            ego_speed=4.3,
        )

        expected = math.sqrt(
            config.max_lateral_accel / (1.0 / radius)
        )
        self.assertLessEqual(curve_limit, expected + 0.15)

    def test_map_specific_curve_lateral_budget(self):
        radius = 25.0
        stations = np.linspace(0.0, 60.0, 61)
        angles = stations / radius
        path = {
            "x": (radius * np.sin(angles)).tolist(),
            "y": (
                radius * (1.0 - np.cos(angles))
            ).tolist(),
            "frame_id": "curve.xodr",
            "stamp": 1,
        }
        config = PlannerConfig()
        config.enable_comfort_mode()
        config.map_curve_lateral_accel_overrides[
            "mt_05-intersection.xodr"
        ] = 2.5
        ordinary = RuleBasedPlanner(config)
        overridden = RuleBasedPlanner(config)
        self.assertTrue(ordinary.reference.update(path))
        self.assertTrue(overridden.reference.update(path))
        ordinary.map_name = "MT_06-other.xodr"
        overridden.map_name = "MT_05-intersection.xodr"

        ordinary_limit = ordinary._curve_speed_limit(
            11.0, 20.0, 4.0
        )
        overridden_limit = overridden._curve_speed_limit(
            11.0, 20.0, 4.0
        )

        self.assertFalse(
            ordinary._last_map_curve_override_active
        )
        self.assertTrue(overridden._last_map_curve_override_active)
        self.assertAlmostEqual(
            ordinary._last_curve_lateral_accel, 0.5
        )
        self.assertAlmostEqual(
            overridden._last_curve_lateral_accel, 2.5
        )
        self.assertGreater(overridden_limit, 1.8 * ordinary_limit)

    def test_selected_map_sprint_uses_heading_continuous_arc_to_goal(self):
        path = {
            "x": [0.0, 0.0, 5.0, 10.0],
            "y": [0.0, -5.0, -10.0, -10.0],
            "frame_id": "MT_05-intersection.xodr",
            "stamp": 1,
        }
        config = PlannerConfig()
        config.enable_comfort_mode()
        config.max_speed = 55.0
        config.override_map_speed_limit = True
        config.configure_sprint(
            enabled=True,
            map_names="MT_05-intersection.xodr",
        )
        planner = RuleBasedPlanner(config)
        current_ego = ego(speed=0.0)
        current_ego.theta = -math.pi / 2.0

        result = planner.plan(
            current_ego,
            [],
            path,
            "MT_05-intersection.xodr",
        )

        self.assertEqual(result.behavior, "SPRINT")
        self.assertAlmostEqual(
            result.target_speed, config.sprint_speed
        )
        self.assertTrue(
            planner.last_debug["sprint_active"]
        )
        self.assertEqual(
            planner.last_debug["sprint_goal"],
            [10.0, -10.0],
        )
        self.assertGreater(
            float(np.max(np.abs(planner.reference.kappa))), 0.05
        )
        self.assertAlmostEqual(planner.reference.x[0], 0.0)
        self.assertAlmostEqual(planner.reference.y[0], 0.0)
        arc_end = int(np.argmax(np.abs(planner.reference.kappa) < 1e-8))
        self.assertAlmostEqual(planner.reference.x[arc_end - 1], 10.0, places=2)
        self.assertAlmostEqual(planner.reference.y[arc_end - 1], -10.0, places=2)

    def test_selected_map_sprint_aligns_then_pulses_one_frame_and_coasts(self):
        path = {
            "x": [0.0, 0.0, 5.0, 10.0],
            "y": [0.0, -5.0, -10.0, -10.0],
            "frame_id": "MT_05-intersection.xodr",
            "stamp": 1,
        }
        config = PlannerConfig()
        config.enable_comfort_mode()
        config.max_speed = 55.0
        config.override_map_speed_limit = True
        config.configure_sprint(
            enabled=True,
            map_names="MT_05-intersection.xodr",
            alignment_duration=0.10,
        )
        planner = RuleBasedPlanner(config)
        controller = StableController(config)
        current_ego = ego(speed=0.0)
        current_ego.theta = -math.pi / 2.0
        result = planner.plan(
            current_ego,
            [],
            path,
            "MT_05-intersection.xodr",
        )
        projection = planner.reference.project(0.0, 0.0)

        align_1 = controller.control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )
        align_2 = controller.control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )
        pulse = controller.control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )
        coast_1 = controller.control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )
        coast_2 = controller.control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )

        self.assertEqual(align_1.acc, 0.0)
        self.assertEqual(align_1.speed, 0.0)
        self.assertEqual(align_2.acc, 0.0)
        self.assertEqual(align_2.speed, 0.0)
        self.assertGreater(align_1.steer, 0.0)
        self.assertLessEqual(
            abs(align_1.steer), config.sprint_max_steer_deg
        )
        self.assertEqual(pulse.acc, config.sprint_accel)
        self.assertEqual(pulse.speed, config.sprint_speed)
        self.assertEqual(coast_1.acc, 0.0)
        self.assertEqual(coast_2.acc, 0.0)
        self.assertTrue(
            controller.last_debug["sprint_mode"]
        )
        self.assertEqual(
            controller.last_debug["sprint_phase"], "COAST"
        )
        self.assertTrue(
            controller.last_debug["sprint_pulse_sent"]
        )

    def test_sprint_does_not_affect_unselected_maps(self):
        path = {
            "x": [0.0, 0.0, 5.0, 10.0],
            "y": [0.0, -5.0, -10.0, -10.0],
            "frame_id": "MT_06-other.xodr",
            "stamp": 1,
        }
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(speed=0.0), [], path, "MT_06-other.xodr"
        )

        self.assertNotEqual(result.behavior, "SPRINT")
        self.assertFalse(
            planner.last_debug["sprint_active"]
        )
        self.assertIsNone(planner._sprint_path)

    def test_generic_sprint_interface_enables_any_selected_map(self):
        path = {
            "x": [0.0, 8.0, 16.0],
            "y": [0.0, 4.0, 8.0],
            "frame_id": "custom_fast_map.xodr",
            "stamp": 1,
        }
        config = PlannerConfig().configure_sprint(
            enabled=True,
            map_names=["custom_fast_map.xodr"],
            speed=22.0,
            accel=31.0,
            max_steer_deg=9.0,
        )
        planner = RuleBasedPlanner(config)
        current_ego = ego(speed=0.0)
        current_ego.theta = 0.0

        result = planner.plan(
            current_ego, [], path, "custom_fast_map.xodr"
        )

        self.assertEqual(result.behavior, "SPRINT")
        self.assertAlmostEqual(result.target_speed, 22.0)
        self.assertTrue(planner.last_debug["sprint_active"])
        self.assertAlmostEqual(config.sprint_accel, 31.0)
        self.assertAlmostEqual(config.sprint_max_steer_deg, 9.0)

    def test_sprint_keeps_collision_checks_by_default(self):
        path = {
            "x": [0.0, 0.0, 5.0, 10.0],
            "y": [0.0, -5.0, -10.0, -10.0],
            "frame_id": "MT_05-intersection.xodr",
            "stamp": 1,
        }
        config = PlannerConfig().configure_sprint(
            enabled=True,
            map_names="MT_05-intersection.xodr",
        )
        planner = RuleBasedPlanner(config)
        blocking = obstacle(3.0, y=-3.0, speed=0.0)

        result = planner.plan(
            ego(speed=3.0),
            [blocking],
            path,
            "MT_05-intersection.xodr",
        )

        self.assertNotEqual(result.behavior, "SPRINT")
        self.assertEqual(
            planner.last_debug["detected_obstacle_count"], 1
        )
        self.assertFalse(
            planner.last_debug["ignore_obstacles"]
        )

    def test_generic_sprint_can_explicitly_bypass_obstacles(self):
        path = {
            "x": [0.0, 0.0, 5.0, 10.0],
            "y": [0.0, -5.0, -10.0, -10.0],
            "frame_id": "custom_fast_map.xodr",
            "stamp": 1,
        }
        config = PlannerConfig().configure_sprint(
            enabled=True,
            map_names="custom_fast_map.xodr",
            ignore_obstacles=True,
        )
        planner = RuleBasedPlanner(config)

        result = planner.plan(
            ego(speed=3.0),
            [obstacle(3.0, y=-3.0, speed=0.0)],
            path,
            "custom_fast_map.xodr",
        )

        self.assertEqual(result.behavior, "SPRINT")
        self.assertTrue(
            planner.last_debug["sprint_collision_bypass"]
        )
        self.assertEqual(
            planner.last_debug["detected_obstacle_count"], 1
        )
        self.assertEqual(
            planner.last_debug["planning_obstacle_count"], 0
        )

    def test_vehicle_restarts_after_pedestrian_leaves_clear_lane(self):
        planner = RuleBasedPlanner()
        current_ego = ego(speed=0.0)
        blocked = planner.plan(
            current_ego,
            [obstacle(4.0, y=0.0, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(
            planner.last_debug.get("direct_clear_path", False)
        )

        clear = planner.plan(
            current_ego,
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(clear.emergency)
        self.assertTrue(planner.last_debug["direct_clear_path"])
        command = StableController(planner.config).control(
            current_ego,
            clear,
            0.1,
            path_lateral_offset=0.0,
        )
        self.assertGreater(command.speed, 0.0)
        self.assertGreater(command.acc, 0.0)

    def test_ego_box_detection_does_not_block_clear_lane(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(speed=0.0),
            [obstacle(0.1, y=0.05, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertTrue(planner.last_debug["direct_clear_path"])
        self.assertEqual(
            planner.last_debug["planning_obstacle_count"], 0
        )

    def test_lead_vehicle_reduces_target_speed(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(speed=5.0),
            [obstacle(18.0, speed=1.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertIn(result.behavior, ("FOLLOW", "AVOID_LEFT", "AVOID_RIGHT"))
        self.assertLess(result.target_speed, 5.0)

    def test_static_vehicle_with_free_side_triggers_moving_avoidance(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(speed=2.0),
            [obstacle(16.0, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertIn(
            result.behavior, ("AVOID_LEFT", "AVOID_RIGHT")
        )
        self.assertGreater(result.target_speed, 0.0)
        self.assertGreater(float(result.trajectory.speed[-1]), 0.3)
        self.assertGreater(
            abs(float(result.trajectory.d[-1])), 2.5
        )
        self.assertTrue(planner.last_debug["static_avoidance"])
        self.assertGreater(
            planner.last_debug["avoidance_candidates"], 0
        )

    def test_roadside_vehicle_does_not_delay_clear_lane_launch(self):
        planner = RuleBasedPlanner()
        roadside = obstacle(15.0, y=-3.4, speed=0.0)
        roadside.theta = math.radians(171.0)
        result = planner.plan(
            ego(speed=0.0),
            [roadside],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertEqual("KEEP_LANE", result.behavior)
        self.assertIsNone(planner.last_debug["lead"])
        self.assertTrue(planner.last_debug["direct_clear_path"])
        self.assertNotIn(
            roadside.id,
            planner.last_debug["direct_lane_obstacle_ids"],
        )
        self.assertGreater(
            float(np.max(result.trajectory.accel)), 1.8
        )

    def test_side_box_is_not_promoted_to_static_lead(self):
        planner = RuleBasedPlanner()
        side = obstacle(14.0, y=-1.70, speed=0.0)
        result = planner.plan(
            ego(speed=2.0),
            [side],
            straight_path(),
            "intersection.xodr",
        )
        self.assertIsNone(planner.last_debug["lead"])
        self.assertTrue(
            planner.last_debug.get("static_avoidance", False)
        )
        self.assertTrue(planner.last_debug["side_bypass"])
        self.assertTrue(planner.last_debug["wide_avoidance"])
        self.assertGreater(result.target_speed, 0.0)
        self.assertLessEqual(result.target_speed, 20.0)
        self.assertEqual("AVOID_LEFT", result.behavior)
        self.assertGreaterEqual(
            float(result.trajectory.d[-1]),
            planner.last_debug["avoidance_origin_d"]
            + planner.config.minimum_bypass_shift
            - 1e-6,
        )

    def test_partial_right_blocker_uses_box_edge_clearance_and_holds_it(self):
        planner = RuleBasedPlanner()
        side = obstacle(9.6, y=-2.85, speed=0.0)
        side.theta = math.radians(171.0)
        first = planner.plan(
            ego(y=0.65, speed=0.5),
            [side],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(first.emergency)
        self.assertEqual("AVOID_LEFT", first.behavior)
        first_debug = dict(planner.last_debug)
        self.assertGreaterEqual(
            float(first.trajectory.d[-1]),
            side.y
            + first_debug["avoidance_required_offset"]
            - 1e-6,
        )

        second = planner.plan(
            ego(y=1.40, speed=1.0),
            [side],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(second.emergency)
        self.assertEqual("AVOID_LEFT", second.behavior)
        self.assertGreaterEqual(
            float(second.trajectory.d[-1]),
            side.y
            + planner.last_debug["avoidance_required_offset"]
            - 1e-6,
        )
        self.assertAlmostEqual(
            0.65,
            planner.last_debug["avoidance_origin_d"],
            places=2,
        )

    def test_standstill_bypass_uses_spatial_candidate_past_parked_row(self):
        planner = RuleBasedPlanner()
        first = obstacle(8.5, y=-2.87, speed=0.0)
        first.id = "parked_1"
        first.theta = 0.52
        second = obstacle(26.0, y=-2.10, speed=0.0)
        second.id = "parked_2"
        second.theta = 0.20
        result = planner.plan(
            ego(y=0.67, speed=0.0),
            [first, second],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertEqual("AVOID_LEFT", result.behavior)
        self.assertGreater(
            float(result.trajectory.speed[-1]), 0.3
        )
        self.assertTrue(
            planner.last_debug["spatial_bypass_targets"]
        )
        self.assertGreater(
            planner.last_debug["avoidance_candidates"], 0
        )

    def test_ignore_obstacles_uses_direct_clear_path(self):
        config = PlannerConfig()
        config.ignore_obstacles = True
        planner = RuleBasedPlanner(config)
        result = planner.plan(
            ego(speed=5.0),
            [obstacle(4.0, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertTrue(planner.last_debug["direct_clear_path"])
        self.assertTrue(planner.last_debug["ignore_obstacles"])
        self.assertEqual(
            planner.last_debug["detected_obstacle_count"], 1
        )
        self.assertEqual(
            planner.last_debug["planning_obstacle_count"], 0
        )

    def test_max_speed_and_acceleration_are_configurable(self):
        config = PlannerConfig()
        config.max_speed = 2.5
        config.max_accel = 0.6
        planner = RuleBasedPlanner(config)
        result = planner.plan(
            ego(speed=0.0),
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertLessEqual(result.target_speed, 2.5 + 1e-6)
        self.assertLessEqual(
            float(np.max(result.trajectory.accel)), 0.6 + 1e-6
        )
        controller = StableController(config)
        for _ in range(20):
            command = controller.control(
                ego(speed=0.0), result, 0.1
            )
        self.assertLessEqual(command.acc, 0.6 + 1e-6)

    def test_map_speed_limit_can_be_explicitly_overridden(self):
        config = PlannerConfig()
        config.max_speed = 40.0 / 3.6
        config.override_map_speed_limit = True
        planner = RuleBasedPlanner(config)
        result = planner.plan(
            ego(speed=5.0),
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertAlmostEqual(40.0 / 3.6, result.target_speed, places=3)
        self.assertTrue(
            planner.last_debug["speed_limits"][
                "override_map_speed_limit"
            ]
        )

    def test_shorter_follow_settings_raise_follow_target(self):
        lead = obstacle(18.0, speed=1.0)
        conservative = PlannerConfig()
        conservative.time_headway = 1.5
        conservative.minimum_gap = 4.0
        close = PlannerConfig()
        close.time_headway = 0.8
        close.minimum_gap = 2.0
        conservative_result = RuleBasedPlanner(conservative).plan(
            ego(speed=5.0),
            [lead],
            straight_path(),
            "intersection.xodr",
        )
        close_result = RuleBasedPlanner(close).plan(
            ego(speed=5.0),
            [lead],
            straight_path(),
            "intersection.xodr",
        )
        self.assertGreater(
            close_result.target_speed,
            conservative_result.target_speed,
        )

    def test_global_path_speed_cap_is_applied(self):
        config = PlannerConfig()
        config.respect_path_speed_limit = True
        planner = RuleBasedPlanner(config)
        path = straight_path()
        path["speed_limit"] = [2.5] * len(path["x"])
        result = planner.plan(ego(speed=1.0), [], path, "intersection.xodr")
        self.assertFalse(result.emergency)
        self.assertLessEqual(result.target_speed, 2.5 + 1e-6)

    def test_global_path_speed_cap_is_ignored_by_default(self):
        planner = RuleBasedPlanner()
        path = straight_path()
        path["speed_limit"] = [2.5] * len(path["x"])
        result = planner.plan(
            ego(speed=1.0), [], path, "intersection.xodr"
        )
        self.assertFalse(result.emergency)
        self.assertGreater(result.target_speed, 2.5)
        self.assertFalse(
            planner.last_debug["speed_limits"][
                "respect_path_speed_limit"
            ]
        )

    def test_endpoint_does_not_brake_or_stop_by_default(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(x=123.0, speed=5.0),
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertGreaterEqual(result.target_speed, 5.0)
        self.assertGreater(float(result.trajectory.x[-1]), 123.0)
        self.assertFalse(
            planner.last_debug["speed_limits"]["stop_at_goal"]
        )

    def test_nonstopping_route_continues_past_endpoint(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        config.sprint_enabled = False
        planner = RuleBasedPlanner(config)
        current_ego = ego(x=123.0, speed=5.0)
        result = planner.plan(
            current_ego,
            [],
            straight_path(),
            "MT_05-intersection.xodr",
        )
        projection = planner.reference.project(
            current_ego.x, current_ego.y
        )
        command = StableController(config).control(
            current_ego,
            result,
            0.05,
            path_lateral_offset=projection["d"],
            path_reference_yaw=projection["yaw"],
            path_reference_curvature=projection["kappa"],
        )

        self.assertFalse(result.emergency)
        self.assertGreaterEqual(result.target_speed, 5.0)
        self.assertGreater(command.speed, 5.0)
        self.assertGreater(command.acc, 0.0)
        self.assertGreater(float(result.trajectory.x[-1]), 123.0)

    def test_endpoint_stop_can_be_restored(self):
        config = PlannerConfig()
        config.stop_at_goal = True
        planner = RuleBasedPlanner(config)
        result = planner.plan(
            ego(x=119.5, speed=2.0),
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertEqual(result.target_speed, 0.0)

    def test_close_stopped_vehicle_requests_emergency(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(speed=6.0),
            [obstacle(4.0, speed=0.0)],
            straight_path(),
            "intersection.xodr",
        )
        self.assertTrue(result.emergency)
        self.assertEqual(result.target_speed, 0.0)
        self.assertGreater(
            result.trajectory.x[-1] - result.trajectory.x[0],
            0.5,
        )

    def test_controller_limits_steering_rate_and_acceleration_jerk(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego(y=0.8, speed=4.0), [], straight_path(), "intersection.xodr"
        )
        controller = StableController(planner.config)
        first = controller.control(ego(y=0.8, speed=4.0), result, 0.05)
        second = controller.control(ego(y=0.8, speed=4.0), result, 0.05)
        max_delta = planner.config.steering_rate_low * 0.05 + 1e-6
        self.assertLessEqual(abs(first.steer), max_delta)
        self.assertLessEqual(abs(second.steer - first.steer), max_delta)
        self.assertLessEqual(
            first.acc,
            planner.config.max_lon_jerk * 0.05 + 1e-6,
        )

    def test_configured_acceleration_is_used_on_clear_direct_path(self):
        config = PlannerConfig()
        config.max_accel = 5.0
        planner = RuleBasedPlanner(config)
        current_ego = ego(speed=0.0)
        result = planner.plan(
            current_ego,
            [],
            straight_path(),
            "intersection.xodr",
        )
        self.assertTrue(planner.last_debug["direct_clear_path"])
        self.assertGreater(
            float(np.max(result.trajectory.accel)), 4.5
        )
        controller = StableController(config)
        commands = [
            controller.control(current_ego, result, 0.05)
            for _ in range(10)
        ]
        self.assertGreater(commands[-1].acc, 4.5)

    def test_offset_without_obstacle_is_recovery_not_avoidance(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego=ego(y=-0.627, speed=0.0),
            obstacles=[],
            global_path=straight_path(),
            map_name="intersection.xodr",
        )
        self.assertEqual(result.behavior, "RECOVER")
        self.assertNotIn(result.behavior, ("AVOID_LEFT", "AVOID_RIGHT"))

    def test_ego_outside_corridor_can_plan_inward_recovery(self):
        planner = RuleBasedPlanner()
        result = planner.plan(
            ego=ego(y=-1.9, speed=2.0),
            obstacles=[],
            global_path=straight_path(),
            map_name="intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertEqual(result.behavior, "RECOVER")
        self.assertGreaterEqual(
            float(result.trajectory.d[-1]),
            planner.config.road_right_bound - 1e-6,
        )

    @staticmethod
    def _straight_trajectory(speed=4.0):
        t = np.array([0.0, 1.0, 2.0])
        zeros = np.zeros(3)
        trajectory = Trajectory(
            t=t,
            s=np.array([0.0, 5.0, 10.0]),
            d=zeros,
            speed=np.full(3, speed),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.array([0.0, 5.0, 10.0])
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = zeros.copy()
        return trajectory

    def test_lateral_controller_steers_toward_straight_path(self):
        controller = StableController()
        result = PlanResult(
            trajectory=self._straight_trajectory(),
            target_speed=4.0,
            behavior="KEEP_LANE",
        )
        command = controller.control(ego(y=0.8, speed=4.0), result, 0.1)
        self.assertLess(controller.last_debug["model_front_angle_deg"], 0.0)
        self.assertLess(command.steer, 0.0)
        self.assertEqual(controller.last_debug["steering_command_sign"], 1.0)

    def test_default_chassis_command_has_same_sign_as_model_angle(self):
        controller = StableController()
        result = PlanResult(
            trajectory=self._straight_trajectory(),
            target_speed=4.0,
            behavior="KEEP_LANE",
        )
        controller.control(ego(y=-0.8, speed=4.0), result, 0.1)
        model_angle = controller.last_debug["model_front_angle_deg"]
        raw_command = controller.last_debug["raw_steer"]
        self.assertGreater(model_angle, 0.0)
        self.assertGreater(raw_command, 0.0)

    def test_centerline_divergence_keeps_recovery_moving_by_default(self):
        controller = StableController()
        result = PlanResult(
            trajectory=self._straight_trajectory(),
            target_speed=4.0,
            behavior="RECOVER",
        )
        current_ego = ego(y=-0.8, speed=3.0)
        controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.70
        )
        controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.79
        )
        command = controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.88
        )
        self.assertFalse(
            controller.last_debug["centerline_safety_stop"]
        )
        self.assertGreater(command.speed, 0.0)
        self.assertGreater(
            controller.last_debug["centerline_front_feedback"], 0.0
        )

    def test_legacy_centerline_emergency_stop_can_be_enabled(self):
        config = PlannerConfig()
        config.centerline_safety_stop_enabled = True
        controller = StableController(config)
        result = PlanResult(
            trajectory=self._straight_trajectory(),
            target_speed=4.0,
            behavior="RECOVER",
        )
        current_ego = ego(y=-0.8, speed=3.0)
        controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.70
        )
        controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.79
        )
        command = controller.control(
            current_ego, result, 0.1, path_lateral_offset=-0.88
        )
        self.assertTrue(
            controller.last_debug["centerline_safety_stop"]
        )
        self.assertEqual(0.0, command.speed)

    def test_global_offset_feedback_works_when_local_error_is_zero(self):
        trajectory = self._straight_trajectory()
        trajectory.y[:] = -1.0
        controller = StableController()
        result = PlanResult(
            trajectory=trajectory,
            target_speed=4.0,
            behavior="RECOVER",
        )
        command = controller.control(
            ego(y=-1.0, speed=4.0),
            result,
            0.1,
            path_lateral_offset=-1.0,
        )
        self.assertAlmostEqual(
            0.0,
            controller.last_debug["lateral_error"],
            places=6,
        )
        self.assertGreater(
            controller.last_debug["centerline_front_feedback"],
            0.0,
        )
        self.assertGreater(command.steer, 0.0)

    def test_stationary_off_center_vehicle_can_restart_recovery(self):
        controller = StableController()
        result = PlanResult(
            trajectory=self._straight_trajectory(),
            target_speed=3.71,
            behavior="RECOVER",
        )
        command = controller.control(
            ego(y=-1.328, speed=0.0),
            result,
            0.1,
            path_lateral_offset=-1.328,
        )
        self.assertFalse(
            controller.last_debug["centerline_safety_stop"]
        )
        self.assertGreater(command.speed, 0.0)
        self.assertGreater(command.acc, 0.0)
        moving_command = controller.control(
            ego(y=-1.328, speed=0.2),
            result,
            0.1,
            path_lateral_offset=-1.328,
        )
        self.assertFalse(
            controller.last_debug["centerline_safety_stop"]
        )
        self.assertGreater(moving_command.speed, 0.0)

    def test_collision_envelope_rejects_diagonal_box_overlap(self):
        planner = RuleBasedPlanner()
        trajectory = self._straight_trajectory(speed=1.0)
        current_ego = {
            "length": 4.6,
            "width": 1.9,
        }
        parked_vehicle = {
            "x": 4.5,
            "y": 2.2,
            "vx": 0.0,
            "vy": 0.0,
            "length": 4.5,
            "width": 2.0,
            "pedestrian": False,
        }
        self.assertFalse(
            planner._collision_free(
                trajectory, current_ego, [parked_vehicle]
            )
        )

    def test_collision_check_uses_rotated_obstacle_envelope(self):
        planner = RuleBasedPlanner()
        trajectory = self._straight_trajectory(speed=1.0)
        current_ego = {"length": 4.6, "width": 1.9}
        rotated_vehicle = {
            "id": "rotated",
            "x": 5.0,
            "y": 3.0,
            "vx": 0.0,
            "vy": 0.0,
            "yaw": math.pi / 2.0,
            "length": 4.5,
            "width": 2.0,
            "pedestrian": False,
            "speed_valid": True,
            "track_predicted": False,
        }
        self.assertFalse(
            planner._collision_free(
                trajectory, current_ego, [rotated_vehicle]
            )
        )
        self.assertEqual(
            "rotated", trajectory.closest_obstacle_id
        )

    def test_dense_collision_sampling_prevents_tunnelling(self):
        planner = RuleBasedPlanner()
        zeros = np.zeros(2)
        trajectory = Trajectory(
            t=np.array([0.0, 1.0]),
            s=np.array([0.0, 20.0]),
            d=zeros,
            speed=np.full(2, 20.0),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.array([0.0, 20.0])
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = zeros.copy()
        crossing_vehicle = {
            "id": "middle",
            "x": 10.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "yaw": 0.0,
            "length": 4.5,
            "width": 2.0,
            "pedestrian": False,
            "speed_valid": True,
            "track_predicted": False,
        }
        self.assertFalse(
            planner._collision_free(
                trajectory,
                {"length": 4.6, "width": 1.9},
                [crossing_vehicle],
            )
        )
        self.assertAlmostEqual(
            0.5,
            trajectory.closest_collision_time,
            delta=0.10,
        )

    def test_static_bypass_does_not_merge_into_side_vehicle(self):
        planner = RuleBasedPlanner()
        lead = obstacle(16.0, y=0.0, speed=0.0)
        side_vehicle = obstacle(0.0, y=3.5, speed=2.0)
        side_vehicle.id = "side"
        result = planner.plan(
            ego(speed=2.0),
            [lead, side_vehicle],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertEqual("AVOID_RIGHT", result.behavior)
        self.assertLess(float(result.trajectory.d[-1]), -2.5)
        self.assertGreater(float(result.trajectory.speed[-1]), 0.3)

    def test_fast_rear_vehicle_is_retained_during_bypass(self):
        planner = RuleBasedPlanner()
        lead = obstacle(16.0, y=0.0, speed=0.0)
        fast_rear = obstacle(-30.0, y=3.5, speed=10.0)
        fast_rear.id = "fast-rear"
        result = planner.plan(
            ego(speed=2.0),
            [lead, fast_rear],
            straight_path(),
            "intersection.xodr",
        )
        self.assertEqual(
            2, planner.last_debug["planning_obstacle_count"]
        )
        self.assertIn(
            result.behavior, ("AVOID_LEFT", "AVOID_RIGHT")
        )
        self.assertFalse(result.emergency)
        self.assertTrue(
            planner._collision_free(
                result.trajectory,
                planner._ego_values(ego(speed=2.0), 0.0),
                planner._prepare_obstacles([lead, fast_rear]),
                avoidance_obstacle_id=lead.id,
            )
        )

    def test_adjacent_rear_vehicle_only_blocks_merge_toward_it(self):
        planner = RuleBasedPlanner()
        planner.plan(
            ego(speed=5.0), [], straight_path(), "intersection.xodr"
        )
        prepared = planner._prepare_obstacles(
            [obstacle(-10.0, y=4.9, speed=20.0)]
        )[0]
        current_ego = planner._ego_values(ego(speed=5.0), 0.0)
        keep = self._straight_trajectory(speed=5.0)
        self.assertTrue(
            planner._is_non_blocking_rear_follower(
                current_ego, prepared, trajectory=keep
            )
        )

        merge = self._straight_trajectory(speed=5.0)
        merge.d = np.array([0.0, 2.5, 4.9])
        merge.y = merge.d.copy()
        self.assertFalse(
            planner._is_non_blocking_rear_follower(
                current_ego, prepared, trajectory=merge
            )
        )
        self.assertFalse(
            planner._collision_free(
                merge, current_ego, [prepared]
            )
        )

    def test_same_lane_replay_rear_does_not_stop_clear_ego(self):
        planner = RuleBasedPlanner()
        rear = obstacle(-6.0, y=-1.30, speed=4.0)
        rear.id = "replay-rear"
        result = planner.plan(
            ego(speed=0.0),
            [rear],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertEqual("KEEP_LANE", result.behavior)
        self.assertGreater(result.target_speed, 0.0)
        self.assertTrue(planner.last_debug["direct_clear_path"])
        self.assertIn(
            "replay-rear",
            planner.last_debug["rear_non_blocking_ids"],
        )
        self.assertEqual(
            "replay-rear",
            planner.last_debug["rear_pressure"]["id"],
        )

    def test_rear_pressure_prefers_fastest_safe_avoidance(self):
        planner = RuleBasedPlanner()
        front = obstacle(16.0, y=0.0, speed=0.0)
        front.id = "front"
        rear = obstacle(-7.0, y=-1.0, speed=8.0)
        rear.id = "replay-rear"
        result = planner.plan(
            ego(speed=2.0),
            [front, rear],
            straight_path(),
            "intersection.xodr",
        )
        self.assertFalse(result.emergency)
        self.assertIn(
            result.behavior, ("AVOID_LEFT", "AVOID_RIGHT")
        )
        self.assertEqual(
            "REAR_PRESSURE_FASTEST_SAFE",
            planner.last_debug["candidate_selection_mode"],
        )

    def test_front_collision_remains_hard_with_replay_rear(self):
        planner = RuleBasedPlanner()
        front = obstacle(3.0, y=0.0, speed=0.0)
        front.id = "front"
        rear = obstacle(-6.0, y=0.0, speed=4.0)
        rear.id = "replay-rear"
        result = planner.plan(
            ego(speed=0.0),
            [front, rear],
            straight_path(),
            "intersection.xodr",
        )
        self.assertTrue(result.emergency)
        self.assertEqual(0.0, result.target_speed)
        self.assertFalse(
            planner.last_debug.get("direct_clear_path", False)
        )
        self.assertIn(
            "replay-rear",
            planner.last_debug["rear_non_blocking_ids"],
        )

    def test_large_heading_error_caps_speed(self):
        config = PlannerConfig()
        config.strict_alignment_speed_guard = True
        controller = StableController(config)
        current_ego = ego(speed=4.0)
        current_ego.theta = np.deg2rad(20.0)
        result = PlanResult(
            trajectory=self._straight_trajectory(speed=5.0),
            target_speed=5.0,
            behavior="KEEP_LANE",
        )
        command = controller.control(current_ego, result, 0.1)
        self.assertLessEqual(command.speed, 0.8 + 1e-6)
        self.assertLess(command.acc, 0.0)

    def test_aggressive_alignment_guard_does_not_crawl_at_20_degrees(self):
        controller = StableController()
        current_ego = ego(speed=4.0)
        current_ego.theta = np.deg2rad(20.0)
        result = PlanResult(
            trajectory=self._straight_trajectory(speed=5.0),
            target_speed=5.0,
            behavior="KEEP_LANE",
        )
        command = controller.control(current_ego, result, 0.1)
        self.assertGreaterEqual(command.speed, 5.0)
        self.assertGreater(command.acc, 0.0)

    def test_straight_tracking_converges_with_steering_actuator_lag(self):
        count = 201
        zeros = np.zeros(count)
        trajectory = Trajectory(
            t=np.linspace(0.0, 20.0, count),
            s=np.linspace(0.0, 100.0, count),
            d=zeros,
            speed=np.full(count, 3.4),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.linspace(0.0, 100.0, count)
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = zeros.copy()
        result = PlanResult(
            trajectory=trajectory,
            target_speed=3.4,
            behavior="KEEP_LANE",
        )
        controller = StableController()
        current_ego = ego(speed=3.4)
        current_ego.theta = math.radians(8.0)
        actual_steer = 0.0
        maximum_lateral_error = 0.0

        for _ in range(200):
            command = controller.control(
                current_ego,
                result,
                0.05,
                steering_feedback=actual_steer,
            )
            requested_rate = (command.steer - actual_steer) / 0.25
            actual_steer += np.clip(requested_rate, -90.0, 90.0) * 0.05
            front_angle = math.radians(
                actual_steer
                / (
                    controller.config.steering_command_sign
                    * controller.config.steering_ratio
                )
            )
            current_ego.theta += (
                current_ego.speed
                / controller.config.controller_wheelbase
                * math.tan(front_angle)
                * 0.05
            )
            current_ego.x += (
                current_ego.speed * math.cos(current_ego.theta) * 0.05
            )
            current_ego.y += (
                current_ego.speed * math.sin(current_ego.theta) * 0.05
            )
            maximum_lateral_error = max(
                maximum_lateral_error, abs(current_ego.y)
            )

        self.assertLess(abs(math.degrees(current_ego.theta)), 0.2)
        self.assertLess(abs(current_ego.y), 0.05)
        self.assertLess(maximum_lateral_error, 0.5)

    def test_global_centerline_control_is_damped_at_high_speed(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        count = 1001
        zeros = np.zeros(count)
        trajectory = Trajectory(
            t=np.linspace(0.0, 20.0, count),
            s=np.linspace(0.0, 500.0, count),
            d=zeros,
            speed=np.full(count, 25.0),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.linspace(0.0, 500.0, count)
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = zeros.copy()
        result = PlanResult(
            trajectory=trajectory,
            target_speed=25.0,
            behavior="RECOVER",
        )
        current_ego = ego(y=1.0, speed=20.0)
        current_ego.theta = math.radians(1.5)
        actual_steer = 0.0
        offsets = []

        for _ in range(300):
            command = controller.control(
                current_ego,
                result,
                0.05,
                steering_feedback=actual_steer,
                path_lateral_offset=current_ego.y,
                path_reference_yaw=0.0,
                path_reference_curvature=0.0,
            )
            actual_steer += (
                (command.steer - actual_steer) / 0.15 * 0.05
            )
            front_angle = math.radians(
                actual_steer / config.steering_ratio
            )
            current_ego.theta += (
                current_ego.speed
                / config.controller_wheelbase
                * math.tan(front_angle)
                * 0.05
            )
            current_ego.x += (
                current_ego.speed
                * math.cos(current_ego.theta)
                * 0.05
            )
            current_ego.y += (
                current_ego.speed
                * math.sin(current_ego.theta)
                * 0.05
            )
            offsets.append(current_ego.y)

        self.assertTrue(
            controller.last_debug["centerline_control_active"]
        )
        self.assertLess(abs(offsets[-1]), 0.01)
        self.assertLess(max(offsets), 1.50)
        self.assertTrue(all(value > 0.0 for value in offsets))

    def test_curve_tracking_recovery_can_exceed_comfort_steer_cap(self):
        """A comfort threshold must not become a run-off-road limit."""
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        count = 101
        zeros = np.zeros(count)
        trajectory = Trajectory(
            t=np.linspace(0.0, 10.0, count),
            s=np.linspace(0.0, 50.0, count),
            d=zeros,
            speed=np.full(count, 4.3),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.linspace(0.0, 50.0, count)
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = np.full(count, 0.04)
        result = PlanResult(
            trajectory=trajectory,
            target_speed=4.3,
            behavior="RECOVER",
        )
        current_ego = ego(y=-1.3, speed=4.3)
        current_ego.theta = math.radians(-12.0)

        for _ in range(12):
            controller.control(
                current_ego,
                result,
                0.05,
                path_lateral_offset=-1.3,
                path_reference_yaw=0.0,
                path_reference_curvature=0.04,
            )

        self.assertTrue(
            controller.last_debug["tracking_recovery_active"]
        )
        self.assertTrue(
            controller.last_debug["curve_authority_active"]
        )
        self.assertGreater(
            controller.last_debug["active_lateral_accel_limit"],
            config.max_lateral_accel,
        )
        self.assertGreater(
            abs(controller.last_debug["estimated_lateral_accel"]),
            config.max_lateral_accel,
        )
        self.assertLessEqual(
            controller.last_debug["active_lateral_accel_limit"],
            config.max_tracking_lateral_accel,
        )

    def test_fast_curve_keeps_tracking_correction_reserve(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        config.map_curve_lateral_accel_overrides[
            "mt_05-intersection.xodr"
        ] = 2.5
        controller = StableController(config)
        curvature = 0.049
        curve_budget = (
            config.map_curve_lateral_accel_overrides[
                "mt_05-intersection.xodr"
            ]
        )
        speed = math.sqrt(curve_budget / curvature)
        stations = np.linspace(0.0, 80.0, 1601)
        angles = curvature * stations
        zeros = np.zeros_like(stations)
        trajectory = Trajectory(
            t=stations / speed,
            s=stations,
            d=zeros,
            speed=np.full_like(stations, speed),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.sin(angles) / curvature
        trajectory.y = (
            1.0 - np.cos(angles)
        ) / curvature
        trajectory.yaw = angles
        trajectory.kappa = np.full_like(
            stations, curvature
        )
        result = PlanResult(
            trajectory=trajectory,
            target_speed=speed,
            behavior="RECOVER",
        )
        current_ego = ego(y=0.685, speed=speed)
        current_ego.theta = math.radians(-1.4)
        actual_steer = 0.0
        offsets = []

        for _ in range(140):
            nearest = int(
                np.argmin(
                    (trajectory.x - current_ego.x) ** 2
                    + (trajectory.y - current_ego.y) ** 2
                )
            )
            dx = current_ego.x - trajectory.x[nearest]
            dy = current_ego.y - trajectory.y[nearest]
            path_yaw = trajectory.yaw[nearest]
            offset = (
                -math.sin(path_yaw) * dx
                + math.cos(path_yaw) * dy
            )
            command = controller.control(
                current_ego,
                result,
                0.05,
                steering_feedback=actual_steer,
                path_lateral_offset=offset,
                path_reference_yaw=path_yaw,
                path_reference_curvature=curvature,
            )
            actual_steer += (
                (command.steer - actual_steer) / 0.15 * 0.05
            )
            front_angle = math.radians(
                actual_steer / config.steering_ratio
            )
            current_ego.theta += (
                current_ego.speed
                / config.controller_wheelbase
                * math.tan(front_angle)
                * 0.05
            )
            current_ego.x += (
                current_ego.speed
                * math.cos(current_ego.theta)
                * 0.05
            )
            current_ego.y += (
                current_ego.speed
                * math.sin(current_ego.theta)
                * 0.05
            )
            offsets.append(offset)

        self.assertGreater(
            config.max_tracking_lateral_accel,
            curve_budget,
        )
        self.assertLess(max(abs(value) for value in offsets), 0.75)
        self.assertLess(abs(offsets[-1]), 0.10)
        self.assertTrue(
            math.isinf(
                controller.last_debug["alignment_speed_cap"]
            )
        )

    def test_cam6_steering_ratio_tracks_logged_merge_curve(self):
        """Reproduce the v6 MT_14 drift with the real chassis ratio.

        The logged episode starts at 24.653 m/s, d=0.009 m and an outward
        heading error of 0.177 degrees on a roughly -0.0005 1/m bend.  A
        controller ratio of 1.0 reaches more than four metres of offset when
        the chassis actually divides its steering-wheel command by 1.65.
        """
        config = PlannerConfig()
        config.enable_comfort_mode()
        self.assertAlmostEqual(config.steering_ratio, 1.65)
        controller = StableController(config)
        count = 1001
        zeros = np.zeros(count)
        trajectory = Trajectory(
            t=np.linspace(0.0, 20.0, count),
            s=np.linspace(0.0, 500.0, count),
            d=zeros,
            speed=np.full(count, 40.0),
            accel=zeros,
            d_speed=zeros,
            d_accel=zeros,
            d_jerk=zeros,
        )
        trajectory.x = np.linspace(0.0, 500.0, count)
        trajectory.y = zeros.copy()
        trajectory.yaw = zeros.copy()
        trajectory.kappa = zeros.copy()
        result = PlanResult(
            trajectory=trajectory,
            target_speed=40.0,
            behavior="KEEP_LANE",
        )
        station = 7.8
        lateral_offset = 0.009
        vehicle_minus_path_yaw = math.radians(0.177)
        speed = 24.653
        steering_feedback = 0.0
        offsets = []

        for _ in range(400):
            path_curvature = (
                -0.0005 if station < 175.0 else 0.0
            )
            current_ego = ego(
                x=station, y=lateral_offset, speed=speed
            )
            current_ego.theta = vehicle_minus_path_yaw
            command = controller.control(
                current_ego,
                result,
                0.02,
                steering_feedback=steering_feedback,
                path_lateral_offset=lateral_offset,
                path_reference_yaw=0.0,
                path_reference_curvature=path_curvature,
            )
            steering_feedback += (
                (command.steer - steering_feedback)
                / 0.08
                * 0.02
            )
            actual_front_angle = math.radians(
                steering_feedback / 1.65
            )
            vehicle_minus_path_yaw += (
                speed
                / config.controller_wheelbase
                * math.tan(actual_front_angle)
                - speed * path_curvature
            ) * 0.02
            lateral_offset += (
                speed
                * math.sin(vehicle_minus_path_yaw)
                * 0.02
            )
            station += (
                speed
                * math.cos(vehicle_minus_path_yaw)
                * 0.02
            )
            speed = min(31.6, speed + 3.0 * 0.02)
            offsets.append(lateral_offset)

        self.assertLess(max(abs(value) for value in offsets), 0.20)
        self.assertLess(abs(offsets[-1]), 0.10)

    def test_stationary_emergency_does_not_leave_negative_acceleration(self):
        controller = StableController()
        stopped_ego = ego(speed=0.0)
        emergency = Object(
            trajectory=None,
            target_speed=0.0,
            emergency=True,
        )
        stopped = controller.control(stopped_ego, emergency, 0.05)
        self.assertEqual(stopped.acc, 0.0)
        self.assertEqual(controller.last_acc, 0.0)

        planner = RuleBasedPlanner(controller.config)
        drive_plan = planner.plan(
            ego=ego(y=-0.658, speed=0.0),
            obstacles=[],
            global_path=straight_path(),
            map_name="MT_48-urbanvillage.xodr",
            ego_lateral_speed=0.0,
        )
        drive = controller.control(
            ego(y=-0.658, speed=0.0), drive_plan, 0.05
        )
        self.assertGreaterEqual(drive.acc, 0.0)

    def test_forward_target_crosses_zero_with_jerk_limit_while_moving(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        controller.last_acc = -config.max_decel
        acc = controller._longitudinal_control(
            ego_speed=1.0,
            target_speed=5.0,
            dt=0.1,
            emergency=False,
        )
        self.assertAlmostEqual(
            -config.max_decel + config.max_lon_jerk * 0.1,
            acc,
        )
        self.assertLess(acc, 0.0)
        self.assertAlmostEqual(
            config.max_lon_jerk,
            controller.last_debug["actual_accel_command_jerk"],
        )
        self.assertFalse(
            controller.last_debug["stationary_stale_brake_cleared"]
        )

    def test_stationary_forward_target_clears_stale_brake(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        controller.last_acc = -config.max_decel
        acc = controller._longitudinal_control(
            ego_speed=0.0,
            target_speed=5.0,
            dt=0.1,
            emergency=False,
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(
            acc,
            config.max_lon_jerk * 0.1 + 1e-9,
        )
        self.assertTrue(
            controller.last_debug["stationary_stale_brake_cleared"]
        )

    def test_changing_forward_target_does_not_create_derivative_braking(self):
        controller = StableController()
        controller._longitudinal_control(
            ego_speed=0.0,
            target_speed=5.0,
            dt=0.02,
            emergency=False,
        )
        acc = controller._longitudinal_control(
            ego_speed=1.0,
            target_speed=1.1,
            dt=0.2,
            emergency=False,
        )
        self.assertGreaterEqual(acc, 0.0)

    def test_emergency_braking_cannot_cross_zero_in_one_cycle(self):
        controller = StableController()
        acc = controller._longitudinal_control(
            ego_speed=0.2,
            target_speed=0.0,
            dt=0.2,
            emergency=True,
        )
        self.assertGreaterEqual(acc, -0.800001)

    def test_planner_and_controller_publish_debug_signals(self):
        planner = RuleBasedPlanner()
        current_ego = ego(y=0.2, speed=2.0)
        result = planner.plan(
            current_ego, [], straight_path(), "intersection.xodr"
        )
        self.assertIn("projection", planner.last_debug)
        self.assertGreater(planner.last_debug["generated_candidates"], 0)
        self.assertGreater(planner.last_debug["accepted_candidates"], 0)
        self.assertIn("selected_target_d", planner.last_debug)

        controller = StableController(planner.config)
        controller.control(current_ego, result, 0.05)
        self.assertIn("lateral_error", controller.last_debug)
        self.assertIn("speed_error", controller.last_debug)
        self.assertIn("desired_acc", controller.last_debug)
        self.assertIn("output_steer", controller.last_debug)

    def test_comfort_mode_applies_evaluator_thresholds(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        self.assertTrue(config.comfort_mode)
        self.assertEqual(config.max_accel, 2.8)
        self.assertEqual(config.max_decel, 2.6)
        self.assertEqual(config.max_lon_jerk, 4.5)
        self.assertEqual(config.max_lat_accel, 0.5)
        self.assertEqual(config.max_lat_jerk, 1.0)
        self.assertEqual(config.max_lateral_accel, 0.5)
        self.assertEqual(config.max_yaw_rate, 0.5)

    def test_comfort_longitudinal_command_respects_accel_and_jerk(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        previous = 0.0
        for _ in range(12):
            command = controller._longitudinal_control(
                ego_speed=0.0,
                target_speed=40.0,
                dt=0.1,
                emergency=False,
            )
            self.assertLessEqual(command, 3.0 + 1e-9)
            self.assertLessEqual(
                abs(command - previous) / 0.1,
                6.0 + 1e-9,
            )
            previous = command

    def test_comfort_steering_command_respects_lateral_thresholds(self):
        config = PlannerConfig()
        config.enable_comfort_mode()
        controller = StableController(config)
        count = 20
        trajectory = Trajectory(
            np.arange(count) * 0.1,
            np.arange(count, dtype=float),
            np.zeros(count),
            np.full(count, 40.0),
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
        )
        trajectory.x = np.arange(count, dtype=float)
        trajectory.y = 0.02 * trajectory.x ** 2
        trajectory.yaw = np.arctan(
            0.04 * trajectory.x
        )
        trajectory.kappa = np.full(count, 0.04)
        result = PlanResult(
            trajectory=trajectory,
            target_speed=40.0,
            behavior="KEEP_LANE",
        )
        controller.control(
            ego(speed=40.0),
            result,
            0.1,
            path_lateral_offset=0.0,
        )
        self.assertLessEqual(
            abs(controller.last_debug["estimated_lateral_accel"]),
            0.5 + 1e-6,
        )
        self.assertLessEqual(
            abs(controller.last_debug["estimated_lateral_jerk"]),
            1.0 + 1e-6,
        )
        self.assertLessEqual(
            abs(controller.last_debug["estimated_yaw_rate"]),
            0.5 + 1e-6,
        )
        self.assertFalse(
            controller.last_debug[
                "trajectory_comfort_cap_applied"
            ]
        )
        self.assertEqual(
            controller.last_debug["trajectory_comfort_speed_cap"],
            40.0,
        )

        avoid_result = PlanResult(
            trajectory=trajectory,
            target_speed=40.0,
            behavior="AVOID_LEFT",
        )
        controller.control(
            ego(speed=40.0),
            avoid_result,
            0.1,
            path_lateral_offset=0.0,
        )
        self.assertTrue(
            controller.last_debug[
                "trajectory_comfort_cap_applied"
            ]
        )
        self.assertLess(
            controller.last_debug["trajectory_comfort_speed_cap"],
            40.0,
        )


if __name__ == "__main__":
    unittest.main()
