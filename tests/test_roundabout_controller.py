import math
import unittest
from types import SimpleNamespace

from roundabout_controller import RoundaboutController


def ego(x=0.0, y=0.0, speed=0.0, theta=0.0):
    return SimpleNamespace(
        x=x,
        y=y,
        theta=theta,
        speed=speed,
        length=4.0,
        width=1.8,
    )


def vehicle(identifier, x, y=0.0, speed=0.0, theta=0.0):
    return SimpleNamespace(
        id=str(identifier),
        x=x,
        y=y,
        theta=theta,
        speed=speed,
        speed_valid=True,
        world_vx=speed * math.cos(theta),
        world_vy=speed * math.sin(theta),
        length=4.0,
        width=1.8,
        obs_type=1,
        roleType="RoleType.VEHICLE",
    )


def straight_path():
    return {
        "x": [float(value) for value in range(0, 101)],
        "y": [0.0] * 101,
        "frame_id": "roundabout.xodr",
        "stamp": 1,
    }


def curved_path(radius=20.0):
    angles = [
        0.5 * math.pi * index / 100.0
        for index in range(101)
    ]
    return {
        "x": [radius * math.sin(value) for value in angles],
        "y": [
            radius * (1.0 - math.cos(value))
            for value in angles
        ],
        "kappa": [1.0 / radius] * len(angles),
        "frame_id": "roundabout.xodr",
        "stamp": 2,
    }


class RoundaboutControllerTest(unittest.TestCase):
    def test_clear_global_path_cruises(self):
        controller = RoundaboutController(
            mode="follow", max_speed=30.0
        )
        plan = controller.plan(ego(), [], straight_path())

        self.assertEqual(plan.behavior, "ROUNDABOUT_CRUISE")
        self.assertEqual(plan.target_speed, 30.0)
        self.assertIsNone(controller.locked_lead_id)

    def test_nearest_path_vehicle_is_selected_each_cycle(self):
        controller = RoundaboutController(mode="follow")
        first = controller.plan(
            ego(),
            [vehicle("near", 7.0), vehicle("far", 14.0)],
            straight_path(),
        )
        second = controller.plan(
            ego(),
            [vehicle("near", 12.0), vehicle("far", 5.0)],
            straight_path(),
        )

        self.assertEqual(first.behavior, "ROUNDABOUT_FOLLOW")
        self.assertEqual(second.behavior, "ROUNDABOUT_FOLLOW")
        self.assertEqual(
            controller.last_debug["lead"]["id"], "far"
        )

    def test_path_vehicle_gate_uses_absolute_d_below_one_metre(self):
        controller = RoundaboutController(
            mode="follow", lane_half_width=1.0
        )
        controller.plan(
            ego(),
            [
                vehicle("outside", 5.0, y=1.01),
                vehicle("inside", 8.0, y=0.99),
            ],
            straight_path(),
        )

        self.assertEqual(
            controller.last_debug["lead"]["id"], "inside"
        )
        self.assertLess(
            abs(controller.last_debug["lead"]["d"]), 1.0
        )

    def test_vehicle_outside_path_corridor_restores_cruise(self):
        controller = RoundaboutController(mode="follow")
        plan = controller.plan(
            ego(),
            [vehicle("other_lane", 7.0, y=2.0)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_CRUISE")
        self.assertEqual(plan.target_speed, controller.max_speed)

    def test_opposite_direction_vehicle_is_not_followed(self):
        controller = RoundaboutController(mode="follow")
        plan = controller.plan(
            ego(),
            [vehicle("wrong_way", 5.0, theta=math.pi)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_CRUISE")
        self.assertIsNone(controller.locked_lead_id)

    def test_one_metre_bumper_gap_is_idm_minimum_gap(self):
        controller = RoundaboutController(
            mode="follow", desired_gap=1.0
        )
        plan = controller.plan(
            ego(speed=8.0),
            [vehicle("lead", 5.0, speed=8.0)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_FOLLOW")
        self.assertAlmostEqual(
            controller.last_debug["lead"]["gap"], 1.0
        )

    def test_relative_speed_causes_earlier_idm_deceleration(self):
        closing_controller = RoundaboutController(mode="follow")
        matched_controller = RoundaboutController(mode="follow")
        closing = closing_controller.control(
            ego(speed=10.0),
            [vehicle("lead", 14.0, speed=2.0)],
            straight_path(),
            0.02,
        )
        matched = matched_controller.control(
            ego(speed=10.0),
            [vehicle("lead", 14.0, speed=10.0)],
            straight_path(),
            0.02,
        )

        self.assertLess(closing.acc, matched.acc)
        self.assertGreater(
            closing_controller.last_debug[
                "desired_dynamic_gap"
            ],
            matched_controller.last_debug[
                "desired_dynamic_gap"
            ],
        )

    def test_ttc_is_emergency_braking_fallback(self):
        controller = RoundaboutController(
            mode="follow",
            ttc_emergency=1.0,
            max_decel=15.5,
        )
        command = controller.control(
            ego(speed=10.0),
            [vehicle("lead", 6.0, speed=0.0)],
            straight_path(),
            0.02,
        )

        self.assertEqual(command.acc, -15.5)
        self.assertEqual(command.speed, 0.0)
        self.assertTrue(
            controller.last_debug["ttc_emergency_active"]
        )
        self.assertEqual(
            controller.last_plan.behavior,
            "ROUNDABOUT_TTC_BRAKE",
        )

    def test_curve_speed_caps_cruise_before_turn(self):
        controller = RoundaboutController(
            mode="follow",
            max_speed=30.0,
            curve_lateral_accel=4.0,
        )
        plan = controller.plan(
            ego(speed=5.0), [], curved_path(radius=20.0)
        )

        self.assertAlmostEqual(
            plan.target_speed, math.sqrt(4.0 / 0.05), delta=0.1
        )
        self.assertLess(plan.target_speed, 30.0)

    def test_constant_speed_lead_is_followed_without_collision(self):
        controller = RoundaboutController(
            mode="follow",
            desired_gap=1.0,
            time_headway=0.8,
            max_speed=30.0,
            follow_max_accel=8.0,
        )
        dt = 0.02
        ego_state = ego()
        lead_x = 15.0
        minimum_gap = float("inf")
        maximum_accel = 0.0
        for _ in range(500):
            command = controller.control(
                ego_state,
                [vehicle("lead", lead_x, speed=3.0)],
                straight_path(),
                dt,
            )
            maximum_accel = max(maximum_accel, command.acc)
            ego_state.speed = max(
                0.0, ego_state.speed + command.acc * dt
            )
            ego_state.x += ego_state.speed * dt
            lead_x += 3.0 * dt
            if controller.last_debug.get("lead") is not None:
                minimum_gap = min(
                    minimum_gap,
                    controller.last_debug["lead"]["gap"],
                )

        self.assertLessEqual(maximum_accel, 8.0)
        self.assertGreater(minimum_gap, 0.5)
        self.assertAlmostEqual(ego_state.speed, 3.0, delta=0.5)

    def test_direct_mode_ignores_traffic(self):
        controller = RoundaboutController(
            mode="direct", max_speed=24.0
        )
        plan = controller.plan(
            ego(),
            [vehicle("blocking", 2.5)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_DIRECT")
        self.assertEqual(plan.target_speed, 24.0)
        self.assertIsNone(controller.locked_lead_id)

    def test_lateral_control_returns_to_global_path(self):
        controller = RoundaboutController(mode="direct")
        command = controller.control(
            ego(y=1.0, speed=8.0),
            [],
            straight_path(),
            0.05,
        )

        self.assertLess(command.steer, 0.0)
        self.assertTrue(
            controller.last_debug["roundabout_exclusive"]
        )


if __name__ == "__main__":
    unittest.main()
