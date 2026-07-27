import math
import unittest
from types import SimpleNamespace

from roundabout_controller import RoundaboutController


def ego(x=0.0, y=0.0, speed=0.0):
    return SimpleNamespace(
        x=x,
        y=y,
        theta=0.0,
        speed=speed,
        length=4.0,
        width=1.8,
    )


def vehicle(identifier, x, y=0.0, speed=0.0):
    return SimpleNamespace(
        id=str(identifier),
        x=x,
        y=y,
        theta=0.0,
        speed=speed,
        speed_valid=True,
        world_vx=speed,
        world_vy=0.0,
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


class RoundaboutControllerTest(unittest.TestCase):
    def test_follow_locks_nearest_same_lane_vehicle_once(self):
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
        self.assertEqual(controller.locked_lead_id, "near")
        self.assertEqual(second.behavior, "ROUNDABOUT_FOLLOW")
        self.assertEqual(
            controller.last_debug["lead"]["id"], "near"
        )

    def test_follow_uses_one_metre_bumper_gap(self):
        controller = RoundaboutController(
            mode="follow",
            desired_gap=1.0,
            max_speed=30.0,
        )
        plan = controller.plan(
            ego(speed=8.0),
            [vehicle("lead", 5.0, speed=8.0)],
            straight_path(),
        )

        self.assertAlmostEqual(
            controller.last_debug["lead"]["gap"], 1.0
        )
        self.assertAlmostEqual(plan.target_speed, 8.0)

    def test_follow_catches_gap_without_switching_target(self):
        controller = RoundaboutController(
            mode="follow",
            desired_gap=1.0,
            max_speed=30.0,
        )
        controller.plan(
            ego(),
            [vehicle("lead", 7.0)],
            straight_path(),
        )
        plan = controller.plan(
            ego(speed=4.0),
            [vehicle("lead", 12.0, speed=6.0)],
            straight_path(),
        )

        self.assertGreater(plan.target_speed, 6.0)
        self.assertEqual(controller.locked_lead_id, "lead")

    def test_follow_does_not_jump_to_speed_ceiling_for_large_gap(self):
        controller = RoundaboutController(
            mode="follow",
            desired_gap=1.0,
            max_speed=30.0,
            catchup_speed=6.0,
        )
        plan = controller.plan(
            ego(),
            [vehicle("lead", 15.0, speed=3.0)],
            straight_path(),
        )

        self.assertLessEqual(plan.target_speed, 9.0)

    def test_constant_speed_lead_is_approached_without_overshoot(self):
        controller = RoundaboutController(
            mode="follow",
            desired_gap=1.0,
            max_speed=30.0,
            max_accel=20.0,
            follow_max_accel=8.0,
        )
        dt = 0.02
        ego_state = ego()
        lead_x = 15.0
        minimum_gap = float("inf")
        maximum_accel = 0.0
        for _ in range(400):
            lead = vehicle("lead", lead_x, speed=3.0)
            command = controller.control(
                ego_state, [lead], straight_path(), dt
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

    def test_locked_lead_survives_lane_projection_departure(self):
        controller = RoundaboutController(mode="follow")
        controller.plan(
            ego(),
            [vehicle("locked", 7.0)],
            straight_path(),
        )
        plan = controller.plan(
            ego(),
            [vehicle("locked", 8.0, y=5.0, speed=2.0)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_FOLLOW")
        self.assertEqual(
            controller.last_debug["lead"]["id"], "locked"
        )

    def test_initial_lock_rejects_opposite_direction_vehicle(self):
        controller = RoundaboutController(mode="follow")
        wrong_way = vehicle("wrong", 5.0)
        wrong_way.theta = math.pi
        plan = controller.plan(
            ego(),
            [wrong_way, vehicle("lead", 8.0)],
            straight_path(),
        )

        self.assertEqual(plan.behavior, "ROUNDABOUT_FOLLOW")
        self.assertEqual(controller.locked_lead_id, "lead")

    def test_lost_locked_lead_does_not_relock_another_vehicle(self):
        controller = RoundaboutController(mode="follow")
        controller.plan(
            ego(),
            [vehicle("locked", 7.0)],
            straight_path(),
        )
        plan = controller.plan(
            ego(),
            [vehicle("other", 6.0)],
            straight_path(),
        )

        self.assertEqual(
            plan.behavior, "ROUNDABOUT_FOLLOW_WAIT"
        )
        self.assertEqual(plan.target_speed, 0.0)
        self.assertEqual(controller.locked_lead_id, "locked")

    def test_direct_mode_ignores_traffic_decisions(self):
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

    def test_waiting_at_standstill_does_not_steer(self):
        controller = RoundaboutController(mode="follow")
        command = controller.control(
            ego(y=-1.0, speed=0.0),
            [],
            straight_path(),
            0.05,
        )

        self.assertEqual(command.speed, 0.0)
        self.assertEqual(command.steer, 0.0)


if __name__ == "__main__":
    unittest.main()
