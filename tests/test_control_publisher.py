import unittest

from control_publisher import FinalSpeedLimiter


class FinalSpeedLimiterTest(unittest.TestCase):
    def test_each_30ms_speed_step_is_limited_to_two_mps2(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(25.416958)

        speed, accel, previous = limiter.step(39.667)

        self.assertAlmostEqual(previous, 25.416958)
        self.assertAlmostEqual(speed, 25.476958)
        self.assertAlmostEqual(accel, 2.0)

    def test_fixed_step_does_not_use_slow_planning_interval(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(25.0)

        speed, accel, _ = limiter.step(40.0)

        self.assertAlmostEqual(speed, 25.06)
        self.assertAlmostEqual(accel, 2.0)

    def test_feedback_window_prevents_accumulated_target_lead(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(0.855)

        speed, accel, _ = limiter.step(39.667, ego_speed=0.855)

        self.assertAlmostEqual(speed, 0.915)
        self.assertAlmostEqual(accel, 2.0)

    def test_acceleration_continues_while_chassis_catches_speed_target(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(0.0)
        first_speed, first_accel, _ = limiter.step(
            39.667,
            ego_speed=0.0,
        )
        second_speed, second_accel, _ = limiter.step(
            39.667,
            ego_speed=0.0,
        )

        self.assertAlmostEqual(first_speed, 0.06)
        self.assertAlmostEqual(first_accel, 2.0)
        self.assertAlmostEqual(second_speed, 0.06)
        self.assertAlmostEqual(second_accel, 2.0)

    def test_same_episode_state_is_continuous_across_desired_modes(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(20.0)
        first_speed, _, _ = limiter.step(40.0)
        stop_speed, stop_accel, previous = limiter.step(0.0)

        self.assertAlmostEqual(first_speed, 20.06)
        self.assertAlmostEqual(previous, 20.06)
        self.assertAlmostEqual(stop_speed, 20.0)
        self.assertAlmostEqual(stop_accel, -2.0)

    def test_new_episode_reset_uses_new_initial_speed(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
        )
        limiter.reset(28.0)
        limiter.step(40.0)
        limiter.reset(20.0)

        speed, accel, previous = limiter.step(20.0)

        self.assertAlmostEqual(previous, 20.0)
        self.assertAlmostEqual(speed, 20.0)
        self.assertAlmostEqual(accel, 0.0)

    def test_missing_initial_speed_waits_for_ego_feedback(self):
        limiter = FinalSpeedLimiter()
        limiter.reset(None)

        self.assertIsNone(limiter.step(30.0, ego_speed=None))
        speed, _, previous = limiter.step(30.0, ego_speed=10.0)
        self.assertAlmostEqual(previous, 10.0)
        self.assertAlmostEqual(speed, 10.06)

    def test_one_shot_target_jumps_once_then_holds_zero_acceleration(self):
        limiter = FinalSpeedLimiter(
            safe_accel=2.0,
            publish_interval=0.03,
            one_shot_target=True,
        )
        limiter.reset(20.0)

        speed, accel, previous = limiter.step(35.0, ego_speed=20.0)
        held_speed, held_accel, held_previous = limiter.step(
            35.0,
            ego_speed=35.0,
        )

        self.assertAlmostEqual(previous, 20.0)
        self.assertAlmostEqual(speed, 35.0)
        self.assertAlmostEqual(accel, 500.0)
        self.assertAlmostEqual(held_previous, 35.0)
        self.assertAlmostEqual(held_speed, 35.0)
        self.assertAlmostEqual(held_accel, 0.0)

    def test_one_shot_uses_simulator_effective_interval(self):
        limiter = FinalSpeedLimiter(
            publish_interval=0.03,
            one_shot_target=True,
            one_shot_effective_interval=1.0 / 24.0,
        )
        limiter.reset(0.0)

        speed, accel, _ = limiter.step(23.8, ego_speed=0.0)

        self.assertAlmostEqual(speed, 23.8)
        self.assertAlmostEqual(accel, 571.2)


if __name__ == "__main__":
    unittest.main()
