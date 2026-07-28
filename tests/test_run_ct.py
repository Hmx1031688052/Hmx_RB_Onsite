import unittest

from run_ct import _next_ct_speed_command


class CtSpeedCommandTest(unittest.TestCase):
    def test_slow_planning_cycle_uses_evaluator_interval(self):
        self.assertAlmostEqual(
            25.06,
            _next_ct_speed_command(
                current_speed=25.0,
                cruise_speed=40.0,
                acceleration=2.0,
                control_dt=0.208334,
                evaluator_interval=0.03,
            ),
        )

    def test_fast_cycle_keeps_actual_control_interval(self):
        self.assertAlmostEqual(
            25.04,
            _next_ct_speed_command(
                current_speed=25.0,
                cruise_speed=40.0,
                acceleration=2.0,
                control_dt=0.02,
                evaluator_interval=0.03,
            ),
        )

    def test_command_does_not_exceed_cruise_speed(self):
        self.assertEqual(
            25.03,
            _next_ct_speed_command(
                current_speed=25.0,
                cruise_speed=25.03,
                acceleration=2.0,
                control_dt=0.2,
                evaluator_interval=0.03,
            ),
        )


if __name__ == "__main__":
    unittest.main()
