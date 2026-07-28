"""Final-stage speed shaping for fixed-period simulator control publication."""

import math


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class FinalSpeedLimiter:
    """Limit the speed field exactly where control messages are published."""

    def __init__(self, safe_accel=2.0, publish_interval=0.03):
        self.safe_accel = max(0.0, float(safe_accel))
        self.publish_interval = max(1e-6, float(publish_interval))
        self.last_published_speed = None

    @property
    def max_speed_step(self):
        return self.safe_accel * self.publish_interval

    def reset(self, initial_speed=None):
        initial_speed = _finite(initial_speed)
        self.last_published_speed = (
            None if initial_speed is None else max(0.0, initial_speed)
        )

    def step(self, desired_speed, ego_speed=None):
        desired_speed = _finite(desired_speed)
        ego_speed = _finite(ego_speed)
        if self.last_published_speed is None:
            if ego_speed is None:
                return None
            self.last_published_speed = max(0.0, ego_speed)

        previous = self.last_published_speed
        desired = previous if desired_speed is None else max(0.0, desired_speed)
        delta = self.max_speed_step
        low = max(0.0, previous - delta)
        high = previous + delta

        # Keep the target within one evaluator step of feedback whenever that
        # window intersects the previous-publication continuity window.
        if ego_speed is not None:
            feedback_low = max(0.0, ego_speed - delta)
            feedback_high = max(0.0, ego_speed + delta)
            intersection_low = max(low, feedback_low)
            intersection_high = min(high, feedback_high)
            if intersection_low <= intersection_high:
                low = intersection_low
                high = intersection_high

        limited_speed = max(low, min(high, desired))
        limited_accel = (
            (limited_speed - previous) / self.publish_interval
        )
        limited_accel = max(
            -self.safe_accel,
            min(self.safe_accel, limited_accel),
        )
        self.last_published_speed = limited_speed
        return limited_speed, limited_accel, previous
