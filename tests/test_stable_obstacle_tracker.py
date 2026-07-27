import os
import sys
import unittest


SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from stable_obstacle_tracker import (  # noqa: E402
    StableObstacleTracker,
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
)


def detection(
    x,
    y=0.0,
    obstacle_type=TYPE_VEHICLE,
    score=0.60,
    length=4.0,
    width=1.8,
):
    return {
        "x": x,
        "y": y,
        "length": length,
        "width": width,
        "heading": 0.0,
        "type": obstacle_type,
        "score": score,
    }


class StableObstacleTrackerTest(unittest.TestCase):
    def test_low_confidence_detection_needs_two_hits(self):
        tracker = StableObstacleTracker()
        self.assertEqual([], tracker.update([detection(10.0)], 1.0))

        output = tracker.update([detection(10.1)], 1.1)
        self.assertEqual(1, len(output))
        self.assertEqual("1", output[0]["id"])
        self.assertEqual(2, output[0]["hits"])

    def test_high_confidence_detection_is_available_immediately(self):
        tracker = StableObstacleTracker()
        output = tracker.update(
            [
                detection(
                    8.0,
                    obstacle_type=TYPE_PEDESTRIAN,
                    score=0.80,
                    length=0.8,
                    width=0.6,
                )
            ],
            1.0,
        )
        self.assertEqual(1, len(output))
        self.assertEqual(TYPE_PEDESTRIAN, output[0]["type"])

    def test_short_miss_coasts_and_reacquires_same_id(self):
        tracker = StableObstacleTracker()
        tracker.update([detection(10.0, score=0.90)], 1.0)
        tracker.update([detection(10.5, score=0.90)], 1.1)

        coasted = tracker.update([], 1.2)
        self.assertEqual(1, len(coasted))
        self.assertEqual("1", coasted[0]["id"])
        self.assertTrue(coasted[0]["predicted"])

        reacquired = tracker.update([detection(11.5, score=0.90)], 1.3)
        self.assertEqual(1, len(reacquired))
        self.assertEqual("1", reacquired[0]["id"])
        self.assertFalse(reacquired[0]["predicted"])

    def test_vehicle_cyclist_class_flicker_keeps_track_id(self):
        tracker = StableObstacleTracker()
        first = tracker.update(
            [detection(5.0, obstacle_type=TYPE_VEHICLE, score=0.90)],
            1.0,
        )
        second = tracker.update(
            [
                detection(
                    5.2,
                    obstacle_type=TYPE_CYCLIST,
                    score=0.55,
                    length=2.0,
                    width=0.7,
                )
            ],
            1.1,
        )
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(TYPE_VEHICLE, second[0]["type"])

    def test_input_order_does_not_change_nearest_association(self):
        tracker = StableObstacleTracker()
        initial = tracker.update(
            [
                detection(0.0, y=-1.0, score=0.95),
                detection(0.0, y=1.0, score=0.95),
            ],
            1.0,
        )
        self.assertEqual(["1", "2"], [item["id"] for item in initial])

        output = tracker.update(
            [
                detection(0.1, y=0.9, score=0.95),
                detection(0.1, y=-0.9, score=0.95),
            ],
            1.1,
        )
        by_id = {item["id"]: item for item in output}
        self.assertLess(by_id["1"]["y"], 0.0)
        self.assertGreater(by_id["2"]["y"], 0.0)

    def test_static_position_jitter_does_not_create_motion(self):
        tracker = StableObstacleTracker()
        positions = [10.00, 10.02, 9.99, 10.01, 10.00, 10.02]
        output = []
        for index, position in enumerate(positions):
            output = tracker.update(
                [detection(position, score=0.90)],
                1.0 + 0.1 * index,
            )
        self.assertEqual(1, len(output))
        self.assertLess(output[0]["speed"], 0.15)
        self.assertTrue(output[0]["speed_valid"])


if __name__ == "__main__":
    unittest.main()
