import math
import os
import sys
import unittest


SAMPLE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from gt_perception import GroundTruthObstacleAdapter


class Object:
    pass


def truth(timestamp, x, y, vx=0.0, vy=0.0, valid_size=True):
    return {
        "timestamp_s": timestamp,
        "roles": [
            {
                "role_name": "Local_car1",
                "model_name": "Veh_GeometryE",
                "class_name": "Vehicle",
                "position": {"x": x, "y": y, "z": -1.9},
                "yaw": 0.1,
                "vector_raw": {"x": vx, "y": vy, "z": 0.0},
                "dimensions": {
                    "length": 4.5 if valid_size else 1.0,
                    "width": 1.9 if valid_size else 1.0,
                    "height": 1.6 if valid_size else 1.0,
                },
                "dimensions_valid": valid_size,
            }
        ],
    }


class GroundTruthObstacleAdapterTests(unittest.TestCase):
    def test_transforms_pose_to_world_frame(self):
        ego = Object()
        ego.x = 100.0
        ego.y = 200.0
        ego.theta = math.pi / 2.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
        )
        obstacle = adapter.update(
            truth(10.0, 10.0, -2.0), ego
        )[0]
        self.assertAlmostEqual(obstacle.x, 102.0)
        self.assertAlmostEqual(obstacle.y, 210.0)
        self.assertAlmostEqual(
            obstacle.theta, math.pi / 2.0 + 0.1
        )
        self.assertEqual(obstacle.id, "Local_car1")
        self.assertTrue(obstacle.speed_valid)

    def test_reported_velocity_and_fallback_size(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
            velocity_filter=1.0,
        )
        adapter.update(
            truth(10.0, 10.0, 0.0, valid_size=False),
            ego,
        )
        obstacle = adapter.update(
            truth(
                10.1,
                10.5,
                0.0,
                vx=5.0,
                valid_size=False,
            ),
            ego,
        )[0]
        self.assertAlmostEqual(obstacle.world_vx, 5.0)
        self.assertAlmostEqual(obstacle.world_vy, 0.0)
        self.assertAlmostEqual(obstacle.speed, 5.0)
        self.assertEqual(obstacle.track_hits, 2)
        self.assertEqual(obstacle.length, 4.8)
        self.assertEqual(obstacle.width, 2.0)

    def test_static_reported_velocity_ignores_pose_jitter(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
        )
        adapter.update(truth(10.0, 10.0, 0.0), ego)
        obstacle = adapter.update(
            truth(10.1, 10.5, -0.4), ego
        )[0]
        self.assertAlmostEqual(0.0, obstacle.speed)
        self.assertTrue(obstacle.is_static)
        self.assertEqual(2, obstacle.track_hits)

    def test_duplicate_truth_uses_original_world_transform(self):
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
        )
        first_ego = Object()
        first_ego.x = 100.0
        first_ego.y = 0.0
        first_ego.theta = 0.0
        first = adapter.update(
            truth(10.0, 20.0, 0.0), first_ego
        )[0]

        moved_ego = Object()
        moved_ego.x = 110.0
        moved_ego.y = 0.0
        moved_ego.theta = 0.0
        duplicate = adapter.update(
            truth(10.0, 20.0, 0.0), moved_ego
        )[0]
        self.assertAlmostEqual(first.x, duplicate.x)
        self.assertAlmostEqual(first.y, duplicate.y)

    def test_short_empty_frame_is_predicted_then_expires(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
            track_hold_seconds=0.35,
        )
        adapter.update(
            truth(10.0, 10.0, 0.0, vx=5.0), ego
        )
        held = adapter.update(
            {"timestamp_s": 10.1, "roles": []}, ego
        )
        self.assertEqual(1, len(held))
        self.assertAlmostEqual(10.5, held[0].x)
        self.assertTrue(held[0].track_predicted)
        self.assertEqual(1, held[0].track_misses)

        expired = adapter.update(
            {"timestamp_s": 10.4, "roles": []}, ego
        )
        self.assertEqual([], expired)

    def test_default_hold_bridges_observed_gt_empty_window(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
        )
        adapter.update(
            truth(10.0, 10.0, 0.0, vx=5.0), ego
        )

        held = adapter.update(
            {"timestamp_s": 10.65, "roles": []}, ego
        )
        self.assertEqual(1, len(held))
        self.assertAlmostEqual(13.25, held[0].x)
        self.assertTrue(held[0].track_predicted)

        expired = adapter.update(
            {"timestamp_s": 11.01, "roles": []}, ego
        )
        self.assertEqual([], expired)

    def test_implausible_position_jump_is_rejected(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
            innovation_gate_m=4.0,
            innovation_gate_speed=12.0,
        )
        adapter.update(
            truth(10.0, 10.0, 0.0, vx=5.0), ego
        )
        obstacle = adapter.update(
            truth(10.1, -10.0, 0.0, vx=5.0), ego
        )[0]
        self.assertAlmostEqual(10.5, obstacle.x)
        self.assertTrue(obstacle.track_predicted)
        self.assertTrue(obstacle.innovation_rejected)
        self.assertEqual(1, obstacle.track_misses)

    def test_predict_at_smoothly_advances_between_gt_frames(self):
        ego = Object()
        ego.x = 0.0
        ego.y = 0.0
        ego.theta = 0.0
        adapter = GroundTruthObstacleAdapter(
            sensor_offset_x=0.0,
            sensor_offset_y=0.0,
        )
        adapter.update(
            truth(10.0, 10.0, 0.0, vx=5.0), ego
        )
        obstacle = adapter.predict_at(10.04)[0]
        self.assertAlmostEqual(10.2, obstacle.x)
        self.assertTrue(obstacle.track_predicted)


if __name__ == "__main__":
    unittest.main()
