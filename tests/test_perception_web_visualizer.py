import json
import os
import sys
import unittest
from urllib.request import urlopen

import numpy as np


SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from perception_web_visualizer import PerceptionWebVisualizer


class PerceptionWebVisualizerTests(unittest.TestCase):
    def test_publish_downsamples_points_and_serializes_detections(self):
        visualizer = PerceptionWebVisualizer(
            port=0,
            max_points=100,
            gt_only=False,
        )
        points = np.column_stack(
            (
                np.arange(250, dtype=float),
                np.zeros(250),
                np.ones(250),
                np.zeros(250),
            )
        )
        boxes = np.array([[10.0, -2.0, 0.5, 4.5, 1.8, 1.6, 0.2]])
        visualizer.publish_points(points)
        detecting = json.loads(visualizer._frame_json.decode("utf-8"))
        self.assertEqual(detecting["stage"], "detecting")
        self.assertEqual(len(detecting["points"]), 100)

        visualizer.publish(points, boxes, np.array([2]), np.array([0.91]))

        payload = json.loads(visualizer._frame_json.decode("utf-8"))
        self.assertEqual(payload["stage"], "ready")
        self.assertEqual(len(payload["points"]), 100)
        self.assertEqual(payload["detections"][0]["label"], "Car")
        self.assertEqual(payload["detections"][0]["score"], 0.91)
        self.assertAlmostEqual(payload["detections"][0]["distance"], 10.198, places=3)

    def test_publish_ground_truth_marks_placeholder_dimensions(self):
        visualizer = PerceptionWebVisualizer(port=0)
        truth = {
            "decoder": "drivesim_npc_binary_v1",
            "timestamp_s": 101.125,
            "roles": [
                {
                    "role_name": "Local_car1",
                    "model_name": "Veh_Lynkco",
                    "class_name": "Vehicle",
                    "coordinate_frame": "ego_or_lidar_unverified",
                    "position": {"x": 16.3, "y": -0.1, "z": -1.88},
                    "yaw": 0.02,
                    "dimensions": {
                        "length": 1.0,
                        "width": 1.0,
                        "height": 1.0,
                    },
                    "dimensions_valid": False,
                }
            ],
        }
        visualizer.publish_ground_truth(
            truth,
            lidar_timestamp=101.0,
        )

        payload = json.loads(visualizer._frame_json.decode("utf-8"))
        self.assertEqual(len(payload["ground_truth"]), 1)
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(
            payload["detections"][0]["source"], "npc_truth"
        )
        box = payload["ground_truth"][0]
        self.assertEqual(box["role_name"], "Local_car1")
        self.assertEqual(box["dimensions_source"], "display_fallback")
        self.assertEqual(box["length"], 4.8)
        self.assertEqual(payload["truth_meta"]["sync_delta_ms"], 125.0)

    def test_gt_only_hides_detector_boxes(self):
        visualizer = PerceptionWebVisualizer(port=0, gt_only=True)
        points = np.zeros((2, 3))
        boxes = np.array(
            [[10.0, -2.0, 0.5, 4.5, 1.8, 1.6, 0.2]]
        )
        visualizer.publish(
            points,
            boxes,
            np.array([2]),
            np.array([0.91]),
        )
        payload = json.loads(
            visualizer._frame_json.decode("utf-8")
        )
        self.assertEqual(payload["detections"], [])

    def test_publish_paths_transforms_map_coordinates_to_ego_frame(self):
        class Object(object):
            pass

        ego = Object()
        ego.x = 100.0
        ego.y = 200.0
        ego.theta = np.pi / 2.0
        trajectory = Object()
        trajectory.x = np.array([100.0, 95.0])
        trajectory.y = np.array([200.0, 200.0])

        visualizer = PerceptionWebVisualizer(port=0)
        visualizer.publish_points(np.zeros((2, 3)))
        visualizer.publish_paths(
            ego=ego,
            global_path={
                "x": [100.0, 100.0, 100.0],
                "y": [190.0, 200.0, 210.0],
            },
            local_trajectory=trajectory,
            behavior="KEEP_LANE",
            target_speed=5.5,
            emergency=False,
            current_s=123.4567,
            current_d=-0.6584,
            map_name="/tmp/maps/MT_14-merge.xodr",
            override_active=True,
            override_name="merge-lane-change",
            override_s_start=120.0,
            override_s_end=180.0,
        )

        payload = json.loads(visualizer._frame_json.decode("utf-8"))
        self.assertEqual(payload["stage"], "detecting")
        self.assertEqual(
            payload["global_path"],
            [[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]],
        )
        self.assertEqual(payload["local_path"], [[0.0, 0.0], [0.0, 5.0]])
        self.assertEqual(payload["planning"]["behavior"], "KEEP_LANE")
        self.assertEqual(payload["planning"]["target_speed"], 5.5)
        self.assertEqual(payload["planning"]["current_s"], 123.457)
        self.assertEqual(payload["planning"]["current_d"], -0.658)
        self.assertEqual(
            payload["planning"]["map_name"], "MT_14-merge.xodr"
        )
        self.assertTrue(payload["planning"]["override_active"])
        self.assertEqual(
            payload["planning"]["override_name"], "merge-lane-change"
        )
        self.assertEqual(
            payload["planning"]["override_range"], "[120.00, 180.00)"
        )

    def test_http_server_exposes_page_health_and_frame(self):
        visualizer = PerceptionWebVisualizer(host="127.0.0.1", port=0)
        self.assertTrue(visualizer.start())
        try:
            base = "http://127.0.0.1:{}".format(visualizer.port)
            with urlopen(base + "/health", timeout=2.0) as response:
                self.assertEqual(json.load(response), {"ok": True})
            with urlopen(base + "/", timeout=2.0) as response:
                page = response.read().decode("utf-8")
                self.assertIn("PointPillars", page)
                self.assertIn("Global path", page)
                self.assertIn("Local path", page)
                self.assertIn('id="currentStation"', page)
                self.assertIn('id="stationMeta"', page)
                self.assertIn("旁车真值框", page)
            with urlopen(base + "/api/frame", timeout=2.0) as response:
                payload = json.load(response)
                self.assertEqual(payload["detections"], [])
                self.assertEqual(payload["ground_truth"], [])
                self.assertEqual(payload["global_path"], [])
                self.assertEqual(payload["local_path"], [])
        finally:
            visualizer.stop()


if __name__ == "__main__":
    unittest.main()
