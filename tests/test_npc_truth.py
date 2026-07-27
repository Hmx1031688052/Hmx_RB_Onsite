import os
import struct
import sys
import unittest


SAMPLE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from npc_truth import (
    NpcTruthDecodeError,
    decode_npc_payload,
    ensure_npc_truth_timestamp,
)


def make_role():
    name = b"Local_car2"
    model = b"Veh_Emgrand"
    numeric = bytearray(172)
    struct.pack_into("<3d", numeric, 0, 15.625, -3.187, -1.919)
    struct.pack_into("<4f", numeric, 24, 0.0, 0.0, -0.1, 0.995)
    struct.pack_into("<3f", numeric, 40, 3.0, 0.0, 0.0)
    struct.pack_into("<3f", numeric, 52, 4.61, 2.26, 1.77)
    struct.pack_into("<I", numeric, 64, 6)
    struct.pack_into("<4f", numeric, 68, 900.0, 450.0, 250.0, 160.0)
    yaw = -0.2003323297758503
    struct.pack_into("<2f", numeric, 164, yaw, 0.2)
    return (
        struct.pack(
            "<QQQIQ",
            1784882209545,
            1784882209545,
            2,
            85130167,
            1784882209545,
        )
        + bytes([1, len(name) + 1])
        + name
        + bytes([0x22, len(model)])
        + model
        + bytes(numeric)
    )


class NpcTruthTest(unittest.TestCase):
    def test_decodes_custom_npc_role(self):
        payload = struct.pack("<I", 1) + make_role()
        decoded = decode_npc_payload(payload)
        self.assertEqual(1, decoded["role_count"])
        role = decoded["roles"][0]
        self.assertEqual("Local_car2", role["role_name"])
        self.assertEqual("Veh_Emgrand", role["model_name"])
        self.assertEqual("Vehicle", role["class_name"])
        self.assertAlmostEqual(15.625, role["position"]["x"])
        self.assertAlmostEqual(4.61, role["dimensions"]["length"], places=2)
        self.assertTrue(role["dimensions_valid"])
        self.assertAlmostEqual(
            role["yaw"], role["quaternion_yaw"], places=3
        )

    def test_empty_frame_is_valid(self):
        decoded = decode_npc_payload(struct.pack("<I", 0))
        self.assertEqual(0, decoded["role_count"])
        self.assertEqual([], decoded["roles"])

    def test_empty_frame_uses_receive_clock_timestamp(self):
        previous = {
            "timestamp_s": 100.0,
            "_received_monotonic": 20.0,
        }
        decoded = decode_npc_payload(struct.pack("<I", 0))
        ensure_npc_truth_timestamp(
            decoded,
            previous_truth=previous,
            received_monotonic=20.1,
            wall_timestamp=999.0,
        )
        self.assertAlmostEqual(100.1, decoded["timestamp_s"])
        self.assertEqual(
            "receive_clock_empty_frame",
            decoded["timestamp_source"],
        )

    def test_rejects_truncated_role(self):
        with self.assertRaises(NpcTruthDecodeError):
            decode_npc_payload(struct.pack("<I", 1) + b"\x00" * 10)


if __name__ == "__main__":
    unittest.main()
