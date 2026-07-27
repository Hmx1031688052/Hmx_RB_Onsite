import os
import sys
import tempfile
import unittest


SAMPLE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from speed_limits import (
    find_prepare_speed_candidates,
    resolve_expected_speed,
    xodr_speed_summary,
)


class ExpectedSpeedTest(unittest.TestCase):
    def test_explicit_prepare_kmh_field_has_priority(self):
        brief = {
            "speed_limit_mps": 10.0,
            "environment": {
                "expected_speed_kmh": 72.0,
            },
            "testees": [
                {
                    "init_state": {"speed": 0.0},
                    "target_state": {"speed": 30.0},
                }
            ],
        }
        result = resolve_expected_speed(
            brief, "MT_48-urbanvillage.xodr"
        )
        self.assertAlmostEqual(result["speed_mps"], 20.0)
        self.assertEqual(
            result["source"],
            "prepare:environment.expected_speed_kmh",
        )
        self.assertEqual(
            len(find_prepare_speed_candidates(brief)), 2
        )

    def test_command_line_expected_speed_overrides_prepare(self):
        result = resolve_expected_speed(
            {"speed_limit_mps": 5.0},
            "MT_01-highway.xodr",
            command_line_mps=30.0,
        )
        self.assertEqual(result["speed_mps"], 30.0)
        self.assertEqual(result["source"], "command-line")

    def test_xodr_speed_is_diagnostic_unless_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "map.xodr")
            with open(path, "w", encoding="utf-8") as file_obj:
                file_obj.write(
                    '<OpenDRIVE><road><type>'
                    '<speed max="72" unit="km/h"/>'
                    '</type></road></OpenDRIVE>'
                )
            summary = xodr_speed_summary(path)
            self.assertEqual(summary["count"], 1)
            self.assertAlmostEqual(summary["median_mps"], 20.0)

            fallback = resolve_expected_speed(
                {}, "MT_01-highway.xodr", xodr_path=path
            )
            self.assertEqual(
                fallback["source"], "map-category-fallback"
            )
            enabled = resolve_expected_speed(
                {},
                "MT_01-highway.xodr",
                xodr_path=path,
                use_xodr=True,
            )
            self.assertEqual(
                enabled["source"],
                "xodr:median-speed-declaration",
            )
            self.assertAlmostEqual(enabled["speed_mps"], 20.0)


if __name__ == "__main__":
    unittest.main()
