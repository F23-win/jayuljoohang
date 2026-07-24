import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rear_range_probe import (  # noqa: E402
    accumulate_bins,
    bin_index,
    is_valid_return,
    sector_stats,
)
from skku_autocar.sensors.lidar import LidarPoint, LidarScan  # noqa: E402


def scan(*points: LidarPoint) -> LidarScan:
    return LidarScan(timestamp=1.0, points=points)


class SectorStatsTest(unittest.TestCase):
    def test_finds_min_distance_and_count_inside_window(self):
        result = sector_stats(
            scan(
                LidarPoint(quality=15, angle_deg=175.0, distance_mm=800.0),
                LidarPoint(quality=15, angle_deg=185.0, distance_mm=500.0),
                LidarPoint(quality=15, angle_deg=90.0, distance_mm=100.0),  # outside window
            ),
            center_deg=180.0,
            half_width_deg=10.0,
            quality_min=1,
            max_range_mm=12000.0,
        )
        self.assertEqual(result, (500.0, 2))

    def test_wraps_across_zero_degrees(self):
        result = sector_stats(
            scan(LidarPoint(quality=15, angle_deg=358.0, distance_mm=300.0)),
            center_deg=0.0,
            half_width_deg=5.0,
            quality_min=1,
            max_range_mm=12000.0,
        )
        self.assertEqual(result, (300.0, 1))

    def test_no_valid_returns_gives_none(self):
        result = sector_stats(
            scan(LidarPoint(quality=0, angle_deg=180.0, distance_mm=500.0)),
            center_deg=180.0,
            half_width_deg=10.0,
            quality_min=1,
            max_range_mm=12000.0,
        )
        self.assertEqual(result, (None, 0))


class ValidReturnTest(unittest.TestCase):
    def test_rejects_low_quality_zero_distance_and_out_of_range(self):
        self.assertFalse(is_valid_return(LidarPoint(0, 0.0, 500.0), quality_min=1, max_range_mm=12000.0))
        self.assertFalse(is_valid_return(LidarPoint(15, 0.0, 0.0), quality_min=1, max_range_mm=12000.0))
        self.assertFalse(is_valid_return(LidarPoint(15, 0.0, 20000.0), quality_min=1, max_range_mm=12000.0))
        self.assertTrue(is_valid_return(LidarPoint(15, 0.0, 500.0), quality_min=1, max_range_mm=12000.0))

    def test_max_range_zero_disables_the_cap(self):
        self.assertTrue(is_valid_return(LidarPoint(15, 0.0, 50000.0), quality_min=1, max_range_mm=0.0))


class BinningTest(unittest.TestCase):
    def test_bin_index_and_accumulate(self):
        self.assertEqual(bin_index(15.0, 10.0), 1)
        self.assertEqual(bin_index(359.0, 10.0), 35)

        bins = {}
        accumulate_bins(
            bins,
            scan(
                LidarPoint(quality=15, angle_deg=12.0, distance_mm=400.0),
                LidarPoint(quality=0, angle_deg=15.0, distance_mm=400.0),
            ),
            bin_width_deg=10.0,
            quality_min=1,
            max_range_mm=12000.0,
        )
        self.assertEqual(bins[1], [1, 2])


if __name__ == "__main__":
    unittest.main()
