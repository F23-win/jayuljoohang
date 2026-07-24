import unittest

import numpy as np

from skku_autocar.estimation.parking_geometry import (
    ParkingGeometry,
    ParkingGeometryConfig,
    ParkingGeometryEstimator,
    axial_angle_difference,
    merge_camera_back_line,
)


def parking_masks(shape=(200, 200)):
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    back = np.zeros(shape, dtype=np.uint8)
    left[40:191, 48:53] = 255
    right[40:191, 148:153] = 255
    back[38:43, 48:153] = 255
    return [left, right, back]


class ParkingGeometryTest(unittest.TestCase):
    def make_estimator(self, **overrides):
        values = dict(
            min_confirm_frames=1,
            smooth_alpha=1.0,
            slot_width_min_px=70.0,
            slot_width_max_px=130.0,
            expected_slot_width_px=100.0,
            desired_back_clearance_px=20.0,
            min_geometry_confidence=0.1,
        )
        values.update(overrides)
        return ParkingGeometryEstimator(ParkingGeometryConfig(**values))

    def test_three_instances_resolve_parking_bay(self):
        geometry = self.make_estimator().estimate(parking_masks(), confidence=1.0)

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertAlmostEqual(geometry.slot_width_px, 100.0, delta=2.0)
        self.assertAlmostEqual(geometry.lateral_error_px, 0.0, delta=2.0)
        self.assertAlmostEqual(geometry.heading_error_deg, 0.0, delta=1.0)
        self.assertAlmostEqual(geometry.depth_to_back_px, 150.0, delta=3.0)
        self.assertAlmostEqual(geometry.depth_remaining_px, 130.0, delta=3.0)
        self.assertAlmostEqual(geometry.vehicle_x_px, 100.0, delta=1.0)
        self.assertAlmostEqual(geometry.stop_target_x_px, 100.0, delta=3.0)
        self.assertAlmostEqual(geometry.stop_target_y_px, 60.0, delta=3.0)
        self.assertEqual(geometry.left.mask_index, 0)
        self.assertEqual(geometry.right.mask_index, 1)
        self.assertEqual(geometry.back.mask_index, 2)

    def test_parallel_pair_can_guide_alignment_without_back_line(self):
        geometry = self.make_estimator().estimate(parking_masks()[:2], confidence=1.0)

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertFalse(geometry.has_back_line)
        self.assertIsNone(geometry.depth_remaining_px)
        self.assertEqual(geometry.reason, "side_pair")

    def test_short_dropout_coasts_then_reports_lost(self):
        estimator = self.make_estimator(max_coast_frames=2)
        found = estimator.estimate(parking_masks(), confidence=1.0)
        first = estimator.estimate([], confidence=0.0)
        second = estimator.estimate([], confidence=0.0)
        lost = estimator.estimate([], confidence=0.0)

        self.assertTrue(found.found)
        self.assertTrue(first.found)
        self.assertTrue(first.coasted)
        self.assertTrue(second.found)
        self.assertFalse(lost.found)

    def test_axial_angle_wraps_at_180_degrees(self):
        self.assertAlmostEqual(axial_angle_difference(2.0, 178.0), 4.0)

    def test_merge_camera_back_line_overrides_lidar_depth_only(self):
        camera_geometry = self.make_estimator().estimate(parking_masks(), confidence=1.0)
        lidar_geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=False,
            heading_error_deg=5.0,
            lateral_error_norm=0.1,
            confidence=0.9,
            depth_remaining_px=9999.0,
            reason="lidar_slot_box",
        )

        merged = merge_camera_back_line(lidar_geometry, camera_geometry)

        self.assertTrue(merged.has_back_line)
        self.assertEqual(merged.back, camera_geometry.back)
        self.assertEqual(merged.depth_remaining_px, camera_geometry.depth_remaining_px)
        # Everything else stays the LiDAR pipeline's own estimate.
        self.assertEqual(merged.heading_error_deg, 5.0)
        self.assertEqual(merged.lateral_error_norm, 0.1)
        self.assertEqual(merged.confidence, 0.9)
        self.assertEqual(merged.reason, "lidar_slot_box")

    def test_merge_camera_back_line_is_noop_without_camera_back_line(self):
        camera_geometry = self.make_estimator().estimate(parking_masks()[:2], confidence=1.0)
        lidar_geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=False,
            depth_remaining_px=None,
            reason="lidar_slot_box",
        )

        merged = merge_camera_back_line(lidar_geometry, camera_geometry)

        self.assertEqual(merged, lidar_geometry)


if __name__ == "__main__":
    unittest.main()
