import unittest

import numpy as np

from skku_autocar.estimation.parking_geometry import (
    ParkingGeometry,
    ParkingGeometryConfig,
    ParkingGeometryEstimator,
    axial_angle_difference,
    filter_parking_car_masks,
    merge_camera_back_line,
    merge_camera_slot_guidance,
    select_parking_line_masks,
)
from skku_autocar.planning.reverse_parking_path import ReverseParkingPathGenerator


def parking_masks(shape=(200, 200)):
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    back = np.zeros(shape, dtype=np.uint8)
    left[40:191, 48:53] = 255
    right[40:191, 148:153] = 255
    back[38:43, 48:153] = 255
    return [left, right, back]


def vertical_mask(x, shape=(200, 200)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[40:190, x : x + 4] = 255
    return mask


def car_mask(left, right, shape=(200, 200)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:80, left:right] = 255
    return mask


def horizontal_mask(y, shape=(200, 200)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y : y + 4, 20:180] = 255
    return mask


def vertical_line(x):
    from skku_autocar.estimation.parking_geometry import ParkingLine

    return ParkingLine(
        center_x=x,
        center_y=100.0,
        direction_x=0.0,
        direction_y=-1.0,
        length_px=150.0,
        residual_px=0.0,
        quality=1.0,
        point_count=100,
    )


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

    def test_parallel_pair_builds_virtual_depth_for_bev_reverse_path(self):
        geometry = self.make_estimator().estimate(parking_masks()[:2], confidence=1.0)

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertEqual(geometry.back.mask_index, -1)
        self.assertIsNotNone(geometry.depth_remaining_px)
        self.assertEqual(geometry.reason, "parking_bay")
        self.assertTrue(ReverseParkingPathGenerator().generate(geometry).found)

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
        camera_geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=False,
            reason="side_pair",
        )
        lidar_geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=False,
            depth_remaining_px=None,
            reason="lidar_slot_box",
        )

        merged = merge_camera_back_line(lidar_geometry, camera_geometry)

        self.assertEqual(merged, lidar_geometry)

    def test_two_yolo_cars_select_lines_between_them(self):
        outside = vertical_mask(5)
        inside = [vertical_mask(50), vertical_mask(100), vertical_mask(150)]

        selected, mode = select_parking_line_masks(
            [outside] + inside,
            [car_mask(10, 40), car_mask(160, 190)],
        )

        self.assertEqual(mode, "two_car")
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(mask is expected for mask, expected in zip(selected, inside)))

    def test_initial_single_yolo_car_restores_slot_on_its_left(self):
        nearest_left = vertical_mask(10)
        right = [vertical_mask(60), vertical_mask(140)]

        selected, mode = select_parking_line_masks(
            [nearest_left] + right,
            [car_mask(20, 50)],
        )

        self.assertEqual(mode, "single_car_left")
        self.assertEqual(selected, (nearest_left,))

    def test_after_two_cars_single_survivor_restores_slot_on_its_right(self):
        estimator = self.make_estimator()
        lines = [vertical_mask(10), vertical_mask(60), vertical_mask(140)]

        for _ in range(3):
            estimator.select_masks(
                lines,
                [car_mask(10, 40), car_mask(160, 190)],
            )
        selected, mode = estimator.select_masks(lines, [car_mask(20, 50)])

        self.assertEqual(mode, "single_car_right")
        self.assertEqual(selected, (lines[1],))

    def test_single_car_boundary_builds_virtual_parallel_slot_edge(self):
        geometry = self.make_estimator().estimate(
            [vertical_mask(110)],
            confidence=1.0,
            selection_mode="single_car_left",
        )

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertEqual(geometry.back.mask_index, -1)
        self.assertEqual(geometry.selection_mode, "single_car_left")
        self.assertAlmostEqual(geometry.slot_width_px, 100.0, delta=2.0)
        self.assertLess(geometry.left.center_x, geometry.right.center_x)

    def test_single_car_boundary_keeps_perpendicular_back_line(self):
        geometry = self.make_estimator().estimate(
            [vertical_mask(110), horizontal_mask(45)],
            confidence=1.0,
            selection_mode="single_car_left",
        )

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_back_line)
        self.assertNotEqual(geometry.back.mask_index, -1)
        self.assertIsNotNone(geometry.stop_target_y_px)

    def test_vehicle_mask_filter_rejects_upright_person_shape(self):
        person = np.zeros((200, 200), dtype=np.uint8)
        person[40:190, 80:115] = 255
        car = np.zeros((200, 200), dtype=np.uint8)
        car[110:170, 30:150] = 255

        self.assertEqual(filter_parking_car_masks([person, car]), (car,))

    def test_camera_only_merge_uses_camera_vehicle_reference_for_depth(self):
        camera = self.make_estimator().estimate(
            [vertical_mask(110)],
            confidence=1.0,
            selection_mode="single_car_left",
        )

        merged = merge_camera_slot_guidance(ParkingGeometry(), camera)

        self.assertTrue(merged.found)
        self.assertAlmostEqual(merged.depth_remaining_px, camera.depth_remaining_px)

    def test_camera_car_count_is_preserved_without_confirmed_geometry(self):
        camera = ParkingGeometry(observed_car_count=1, reason="need_two_lines")

        merged = merge_camera_slot_guidance(ParkingGeometry(), camera)

        self.assertEqual(merged.observed_car_count, 1)

    def test_car_selected_camera_lines_override_lidar_center_guidance(self):
        camera = self.make_estimator().estimate(
            parking_masks(),
            confidence=1.0,
            selection_mode="two_car",
        )
        lidar = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            left=vertical_line(20.0),
            right=vertical_line(180.0),
            back=camera.back,
            lateral_error_px=35.0,
            lateral_error_norm=0.5,
            heading_error_deg=20.0,
            depth_to_back_px=150.0,
            depth_remaining_px=130.0,
            slot_width_px=160.0,
            vehicle_x_px=100.0,
            vehicle_y_px=190.0,
            back_center_x_px=100.0,
            back_center_y_px=40.0,
            stop_target_x_px=100.0,
            stop_target_y_px=60.0,
            confidence=0.9,
            reason="lidar_slot_box",
        )

        merged = merge_camera_slot_guidance(lidar, camera)

        self.assertEqual(merged.selection_mode, "two_car")
        self.assertEqual(merged.left, camera.left)
        self.assertEqual(merged.right, camera.right)
        self.assertAlmostEqual(merged.lateral_error_norm, camera.lateral_error_norm)
        self.assertAlmostEqual(merged.heading_error_deg, camera.heading_error_deg)

    def test_line_only_camera_side_pair_can_guide_after_car_trigger(self):
        camera = self.make_estimator().estimate(
            parking_masks(),
            confidence=1.0,
            selection_mode="line_only",
        )

        merged = merge_camera_slot_guidance(ParkingGeometry(), camera)

        self.assertTrue(merged.found)
        self.assertTrue(merged.has_side_pair)
        self.assertEqual(merged.selection_mode, "line_only")


if __name__ == "__main__":
    unittest.main()
