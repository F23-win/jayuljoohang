import unittest

import numpy as np

from skku_autocar.estimation.parking_geometry import (
    ParkingGeometry,
    ParkingGeometryConfig,
    ParkingGeometryDepthStabilizer,
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


def slanted_mask(x_top, x_bottom, shape=(200, 200)):
    mask = np.zeros(shape, dtype=np.uint8)
    for y in range(40, 190):
        ratio = (y - 40) / 149.0
        x = int(round(x_top + (x_bottom - x_top) * ratio))
        mask[y, max(0, x - 2) : min(shape[1], x + 3)] = 255
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

    def test_three_line_cross_anchor_accepts_diverging_side_lines(self):
        estimator = self.make_estimator(
            slot_width_max_px=160.0,
            expected_slot_width_px=120.0,
        )
        left = slanted_mask(50, 25)
        right = slanted_mask(150, 175)

        geometry = estimator.estimate(
            [left, right, horizontal_mask(40)],
            confidence=1.0,
            selection_mode="three_line_after_car",
        )

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertEqual(geometry.selection_mode, "three_line_after_car")
        self.assertGreater(
            axial_angle_difference(
                geometry.left.angle_deg,
                geometry.right.angle_deg,
            ),
            estimator.config.parallel_tolerance_deg,
        )
        self.assertGreaterEqual(geometry.back.mask_index, 0)

    def test_three_parallel_lines_are_rejected_after_car(self):
        estimator = self.make_estimator()

        geometry = estimator.estimate(
            [vertical_mask(40), vertical_mask(100), vertical_mask(160)],
            confidence=1.0,
            selection_mode="three_line_after_car",
        )

        self.assertFalse(geometry.found)
        self.assertEqual(geometry.selection_mode, "three_line_after_car")
        self.assertEqual(geometry.observed_line_count, 3)
        self.assertEqual(geometry.reason, "three_line_bay_invalid")

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

    def test_jump_holds_previous_path_then_switches_after_five_frames(self):
        estimator = self.make_estimator(
            min_confirm_frames=3,
            jump_reconfirm_frames=5,
            max_jump_hold_frames=3,
            max_lateral_jump_px=40.0,
        )
        old = None
        for _ in range(3):
            old = estimator.estimate(
                [vertical_mask(160)],
                confidence=1.0,
                selection_mode="single_car_left",
            )
        self.assertTrue(old.found)

        observations = [
            estimator.estimate(
                [vertical_mask(20)],
                confidence=1.0,
                selection_mode="single_car_left",
            )
            for _ in range(5)
        ]

        self.assertTrue(all(item.found and item.coasted for item in observations[:4]))
        self.assertTrue(all("reconfirming_jump" in item.reason for item in observations[:4]))
        self.assertTrue(observations[4].found)
        self.assertFalse(observations[4].coasted)
        self.assertNotEqual(observations[4].slot_center_x_px, old.slot_center_x_px)

    def test_unstable_jump_stops_after_three_held_frames(self):
        estimator = self.make_estimator(
            min_confirm_frames=3,
            jump_reconfirm_frames=5,
            max_jump_hold_frames=3,
            max_lateral_jump_px=40.0,
        )
        for _ in range(3):
            confirmed = estimator.estimate(
                [vertical_mask(160)],
                confidence=1.0,
                selection_mode="single_car_left",
            )
        self.assertTrue(confirmed.found)

        results = [
            estimator.estimate(
                [vertical_mask(x)],
                confidence=1.0,
                selection_mode="single_car_left",
            )
            for x in (20, 180, 20, 180, 20)
        ]

        self.assertTrue(all(item.found and item.coasted for item in results[:4]))
        self.assertFalse(results[4].found)

    def test_heading_jump_is_symmetric_for_both_directions(self):
        estimator = self.make_estimator(max_heading_jump_deg=35.0)
        positive = ParkingGeometry(heading_error_deg=40.0)
        negative = ParkingGeometry(heading_error_deg=-20.0)

        self.assertTrue(estimator._is_jump(positive, negative))
        self.assertTrue(estimator._is_jump(negative, positive))

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

    def test_first_yolo_car_always_anchors_virtual_slot_on_its_left(self):
        nearest_left = vertical_mask(10)
        right = [vertical_mask(60), vertical_mask(140)]

        selected, mode = select_parking_line_masks(
            [nearest_left] + right,
            [car_mask(20, 50)],
        )

        self.assertEqual(mode, "single_car_left")
        self.assertEqual(selected, (nearest_left,))

    def test_single_right_yolo_car_restores_slot_on_its_left(self):
        lines = [vertical_mask(10), vertical_mask(60), vertical_mask(140)]

        selected, mode = select_parking_line_masks(
            lines,
            [car_mask(150, 190)],
        )

        self.assertEqual(mode, "single_car_left")
        self.assertEqual(selected, (lines[2],))

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

    def test_confirmed_two_car_gap_is_coasted_before_survivor_fallback(self):
        estimator = self.make_estimator(max_coast_frames=2)
        lines = [vertical_mask(10), vertical_mask(60), vertical_mask(140)]
        cars = [car_mask(10, 40), car_mask(160, 190)]

        selected, mode = estimator.select_masks(lines, cars)
        locked = estimator.estimate(selected, 1.0, mode, observed_car_count=2)
        self.assertEqual(locked.selection_mode, "two_car")

        selected, mode = estimator.select_masks(lines, [cars[0]])
        first = estimator.estimate(selected, 1.0, mode, observed_car_count=1)
        selected, mode = estimator.select_masks(lines, [cars[0]])
        second = estimator.estimate(selected, 1.0, mode, observed_car_count=1)

        self.assertTrue(first.coasted)
        self.assertTrue(second.coasted)
        self.assertEqual(first.selection_mode, "locked_gap_coast")
        self.assertEqual(first.reason, "coast:locked_two_car_gap")

        selected, mode = estimator.select_masks(lines, [cars[0]])
        fallback = estimator.estimate(selected, 1.0, mode, observed_car_count=1)

        self.assertEqual(mode, "single_car_right")
        self.assertFalse(fallback.coasted)
        self.assertEqual(fallback.selection_mode, "single_car_right")

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

    def test_vehicle_mask_filter_keeps_wide_car_in_upper_camera_region(self):
        car = np.zeros((200, 200), dtype=np.uint8)
        car[20:48, 45:155] = 255

        self.assertEqual(filter_parking_car_masks([car], 0.20), (car,))

    def test_vehicle_mask_filter_rejects_upper_background_car_by_default(self):
        background_car = np.zeros((200, 200), dtype=np.uint8)
        background_car[20:48, 45:155] = 255

        self.assertEqual(filter_parking_car_masks([background_car]), ())

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
        camera = ParkingGeometry(
            observed_car_count=1,
            observed_line_count=1,
            reason="need_two_lines",
        )

        merged = merge_camera_slot_guidance(ParkingGeometry(), camera)

        self.assertEqual(merged.observed_car_count, 1)
        self.assertEqual(merged.observed_line_count, 1)

    def test_final_depth_jump_is_held_until_reconfirmed(self):
        stabilizer = ParkingGeometryDepthStabilizer(
            max_jump_px=160.0,
            reconfirm_frames=3,
            candidate_tolerance_px=45.0,
        )
        accepted = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            depth_remaining_px=564.0,
            stop_target_x_px=300.0,
            stop_target_y_px=100.0,
            selection_mode="single_car_left",
            reason="locked_slot",
        )
        jumped = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            depth_remaining_px=318.0,
            stop_target_x_px=50.0,
            stop_target_y_px=500.0,
            selection_mode="line_only",
            reason="camera_back_line",
        )

        stabilizer.update(accepted)
        first = stabilizer.update(jumped)
        second = stabilizer.update(jumped)
        third = stabilizer.update(jumped)

        self.assertEqual(first.depth_remaining_px, 564.0)
        self.assertEqual(second.depth_remaining_px, 564.0)
        self.assertEqual(first.stop_target_x_px, 300.0)
        self.assertEqual(first.selection_mode, "single_car_left")
        self.assertTrue(first.coasted)
        self.assertIn("depth_reconfirming:1/3", first.reason)
        self.assertEqual(third.depth_remaining_px, 318.0)

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

    def test_three_line_bay_takes_over_after_yolo_car_disappears(self):
        estimator = self.make_estimator(
            min_confirm_frames=3,
            jump_reconfirm_frames=5,
        )
        masks = parking_masks()
        for _ in range(3):
            selected, car_mode = estimator.select_masks(
                masks,
                [car_mask(160, 190)],
            )
            estimator.estimate(selected, confidence=1.0, selection_mode=car_mode)

        selected, mode = estimator.select_masks(masks, [])
        attempts = [
            estimator.estimate(
                selected,
                confidence=1.0,
                selection_mode=mode,
            )
            for _ in range(3)
        ]
        recovered = attempts[-1]

        self.assertEqual(mode, "three_line_after_car")
        self.assertTrue(all(not item.found for item in attempts[:2]))
        self.assertTrue(recovered.found)
        self.assertTrue(recovered.has_side_pair)
        self.assertTrue(recovered.has_back_line)
        self.assertEqual(recovered.selection_mode, "three_line_after_car")
        self.assertFalse(recovered.coasted)

    def test_two_parallel_lines_keep_live_path_after_car_disappears(self):
        estimator = self.make_estimator(max_coast_frames=2)
        masks = parking_masks()
        estimator.select_masks(masks, [car_mask(160, 190)])
        selected, mode = estimator.select_masks(masks, [])
        recovered = estimator.estimate(
            selected,
            confidence=1.0,
            selection_mode=mode,
        )

        side_lines, mode = estimator.select_masks(masks[:2], [])
        lost = estimator.estimate(
            side_lines,
            confidence=1.0,
            selection_mode=mode,
        )

        self.assertTrue(recovered.found)
        self.assertTrue(lost.found)
        self.assertFalse(lost.coasted)
        self.assertEqual(lost.observed_line_count, 2)
        self.assertEqual(lost.selection_mode, "two_line_after_car")

    def test_two_line_t_corner_builds_missing_side_after_car_disappears(self):
        estimator = self.make_estimator()
        masks = parking_masks()
        estimator.select_masks(masks, [car_mask(160, 190)])

        selected, mode = estimator.select_masks([masks[0], masks[2]], [])
        recovered = estimator.estimate(
            selected,
            confidence=1.0,
            selection_mode=mode,
        )
        path = ReverseParkingPathGenerator().generate(recovered)

        self.assertEqual(mode, "two_line_after_car")
        self.assertTrue(recovered.found)
        self.assertTrue(recovered.has_side_pair)
        self.assertTrue(recovered.has_back_line)
        self.assertEqual(recovered.back.mask_index, 1)
        self.assertFalse(recovered.coasted)
        self.assertTrue(path.found)

    def test_one_car_without_lines_selects_car_only_path(self):
        car = car_mask(140, 190)

        selected, mode = select_parking_line_masks([], [car])

        self.assertEqual(selected, ())
        self.assertEqual(mode, "car_only_left")

    def test_car_only_mask_builds_virtual_bev_path(self):
        estimator = self.make_estimator()
        car = car_mask(140, 190)
        selected, mode = estimator.select_masks([], [car])

        geometry = estimator.estimate(
            selected,
            confidence=1.0,
            selection_mode=mode,
            observed_car_count=1,
            car_masks=[car],
        )
        path = ReverseParkingPathGenerator().generate(geometry)

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertEqual(geometry.selection_mode, "car_only_left")
        self.assertTrue(path.found)


if __name__ == "__main__":
    unittest.main()
