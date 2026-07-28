import unittest
from dataclasses import replace

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.planning.reverse_parking_path import (
    ReverseParkingPathGenerator,
    ReversePathConfig,
    ReversePathStatus,
)


def geometry(target_x=300.0, target_y=100.0):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=True,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.0,
        slot_direction_y=-1.0,
        stop_target_x_px=target_x,
        stop_target_y_px=target_y,
    )


def footprint_geometry(
    *,
    slot_center_x=300.0,
    slot_width=220.0,
    slot_depth=350.0,
    target_y=210.0,
):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=True,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        vehicle_width_px=140.0,
        vehicle_length_px=230.0,
        rear_axle_to_rear_bumper_px=45.0,
        slot_center_x_px=slot_center_x,
        slot_center_y_px=325.0,
        slot_width_px=slot_width,
        slot_depth_px=slot_depth,
        slot_direction_x=0.0,
        slot_direction_y=-1.0,
        stop_target_x_px=slot_center_x,
        stop_target_y_px=target_y,
    )


def recorded_rotated_geometry():
    """Pose reconstructed from session 20260728_123238 at VERIFY_SLOT_BOX."""

    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=True,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        vehicle_width_px=138.947,
        vehicle_length_px=231.579,
        rear_axle_to_rear_bumper_px=46.316,
        slot_center_x_px=745.807,
        slot_center_y_px=277.997,
        slot_width_px=220.0,
        slot_depth_px=347.368,
        slot_direction_x=0.853146,
        slot_direction_y=-0.521672,
        stop_target_x_px=842.796,
        stop_target_y_px=218.691,
    )


class ReverseParkingPathTest(unittest.TestCase):
    def make_generator(self):
        return ReverseParkingPathGenerator(
            ReversePathConfig(
                samples=31,
                start_tangent_px=100.0,
                end_tangent_px=100.0,
                lookahead_px=80.0,
                maximum_curvature_per_px=0.05,
            )
        )

    def test_path_starts_at_vehicle_and_ends_at_local_lookahead(self):
        path = self.make_generator().generate(geometry())

        self.assertTrue(path.found)
        self.assertEqual(len(path.points), 31)
        self.assertAlmostEqual(path.points[0][0], 300.0)
        self.assertAlmostEqual(path.points[0][1], 570.0)
        self.assertAlmostEqual(path.points[-1][0], 300.0)
        self.assertAlmostEqual(path.points[-1][1], 490.0)
        self.assertAlmostEqual(path.lookahead_point[1], 490.0)
        self.assertEqual(path.reason, "local_target_ready")
        self.assertEqual(path.status, ReversePathStatus.READY)
        self.assertAlmostEqual(path.curvature_per_px, 0.0, places=5)

    def test_target_to_right_generates_signed_curvature(self):
        path = self.make_generator().generate(geometry(target_x=380.0))

        self.assertTrue(path.found)
        self.assertGreater(path.curvature_per_px, 0.0)
        self.assertIsNotNone(path.lookahead_point)

    def test_back_line_is_required(self):
        incomplete = ParkingGeometry(
            found=True,
            has_side_pair=True,
            vehicle_x_px=300.0,
            vehicle_y_px=570.0,
        )

        path = self.make_generator().generate(incomplete)

        self.assertFalse(path.found)
        self.assertEqual(path.status, ReversePathStatus.GEOMETRY_UNAVAILABLE)
        self.assertEqual(path.reason, "parking_back_line_missing")

    def test_current_pose_that_cannot_reverse_toward_target_needs_alignment(self):
        path = self.make_generator().generate(
            footprint_geometry(target_y=700.0)
        )

        self.assertFalse(path.found)
        self.assertEqual(path.status, ReversePathStatus.NEEDS_ALIGNMENT)
        self.assertEqual(path.reason, "target_not_behind_vehicle")

    def test_slot_narrower_than_body_is_physically_impossible(self):
        path = self.make_generator().generate(
            footprint_geometry(slot_width=130.0)
        )

        self.assertFalse(path.found)
        self.assertEqual(path.status, ReversePathStatus.PHYSICALLY_IMPOSSIBLE)
        self.assertEqual(
            path.reason,
            "slot_too_narrow_for_vehicle_footprint",
        )

    def test_swept_rotated_body_crossing_neighbor_boundary_is_collision_risk(self):
        path = self.make_generator().generate(
            footprint_geometry(slot_center_x=400.0)
        )

        self.assertFalse(path.found)
        self.assertEqual(path.status, ReversePathStatus.COLLISION_RISK)
        self.assertGreater(len(path.points), 1)
        self.assertEqual(
            path.reason,
            "vehicle_footprint_crosses_side_boundary",
        )

    def test_straight_body_sweep_inside_slot_is_ready(self):
        path = self.make_generator().generate(footprint_geometry())

        self.assertTrue(path.found)
        self.assertEqual(path.status, ReversePathStatus.READY)

    def test_short_arc_outside_slot_cannot_report_ready(self):
        path = ReverseParkingPathGenerator().generate(
            recorded_rotated_geometry()
        )

        self.assertFalse(path.found)
        self.assertEqual(path.status, ReversePathStatus.COLLISION_RISK)
        self.assertEqual(
            path.reason,
            "vehicle_footprint_crosses_side_boundary",
        )
        self.assertGreater(path.maximum_entry_depth_px, 0.0)

    def test_search_prefers_full_lock_when_multiple_candidates_are_safe(self):
        base = recorded_rotated_geometry()
        shift = 90.0
        safe_pose = replace(
            base,
            slot_center_x_px=base.slot_center_x_px - 0.521672 * shift,
            slot_center_y_px=base.slot_center_y_px - 0.853146 * shift,
            stop_target_x_px=base.stop_target_x_px - 0.521672 * shift,
            stop_target_y_px=base.stop_target_y_px - 0.853146 * shift,
        )

        path = ReverseParkingPathGenerator().generate(safe_pose)

        self.assertTrue(path.found, path.reason)
        self.assertEqual(path.status, ReversePathStatus.READY)
        self.assertEqual(path.reason, "full_arc_straight_ready")
        self.assertAlmostEqual(path.entry_steering_ratio, 1.0)
        self.assertGreater(path.maximum_entry_depth_px, 300.0)
        self.assertGreater(path.minimum_side_clearance_px, 0.0)
        self.assertLess(abs(path.final_lateral_offset_px), 20.0)


if __name__ == "__main__":
    unittest.main()
