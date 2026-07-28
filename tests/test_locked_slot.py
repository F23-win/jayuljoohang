import math
import unittest
from dataclasses import FrozenInstanceError, replace

from skku_autocar.estimation.locked_slot import (
    FixedSlotFrame,
    FrozenSlotLandmarkMap,
    LockedSlotGeometryEstimator,
    LockedSlotPose,
    LockedSlotTracker,
    LockedSlotTrackerConfig,
    MapLocalizationStatus,
    SlotPoseSource,
)
from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import (
    LidarParkingObservation,
    RawTwoCarGapMeasurement,
)


def static_scene():
    points = []
    for index in range(45):
        x = -1200.0 + index * 55.0
        points.append((x, 420.0 + 0.00025 * x * x))
    for index in range(35):
        y = -1100.0 + index * 67.0
        points.append((1350.0, y))
    return points


def two_car_surfaces(center_x=1000.0):
    points = []
    for side_y in (-650.0, 650.0):
        for depth_x in range(-200, 1001, 100):
            points.append((center_x + depth_x, side_y))
    return points


class LockedSlotTrackerTest(unittest.TestCase):
    def make_tracker(self, hold_scans=2):
        return LockedSlotTracker(
            LockedSlotTrackerConfig(
                min_points=10,
                max_points=120,
                max_correspondence_mm=180.0,
                trim_ratio=0.8,
                iterations=8,
                max_translation_per_scan_mm=100.0,
                max_rotation_per_scan_deg=5.0,
                max_hold_scans=hold_scans,
            )
        )

    def test_consecutive_scan_motion_moves_same_locked_rectangle(self):
        previous = static_scene()
        # The LiDAR moved +20 mm right and +12 mm rearward in the world, so
        # stationary world features and the physical bay both appear offset by
        # the inverse amount in the new sensor frame.
        current = [(x - 20.0, y - 12.0) for x, y in previous]
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        tracker = self.make_tracker()

        locked = tracker.lock(polygon, previous)
        tracked = tracker.update(current)

        self.assertTrue(locked.locked)
        self.assertTrue(tracked.tracked, tracked.reason)
        self.assertAlmostEqual(tracked.polygon[0][0], -495.0, delta=3.0)
        self.assertAlmostEqual(tracked.polygon[0][1], -12.0, delta=3.0)
        width = math.hypot(
            tracked.polygon[1][0] - tracked.polygon[0][0],
            tracked.polygon[1][1] - tracked.polygon[0][1],
        )
        depth = math.hypot(
            tracked.polygon[3][0] - tracked.polygon[0][0],
            tracked.polygon[3][1] - tracked.polygon[0][1],
        )
        self.assertAlmostEqual(width, 950.0, delta=0.01)
        self.assertAlmostEqual(depth, 1500.0, delta=0.01)

    def test_missing_scans_hold_then_report_lost_without_resizing_box(self):
        tracker = self.make_tracker(hold_scans=2)
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        tracker.lock(polygon, static_scene())

        first = tracker.update([])
        second = tracker.update([])
        lost = tracker.update([])

        self.assertTrue(first.held)
        self.assertTrue(second.held)
        self.assertEqual(first.polygon, polygon)
        self.assertTrue(lost.lost)
        self.assertEqual(lost.reason, "locked_slot_lost")

    def test_eight_nearby_points_are_enough_for_stationary_slot_tracking(self):
        tracker = LockedSlotTracker(
            LockedSlotTrackerConfig(
                min_points=8,
                max_correspondence_mm=180.0,
                trim_ratio=1.0,
                iterations=4,
            )
        )
        points = [
            (-700.0, 450.0),
            (-500.0, 420.0),
            (-300.0, 410.0),
            (-100.0, 405.0),
            (100.0, 405.0),
            (300.0, 410.0),
            (500.0, 420.0),
            (700.0, 450.0),
        ]
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))

        locked = tracker.lock(polygon, points)
        tracked = tracker.update(points)

        self.assertTrue(locked.locked)
        self.assertTrue(tracked.tracked, tracked.reason)
        self.assertFalse(tracked.held)

    def test_reanchor_replaces_accumulated_pose_without_resizing_box(self):
        tracker = self.make_tracker()
        points = static_scene()
        initial = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        corrected = ((-275.0, 100.0), (675.0, 100.0), (675.0, 1600.0), (-275.0, 1600.0))
        tracker.lock(initial, points)

        pose = tracker.reanchor(corrected, points)

        self.assertTrue(pose.tracked)
        self.assertEqual(pose.reason, "locked_slot_reanchored")
        self.assertEqual(pose.polygon, corrected)
        self.assertAlmostEqual(
            math.hypot(
                pose.polygon[1][0] - pose.polygon[0][0],
                pose.polygon[1][1] - pose.polygon[0][1],
            ),
            950.0,
            delta=0.01,
        )


class RecordingProjector:
    def __init__(self):
        self.polygons = []

    def project(self, observation):
        return ParkingGeometry(reason="dynamic_slot")

    def project_polygon(self, polygon, **kwargs):
        self.polygons.append(tuple(polygon))
        return ParkingGeometry(reason=kwargs["reason"])


class LockedSlotGeometryEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.points = static_scene()
        self.projector = RecordingProjector()
        self.estimator = LockedSlotGeometryEstimator(
            self.projector,
            LockedSlotTracker(
                LockedSlotTrackerConfig(
                    min_points=10,
                    max_points=120,
                    max_correspondence_mm=180.0,
                    trim_ratio=0.8,
                    iterations=8,
                )
            ),
            width_mm=950.0,
            depth_mm=1500.0,
        )

    @staticmethod
    def observation(
        center_x,
        *,
        pair_observed,
        scan_timestamp=1.0,
        is_new_scan=True,
        smoothed_center_x=None,
        pair_rotation_deg=0.0,
        single_cluster_observed=False,
    ):
        angle = math.radians(pair_rotation_deg)
        depth_x = math.cos(angle)
        depth_y = math.sin(angle)
        width_x = -depth_y
        width_y = depth_x
        raw_gap = (
            RawTwoCarGapMeasurement(
                scan_timestamp=scan_timestamp,
                observed_width_mm=1300.0,
                first_edge_x_right_mm=center_x - 650.0 * width_x,
                first_edge_y_back_mm=-650.0 * width_y,
                second_edge_x_right_mm=center_x + 650.0 * width_x,
                second_edge_y_back_mm=650.0 * width_y,
                slot_depth_x_right=depth_x,
                slot_depth_y_back=depth_y,
            )
            if pair_observed
            else None
        )
        observed_center_x = (
            center_x if smoothed_center_x is None else smoothed_center_x
        )
        return LidarParkingObservation(
            timestamp=scan_timestamp,
            is_new_scan=is_new_scan,
            valid=True,
            car_count=2 if pair_observed else 1,
            gap_found=True,
            gap_confirmed=True,
            gap_pair_observed=pair_observed,
            gap_single_cluster_observed=single_cluster_observed,
            raw_two_car_gap=raw_gap,
            gap_near_edge_x_right_mm=observed_center_x,
            gap_near_edge_y_back_mm=-650.0,
            gap_far_edge_x_right_mm=observed_center_x,
            gap_far_edge_y_back_mm=650.0,
            slot_depth_x_right=depth_x,
            slot_depth_y_back=depth_y,
        )

    def test_large_direct_pair_jump_slew_corrects_even_when_map_localizes(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=True,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(
            geometry.reason,
            "direct_pair_slew_correction",
        )
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.DIRECT_PAIR_RECOVERY,
        )
        self.assertEqual(self.estimator.pose.scan_timestamp, 2.0)
        self.assertAlmostEqual(
            self.estimator.pose.translation_mm,
            50.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1700.0,
            delta=0.01,
        )

    def test_small_direct_pair_difference_applies_bounded_correction(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                960.0,
                pair_observed=True,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair_gated_correction")
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.DIRECT_PAIR_CORRECTION,
        )
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1740.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            self.estimator.pose.translation_mm,
            10.0,
            delta=0.01,
        )

    def test_large_direct_pair_delta_slew_recovers_when_map_is_unavailable(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=True,
                scan_timestamp=2.0,
            ),
            [],
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair_slew_recovery")
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.DIRECT_PAIR_RECOVERY,
        )
        self.assertFalse(self.estimator.pose.lost)
        self.assertTrue(self.estimator.pose.tracked)
        self.assertAlmostEqual(
            self.estimator.pose.translation_mm,
            50.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1700.0,
            delta=0.01,
        )

    def test_single_visible_car_updates_box_when_map_is_unavailable(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=False,
                single_cluster_observed=True,
                scan_timestamp=2.0,
            ),
            [],
            lock_requested=True,
        )

        self.assertEqual(
            geometry.reason,
            "single_car_translation_slew_recovery",
        )
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.SINGLE_CAR_RECOVERY,
        )
        self.assertTrue(self.estimator.pose.tracked)
        self.assertFalse(self.estimator.pose.lost)
        self.assertAlmostEqual(
            self.estimator.pose.translation_mm,
            50.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1700.0,
            delta=0.01,
        )

    def test_single_car_correction_preserves_reference_rotation(self):
        reference = self.estimator.slot_frame.direct_pair_pose(
            self.observation(
                1000.0,
                pair_observed=True,
                pair_rotation_deg=11.0,
            ).raw_two_car_gap
        )
        reference = replace(reference, rotation_deg=-3.5)
        observed = self.estimator._single_car_pose(
            self.observation(
                1040.0,
                pair_observed=False,
                single_cluster_observed=True,
                pair_rotation_deg=0.0,
                scan_timestamp=2.0,
            )
        )

        corrected = self.estimator._single_car_correction(
            reference,
            observed,
        )

        self.assertIsNotNone(corrected)
        self.assertEqual(
            corrected.source,
            SlotPoseSource.SINGLE_CAR_CORRECTION,
        )
        self.assertAlmostEqual(corrected.rotation_deg, -3.5)
        for first_index, second_index in ((0, 1), (0, 3)):
            reference_axis = (
                reference.polygon[second_index][0]
                - reference.polygon[first_index][0],
                reference.polygon[second_index][1]
                - reference.polygon[first_index][1],
            )
            corrected_axis = (
                corrected.polygon[second_index][0]
                - corrected.polygon[first_index][0],
                corrected.polygon[second_index][1]
                - corrected.polygon[first_index][1],
            )
            self.assertAlmostEqual(corrected_axis[0], reference_axis[0])
            self.assertAlmostEqual(corrected_axis[1], reference_axis[1])

    def test_map_failure_uses_scan_rotation_before_single_car_translation(self):
        estimator = LockedSlotGeometryEstimator(
            self.projector,
            LockedSlotTracker(
                LockedSlotTrackerConfig(
                    min_points=10,
                    max_points=120,
                    max_correspondence_mm=180.0,
                    map_max_correspondence_mm=1.0,
                    trim_ratio=0.8,
                    iterations=8,
                )
            ),
            width_mm=950.0,
            depth_mm=1500.0,
        )
        initial_points = static_scene()
        estimator.update(
            self.observation(1000.0, pair_observed=True),
            initial_points,
            lock_requested=True,
        )
        angle = math.radians(4.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated_points = [
            (
                cosine * x - sine * y,
                sine * x + cosine * y,
            )
            for x, y in initial_points
        ]

        geometry = estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                single_cluster_observed=True,
                pair_rotation_deg=0.0,
                scan_timestamp=2.0,
            ),
            rotated_points,
            lock_requested=True,
        )

        self.assertEqual(
            geometry.reason,
            "single_car_translation_correction",
        )
        self.assertEqual(
            estimator.pose.source,
            SlotPoseSource.SINGLE_CAR_CORRECTION,
        )
        depth_axis = (
            estimator.pose.polygon[3][0] - estimator.pose.polygon[0][0],
            estimator.pose.polygon[3][1] - estimator.pose.polygon[0][1],
        )
        depth_angle_deg = math.degrees(
            math.atan2(depth_axis[1], depth_axis[0])
        )
        self.assertGreater(depth_angle_deg, 2.0)
        self.assertAlmostEqual(
            depth_angle_deg,
            -estimator.pose.rotation_deg,
            delta=0.2,
        )

    def test_unarmed_lock_is_released_when_forward_alignment_resumes(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                900.0,
                pair_observed=True,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=False,
        )

        self.assertFalse(self.estimator.pose.locked)
        self.assertIsNone(self.estimator.landmark_map)
        self.assertEqual(geometry.reason, "dynamic_slot")

    def test_first_direct_pair_freezes_two_car_slot_landmark_map(self):
        self.estimator.update(
            self.observation(
                1000.0,
                pair_observed=True,
                scan_timestamp=1.0,
            ),
            self.points,
            lock_requested=True,
        )
        frozen = self.estimator.landmark_map

        self.assertIsNotNone(frozen)
        self.assertTrue(frozen.frozen)
        self.assertEqual(frozen.source_scan_timestamp, 1.0)
        self.assertEqual(len(frozen.landmarks), 2)
        self.assertEqual(
            tuple(item.landmark_id for item in frozen.landmarks),
            ("border_car_first", "border_car_second"),
        )
        self.assertAlmostEqual(frozen.landmarks[0].local_x_mm, -650.0)
        self.assertAlmostEqual(frozen.landmarks[1].local_x_mm, 650.0)
        self.assertAlmostEqual(frozen.landmarks[0].local_y_mm, 0.0)
        self.assertAlmostEqual(frozen.landmarks[1].local_y_mm, 0.0)

    def test_later_direct_pair_slew_moves_pose_but_not_frozen_map(self):
        self.estimator.update(
            self.observation(
                1000.0,
                pair_observed=True,
                scan_timestamp=1.0,
            ),
            self.points,
            lock_requested=True,
        )
        frozen = self.estimator.landmark_map

        self.estimator.update(
            self.observation(
                700.0,
                pair_observed=True,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertIs(self.estimator.landmark_map, frozen)
        self.assertEqual(self.estimator.landmark_map.source_scan_timestamp, 1.0)
        displayed = dict(
            self.estimator.landmark_map.points_in_vehicle_frame(
                self.estimator.pose,
            )
        )
        self.assertAlmostEqual(displayed["border_car_first"][0], 950.0)
        self.assertAlmostEqual(displayed["border_car_first"][1], -650.0)
        self.assertAlmostEqual(displayed["border_car_second"][0], 950.0)
        self.assertAlmostEqual(displayed["border_car_second"][1], 650.0)

    def test_direct_pair_rotation_is_corrected_more_aggressively_than_translation(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                1040.0,
                pair_observed=True,
                pair_rotation_deg=4.0,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair_gated_correction")
        self.assertAlmostEqual(self.estimator.pose.translation_mm, 10.0)
        self.assertAlmostEqual(self.estimator.pose.rotation_deg, 2.2)

    def test_large_direct_pair_rotation_is_slew_limited_not_rejected(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                1000.0,
                pair_observed=True,
                pair_rotation_deg=12.0,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair_slew_correction")
        self.assertAlmostEqual(self.estimator.pose.rotation_deg, 3.0)

    def test_reset_discards_frozen_landmark_map_for_next_mission(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )
        self.assertIsNotNone(self.estimator.landmark_map)

        self.estimator.reset()

        self.assertIsNone(self.estimator.landmark_map)

    def test_direct_pose_uses_raw_pair_instead_of_smoothed_box(self):
        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=True,
                smoothed_center_x=1000.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair")
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1450.0,
            delta=0.01,
        )

    def test_duplicate_scan_keeps_direct_pose_without_running_fallback(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )
        initial_pose = self.estimator.pose
        shifted_points = [(x - 40.0, y - 25.0) for x, y in self.points]

        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=False,
                is_new_scan=False,
            ),
            shifted_points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "direct_pair")
        self.assertEqual(self.estimator.pose, initial_pose)

    def test_single_border_new_scan_uses_frozen_map_localization(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )
        initial_polygon = self.estimator.pose.polygon

        geometry = self.estimator.update(
            self.observation(
                700.0,
                pair_observed=False,
                scan_timestamp=2.0,
            ),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "slot_map_localized")
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.MAP_LOCALIZATION,
        )
        for before, after in zip(initial_polygon, self.estimator.pose.polygon):
            self.assertAlmostEqual(after[0], before[0], delta=0.01)
            self.assertAlmostEqual(after[1], before[1], delta=0.01)

    def test_map_failure_uses_only_bounded_short_icp_fallback(self):
        estimator = LockedSlotGeometryEstimator(
            self.projector,
            LockedSlotTracker(
                LockedSlotTrackerConfig(
                    min_points=10,
                    max_points=120,
                    max_correspondence_mm=180.0,
                    map_max_correspondence_mm=5.0,
                    map_icp_fallback_max_scans=1,
                    trim_ratio=0.8,
                    iterations=8,
                )
            ),
            width_mm=950.0,
            depth_mm=1500.0,
        )
        initial_points = two_car_surfaces()
        estimator.update(
            self.observation(1000.0, pair_observed=True),
            initial_points,
            lock_requested=True,
        )
        shifted_once = [(x - 20.0, y - 12.0) for x, y in initial_points]

        first = estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                scan_timestamp=2.0,
            ),
            shifted_once,
            lock_requested=True,
        )

        self.assertEqual(first.reason, "short_icp_fallback")
        self.assertEqual(
            estimator.pose.source,
            SlotPoseSource.TRACKER_FALLBACK,
        )
        self.assertFalse(estimator.pose.lost)

        shifted_twice = [(x - 40.0, y - 24.0) for x, y in initial_points]
        second = estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                scan_timestamp=3.0,
            ),
            shifted_twice,
            lock_requested=True,
        )

        self.assertTrue(estimator.pose.lost)
        self.assertEqual(second.reason, "short_icp_fallback_exhausted")

    def test_all_localization_failures_report_lost_on_same_scan(self):
        estimator = LockedSlotGeometryEstimator(
            self.projector,
            LockedSlotTracker(
                LockedSlotTrackerConfig(
                    min_points=10,
                    map_max_correspondence_mm=5.0,
                    map_icp_fallback_max_scans=2,
                )
            ),
            width_mm=950.0,
            depth_mm=1500.0,
        )
        estimator.update(
            self.observation(1000.0, pair_observed=True),
            two_car_surfaces(),
            lock_requested=True,
        )

        geometry = estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                scan_timestamp=2.0,
            ),
            [],
            lock_requested=True,
        )

        self.assertTrue(estimator.pose.lost)
        self.assertFalse(estimator.pose.held)
        self.assertTrue(
            geometry.reason.startswith("all_slot_localization_failed:")
        )

    def test_bilateral_flat_surfaces_still_localize_depth(self):
        flat_points = [
            (1000.0, y)
            for y in range(-1000, 1001, 100)
            if abs(y) >= 200
        ]
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            flat_points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                scan_timestamp=2.0,
            ),
            flat_points,
            lock_requested=True,
        )

        self.assertTrue(self.estimator.pose.tracked)
        self.assertFalse(self.estimator.pose.lost)
        self.assertFalse(self.estimator.pose.held)
        self.assertTrue(self.estimator.pose.bilateral_support)
        self.assertEqual(geometry.reason, "slot_map_localized")

    def test_one_sided_scan_without_depth_span_uses_short_icp(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            two_car_surfaces(),
            lock_requested=True,
        )
        flat_current = [
            (1000.0, float(y))
            for y in range(-700, -599, 10)
        ]

        geometry = self.estimator.update(
            self.observation(
                1000.0,
                pair_observed=False,
                scan_timestamp=2.0,
            ),
            flat_current,
            lock_requested=True,
        )

        self.assertFalse(self.estimator.pose.lost)
        self.assertTrue(self.estimator.pose.tracked)
        self.assertEqual(
            self.estimator.pose.source,
            SlotPoseSource.TRACKER_FALLBACK,
        )
        self.assertEqual(geometry.reason, "short_icp_fallback")
        self.assertTrue(self.estimator.tracker.locked)


class FixedSlotFrameTest(unittest.TestCase):
    def test_direct_pair_pose_has_fixed_official_dimensions(self):
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        measurement = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=1000.0,
            first_edge_y_back_mm=-650.0,
            second_edge_x_right_mm=1000.0,
            second_edge_y_back_mm=650.0,
            slot_depth_x_right=1.0,
            slot_depth_y_back=0.0,
        )

        pose = frame.direct_pair_pose(measurement)

        self.assertIsNotNone(pose)
        self.assertEqual(pose.source, SlotPoseSource.DIRECT_PAIR)
        self.assertEqual(pose.scan_timestamp, 3.0)
        width = math.hypot(
            pose.polygon[1][0] - pose.polygon[0][0],
            pose.polygon[1][1] - pose.polygon[0][1],
        )
        depth = math.hypot(
            pose.polygon[3][0] - pose.polygon[0][0],
            pose.polygon[3][1] - pose.polygon[0][1],
        )
        self.assertAlmostEqual(width, 950.0, delta=0.01)
        self.assertAlmostEqual(depth, 1500.0, delta=0.01)

    def test_direct_pair_pose_is_invariant_to_border_car_order(self):
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        first = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=-650.0,
            first_edge_y_back_mm=20.0,
            second_edge_x_right_mm=650.0,
            second_edge_y_back_mm=-10.0,
            slot_depth_x_right=0.02,
            slot_depth_y_back=1.0,
        )
        swapped = RawTwoCarGapMeasurement(
            scan_timestamp=4.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=650.0,
            first_edge_y_back_mm=-10.0,
            second_edge_x_right_mm=-650.0,
            second_edge_y_back_mm=20.0,
            slot_depth_x_right=0.02,
            slot_depth_y_back=1.0,
        )

        first_pose = frame.direct_pair_pose(first)
        swapped_pose = frame.direct_pair_pose(swapped)

        self.assertIsNotNone(first_pose)
        self.assertIsNotNone(swapped_pose)
        for first_corner, swapped_corner in zip(
            first_pose.polygon,
            swapped_pose.polygon,
        ):
            self.assertAlmostEqual(first_corner[0], swapped_corner[0])
            self.assertAlmostEqual(first_corner[1], swapped_corner[1])

    def test_landmark_map_requires_direct_pair_pose_and_is_immutable(self):
        measurement = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=1000.0,
            first_edge_y_back_mm=-650.0,
            second_edge_x_right_mm=1000.0,
            second_edge_y_back_mm=650.0,
            slot_depth_x_right=1.0,
            slot_depth_y_back=0.0,
        )
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        direct_pose = frame.direct_pair_pose(measurement)

        landmark_map = FrozenSlotLandmarkMap.from_direct_pair(
            direct_pose,
            measurement,
            slot_width_mm=950.0,
            slot_depth_mm=1500.0,
        )

        self.assertIsNotNone(landmark_map)
        with self.assertRaises(FrozenInstanceError):
            landmark_map.source_scan_timestamp = 4.0
        self.assertIsNone(
            FrozenSlotLandmarkMap.from_direct_pair(
                LockedSlotPose(),
                measurement,
                slot_width_mm=950.0,
                slot_depth_mm=1500.0,
            )
        )

    def test_landmark_map_waits_for_support_from_both_cars(self):
        measurement = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=-650.0,
            first_edge_y_back_mm=0.0,
            second_edge_x_right_mm=650.0,
            second_edge_y_back_mm=0.0,
            slot_depth_x_right=0.0,
            slot_depth_y_back=1.0,
        )
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        direct_pose = frame.direct_pair_pose(measurement)
        one_car_only = [
            (-650.0, float(depth))
            for depth in range(0, 600, 100)
        ]

        landmark_map = FrozenSlotLandmarkMap.from_direct_pair(
            direct_pose,
            measurement,
            slot_width_mm=950.0,
            slot_depth_mm=1500.0,
            vehicle_points=one_car_only,
            min_surface_points=6,
            min_points_per_landmark=2,
        )

        self.assertIsNone(landmark_map)

    def test_four_bilateral_returns_localize_against_quality_map(self):
        measurement = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=-650.0,
            first_edge_y_back_mm=0.0,
            second_edge_x_right_mm=650.0,
            second_edge_y_back_mm=0.0,
            slot_depth_x_right=0.0,
            slot_depth_y_back=1.0,
        )
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        direct_pose = frame.direct_pair_pose(measurement)
        map_points = [
            (side_x, float(depth_y))
            for side_x in (-650.0, 650.0)
            for depth_y in (0, 30, 60)
        ]
        landmark_map = FrozenSlotLandmarkMap.from_direct_pair(
            direct_pose,
            measurement,
            slot_width_mm=950.0,
            slot_depth_mm=1500.0,
            vehicle_points=map_points,
            min_surface_points=6,
            min_points_per_landmark=2,
        )
        current_points = [
            (x - 20.0, y - 12.0)
            for x, y in (
                map_points[0],
                map_points[2],
                map_points[3],
                map_points[5],
            )
        ]

        result = landmark_map.localize(
            current_points,
            direct_pose,
            LockedSlotTrackerConfig(
                map_min_points=6,
                map_min_points_per_landmark=2,
                map_min_depth_span_mm=120.0,
                trim_ratio=1.0,
                iterations=8,
            ),
            scan_timestamp=4.0,
        )

        self.assertEqual(result.status, MapLocalizationStatus.LOCALIZED)
        self.assertEqual(result.pose.support_points, 4)
        self.assertTrue(result.pose.bilateral_support)
        self.assertLess(result.pose.depth_span_mm, 120.0)

    def test_scan_to_frozen_map_localizes_vehicle_motion(self):
        measurement = RawTwoCarGapMeasurement(
            scan_timestamp=3.0,
            observed_width_mm=1300.0,
            first_edge_x_right_mm=-650.0,
            first_edge_y_back_mm=0.0,
            second_edge_x_right_mm=650.0,
            second_edge_y_back_mm=0.0,
            slot_depth_x_right=0.0,
            slot_depth_y_back=1.0,
        )
        frame = FixedSlotFrame(width_mm=950.0, depth_mm=1500.0)
        direct_pose = frame.direct_pair_pose(measurement)
        map_points = [
            (side_x, float(depth_y))
            for side_x in (-650.0, 650.0)
            for depth_y in range(-200, 1001, 100)
        ]
        landmark_map = FrozenSlotLandmarkMap.from_direct_pair(
            direct_pose,
            measurement,
            slot_width_mm=950.0,
            slot_depth_mm=1500.0,
            vehicle_points=map_points,
        )
        current_points = [
            (x - 20.0, y - 12.0) for x, y in map_points
        ]

        result = landmark_map.localize(
            current_points,
            direct_pose,
            LockedSlotTrackerConfig(
                min_points=10,
                map_min_points=10,
                trim_ratio=0.8,
                iterations=8,
            ),
            scan_timestamp=4.0,
        )

        self.assertEqual(result.status, MapLocalizationStatus.LOCALIZED)
        self.assertEqual(
            result.pose.source,
            SlotPoseSource.MAP_LOCALIZATION,
        )
        self.assertAlmostEqual(
            result.pose.polygon[0][0],
            direct_pose.polygon[0][0] - 20.0,
            delta=2.0,
        )
        self.assertAlmostEqual(
            result.pose.polygon[0][1],
            direct_pose.polygon[0][1] - 12.0,
            delta=2.0,
        )


if __name__ == "__main__":
    unittest.main()
