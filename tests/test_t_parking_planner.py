import unittest
import math
from dataclasses import replace

from skku_autocar.estimation.parking_geometry import (
    ParkingGeometry,
    ParkingGeometryDepthStabilizer,
    ParkingLine,
)
from skku_autocar.estimation.parking_lidar import CarCluster, LidarParkingObservation
from skku_autocar.planning.reverse_parking_path import ReversePath, ReversePathConfig
from skku_autocar.planning.t_parking_planner import (
    ParkingPlannerConfig,
    ParkingState,
    TParkingPlanner,
)


def geometry(
    heading=20.0,
    lateral=0.4,
    remaining=100.0,
    reason="parking_bay",
    fully_inside=True,
    selection_mode="line_only",
    observed_lines=2,
    observed_cars=0,
):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=remaining is not None,
        heading_error_deg=heading,
        lateral_error_norm=lateral,
        depth_remaining_px=remaining,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.0,
        slot_direction_y=-1.0,
        stop_target_x_px=300.0 + 125.0 * lateral,
        stop_target_y_px=100.0,
        vehicle_fully_inside=fully_inside,
        confidence=0.9,
        reason=reason,
        selection_mode=selection_mode,
        observed_line_count=observed_lines,
        observed_car_count=observed_cars,
    )


def lidar_gap(
    entry_error=200.0,
    reached=False,
    unsafe=False,
    pair_observed=False,
    safety_x=None,
):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        unsafe=unsafe,
        safety_center_x_right_mm=safety_x,
        observed_points=20,
        car_count=2,
        first_car_seen=True,
        second_car_seen=True,
        gap_found=True,
        gap_confirmed=True,
        gap_pair_observed=pair_observed,
        gap_width_mm=1375.0,
        gap_center_y_back_mm=380.0 if not reached else 180.0,
        entry_target_y_back_mm=180.0,
        entry_error_mm=entry_error,
        entry_reached=reached,
        reason="gap_confirmed",
    )


def prealign_lidar(
    slot_heading_deg=90.0,
    entry_bearing_deg=90.0,
    distance_mm=1000.0,
    unsafe=False,
    first_car_bearing_deg=130.0,
    fresh_pair=True,
):
    slot_angle = math.radians(slot_heading_deg)
    bearing = math.radians(entry_bearing_deg)
    rear_axle_y = -300.0
    first_car_bearing = math.radians(first_car_bearing_deg)
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        unsafe=unsafe,
        observed_points=20,
        car_count=2,
        gap_found=True,
        gap_confirmed=True,
        gap_pair_observed=fresh_pair,
        coasted=not fresh_pair,
        gap_width_mm=1375.0,
        gap_center_x_right_mm=math.sin(bearing) * distance_mm,
        gap_center_y_back_mm=rear_axle_y + math.cos(bearing) * distance_mm,
        entry_target_y_back_mm=rear_axle_y,
        slot_depth_x_right=math.sin(slot_angle),
        slot_depth_y_back=math.cos(slot_angle),
        first_car_slot_edge_x_right_mm=math.sin(first_car_bearing) * 1400.0,
        first_car_slot_edge_y_back_mm=-math.cos(first_car_bearing) * 1400.0,
        reason="gap_confirmed",
    )


def first_car_lidar(turn_reached=False, turn_error=100.0):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        observed_points=10,
        car_count=1,
        first_car_seen=True,
        first_car_confirmed=True,
        first_car_slot_edge_x_right_mm=1500.0,
        first_car_slot_edge_y_back_mm=-650.0 - turn_error,
        first_car_turn_error_mm=turn_error,
        first_car_turn_reached=turn_reached,
        reason="first_car_confirmed",
    )


class TParkingPlannerTest(unittest.TestCase):
    def make_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                reverse_entry_continuous_steering=False,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
                exit_straight_s=3.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )

    def make_prealign_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_speed=14,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                prealign_timeout_s=2.0,
                prealign_confirm_frames=2,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
            )
        )

    def make_correction_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                correction_enabled=True,
                correction_steer_settle_s=0.0,
                correction_forward_s=0.5,
                correction_reverse_s=0.5,
                correction_min_reverse_s=0.0,
                correction_depth_trigger_px=1000.0,
                correction_heading_trigger_deg=15.0,
                correction_lateral_trigger_norm=0.30,
                correction_trigger_frames=2,
                correction_max_attempts=2,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                reverse_entry_continuous_steering=False,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )

    @staticmethod
    def enter_prealign(planner):
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        return planner.update(geometry(), lidar_gap(0.0, reached=True), 0.1)

    def arm_reverse(self, planner):
        planner.start(0.0)
        one_car = LidarParkingObservation(
            timestamp=0.0,
            valid=True,
            observed_points=10,
            car_count=1,
            first_car_seen=True,
            first_car_confirmed=True,
            reason="one_parked_car",
        )
        planner.update(geometry(), one_car, 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        return planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)

    def test_complete_sequence_stops_inside_locked_slot(self):
        planner = self.make_planner()
        planner.start(0.0)
        one_car = LidarParkingObservation(
            timestamp=0.0,
            valid=True,
            observed_points=10,
            car_count=1,
            first_car_seen=True,
            first_car_confirmed=True,
            reason="one_parked_car",
        )

        tracking = planner.update(geometry(), one_car, 0.0)
        positioned = planner.update(geometry(), lidar_gap(), 0.1)
        verifying = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        path_plan = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        armed = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        aligned = planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        parked = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )
        hold = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.5,
            right_ultrasonic_mm=500.0,
        )
        exit_right = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.7,
            right_ultrasonic_mm=500.0,
        )
        exit_straight = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            5.4,
            right_ultrasonic_mm=500.0,
        )
        exit_done = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            8.5,
            right_ultrasonic_mm=500.0,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(positioned.state, ParkingState.POSITION_REAR_AXLE)
        self.assertEqual(verifying.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(path_plan.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIsNotNone(armed.path)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(aligned.command.steering, 0)
        self.assertEqual(
            aligned.reason,
            "following_slot_center:entry_aligned_released",
        )
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertTrue(parked.command.brake)
        self.assertEqual(hold.state, ParkingState.PARKED)
        self.assertEqual(hold.reason, "parked_hold")
        self.assertTrue(hold.command.brake)
        self.assertEqual(exit_right.state, ParkingState.EXIT_RIGHT)
        self.assertEqual(exit_right.command.speed, planner.config.exit_speed)
        self.assertEqual(exit_right.command.steering, planner.config.exit_turn_steering)
        self.assertEqual(exit_straight.state, ParkingState.EXIT_STRAIGHT)
        self.assertEqual(exit_straight.command.speed, planner.config.exit_speed)
        self.assertEqual(exit_straight.command.steering, 0)
        self.assertEqual(exit_done.state, ParkingState.EXIT_DONE)
        self.assertTrue(exit_done.command.brake)

    def test_reverse_path_must_be_confirmed_before_reverse_entry(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=3,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)

        first = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        second = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.5)
        armed = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.6)

        self.assertEqual(first.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(first.reason, "reverse_path_confirming:1/3")
        self.assertIsNotNone(first.path)
        self.assertEqual(second.reason, "reverse_path_confirming:2/3")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(armed.reason, "reverse_path_armed")

    def test_reverse_path_confirm_counter_tolerates_brief_path_loss(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=3,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        missing_geometry = ParkingGeometry(reason="lidar_slot_box_unavailable")
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)

        first = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        second = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.5)
        lost = planner.update(missing_geometry, lidar_gap(0.0, reached=True), 0.6)
        recovered = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.7)
        armed = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.8)

        self.assertEqual(first.reason, "reverse_path_confirming:1/3")
        self.assertEqual(second.reason, "reverse_path_confirming:2/3")
        self.assertIn("confirm=1/3", lost.reason)
        self.assertEqual(recovered.reason, "reverse_path_confirming:2/3")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_side_ultrasonic_is_ignored_during_exit(self):
        planner = self.make_planner()
        planner.start(0.0)
        one_car = LidarParkingObservation(
            timestamp=0.0,
            valid=True,
            observed_points=10,
            car_count=1,
            first_car_seen=True,
            first_car_confirmed=True,
            reason="one_parked_car",
        )

        planner.update(geometry(), one_car, 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        moving = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.7,
            right_ultrasonic_mm=95.0,
        )

        self.assertEqual(moving.state, ParkingState.EXIT_RIGHT)
        self.assertFalse(moving.command.brake)
        self.assertEqual(moving.command.steering, planner.config.exit_turn_steering)

    def test_search_stops_after_rollout_while_waiting_for_lidar(self):
        planner = self.make_planner()
        planner.start(0.0)

        searching = planner.update(
            geometry(),
            LidarParkingObservation(reason="no_scan"),
            0.1,
        )

        self.assertEqual(searching.state, ParkingState.SEARCH_CARS)
        self.assertEqual(searching.command.speed, 0)
        self.assertEqual(searching.command.steering, 0)
        self.assertTrue(searching.command.brake)
        self.assertEqual(searching.reason, "lidar_unavailable:no_scan")

    def test_straight_trim_only_offsets_intentional_straight_steering(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-10,
                prealign_steering=-150,
                max_steering=150,
                start_forward_s=1.0,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        rollout = planner.update(
            geometry(),
            LidarParkingObservation(valid=True, reason="no_objects"),
            0.1,
        )
        waiting = planner.update(
            geometry(),
            LidarParkingObservation(valid=True, reason="no_objects"),
            1.1,
        )

        self.assertGreater(rollout.command.speed, 0)
        self.assertEqual(rollout.command.steering, 0)
        self.assertGreater(waiting.command.speed, 0)
        self.assertEqual(waiting.command.steering, 0)
        self.assertEqual(planner._straight_steering(), -10)
        self.assertEqual(planner._prealign_steering(), -150)
        self.assertEqual(planner._fixed_right_entry_steering(), 150)

    def test_unconfirmed_first_car_does_not_arm_prealign(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=1.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)
        unconfirmed = LidarParkingObservation(
            timestamp=1.0,
            valid=True,
            observed_points=10,
            car_roi_points=2,
            car_count=1,
            first_car_seen=True,
            first_car_confirmed=False,
            first_car_turn_reached=False,
            reason="one_parked_car",
        )

        confirming = planner.update(geometry(), unconfirmed, 0.1)
        still_confirming = planner.update(geometry(), unconfirmed, 2.0)

        self.assertEqual(confirming.state, ParkingState.SEARCH_CARS)
        self.assertEqual(confirming.command.speed, planner.config.first_car_approach_speed)
        self.assertEqual(confirming.command.steering, 0)
        self.assertEqual(confirming.reason, "first_car_seen:waiting_for_confirmation")
        self.assertEqual(still_confirming.state, ParkingState.SEARCH_CARS)
        self.assertEqual(still_confirming.command.steering, 0)

    def test_confirmed_first_car_keeps_creeping_until_turn_target_is_reached(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=1.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        tracking = planner.update(geometry(), first_car_lidar(turn_reached=False), 0.1)
        still_creeping = planner.update(
            geometry(),
            first_car_lidar(turn_reached=False),
            5.0,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(still_creeping.state, ParkingState.TRACK_GAP)
        self.assertEqual(still_creeping.command.speed, planner.config.first_car_approach_speed)
        self.assertEqual(still_creeping.command.steering, 0)
        self.assertEqual(still_creeping.reason, "first_car_creeping_to_turn_point")

    def test_gap_does_not_bypass_a_confirmed_first_car_corner_target(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        planner.update(geometry(), first_car_lidar(turn_reached=False), 0.1)
        gap_before_corner = planner.update(geometry(), lidar_gap(), 0.2)
        corner = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=0.0),
            0.3,
        )

        self.assertEqual(gap_before_corner.state, ParkingState.TRACK_GAP)
        self.assertEqual(gap_before_corner.command.steering, 0)
        self.assertEqual(corner.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(corner.command.steering, planner.config.prealign_steering)

    def test_preemptive_gap_confirm_routes_to_turn_flow_not_rear_axle_reverse(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=True,
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=1.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        # gap이 확정됐지만 first_car는 선형성 게이트를 통과 못한 케이스
        # (first_car_confirmed=False, gap_confirmed=True). preemptive 모드에서는
        # POSITION_REAR_AXLE 후진 정렬이 아니라 TRACK_GAP 선회 흐름으로 가야 한다.
        routed = planner.update(geometry(), lidar_gap(), 0.1)

        self.assertEqual(routed.state, ParkingState.TRACK_GAP)
        self.assertNotEqual(routed.state, ParkingState.POSITION_REAR_AXLE)

    def test_start_rollout_drives_straight_even_with_immediate_first_car(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=5.0,
                first_car_straight_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            4.9,
            left_ultrasonic_mm=500.0,
        )
        after_rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            5.1,
        )
        self.assertEqual(rollout.state, ParkingState.SEARCH_CARS)
        self.assertEqual(rollout.command.speed, planner.config.search_speed)
        self.assertEqual(rollout.command.steering, 0)
        self.assertFalse(rollout.command.brake)
        self.assertEqual(rollout.reason, "start_forward_rollout")
        self.assertEqual(after_rollout.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(after_rollout.command.speed, 0)
        self.assertEqual(after_rollout.command.steering, planner.config.prealign_steering)
        self.assertEqual(
            after_rollout.reason,
            "first_car_edge_aligned:settling_max_left",
        )

    def test_initial_search_trim_ends_permanently_after_first_car_seen(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=1.0,
                initial_search_steering_trim=-10,
                straight_steering_trim=0,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        before_car = planner.update(
            geometry(),
            LidarParkingObservation(valid=True),
            0.1,
        )
        first_seen = planner.update(
            geometry(),
            LidarParkingObservation(valid=True, first_car_seen=True),
            0.2,
        )
        after_dropout = planner.update(
            geometry(),
            LidarParkingObservation(valid=True),
            0.3,
        )

        self.assertEqual(before_car.command.steering, -10)
        self.assertEqual(first_seen.command.steering, 0)
        self.assertEqual(after_dropout.command.steering, 0)

    def test_non_right_lidar_clusters_do_not_trigger_prealign(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=1.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)
        non_right_cluster = LidarParkingObservation(
            timestamp=1.0,
            valid=True,
            observed_points=8,
            car_count=1,
            first_car_seen=False,
            first_car_confirmed=False,
            gap_found=False,
            gap_confirmed=False,
            reason="searching_for_parked_cars",
        )

        searching = planner.update(geometry(), non_right_cluster, 0.1)
        still_searching = planner.update(geometry(), non_right_cluster, 2.0)

        self.assertEqual(searching.state, ParkingState.SEARCH_CARS)
        self.assertEqual(searching.command.speed, planner.config.search_speed)
        self.assertEqual(searching.command.steering, 0)
        self.assertEqual(searching.reason, "searching_for_parked_cars")
        self.assertEqual(still_searching.state, ParkingState.SEARCH_CARS)
        self.assertEqual(still_searching.command.speed, planner.config.search_speed)
        self.assertEqual(still_searching.command.steering, 0)

    def test_object_detection_slowdown_survives_short_detection_dropout(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                detection_slow_hold_s=1.5,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)
        detected = LidarParkingObservation(valid=True, car_count=1, first_car_seen=True)
        missing = LidarParkingObservation(valid=True, car_count=0)

        first = planner.update(ParkingGeometry(), detected, 1.0)
        held = planner.update(ParkingGeometry(), missing, 2.0)
        released = planner.update(ParkingGeometry(), missing, 2.6)

        self.assertEqual(first.command.speed, planner.config.first_car_approach_speed)
        self.assertEqual(held.command.speed, planner.config.first_car_approach_speed)
        self.assertEqual(held.reason, "right_lidar_detected:slow_hold")
        self.assertEqual(released.command.speed, planner.config.search_speed)

    def test_camera_car_does_not_arm_drive_state_before_lidar_edge(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        plan = planner.update(
            ParkingGeometry(observed_car_count=1),
            LidarParkingObservation(valid=True),
            1.0,
        )
        waiting = planner.update(
            ParkingGeometry(),
            LidarParkingObservation(valid=True),
            2.7,
        )
        turn = planner.update(
            ParkingGeometry(),
            first_car_lidar(turn_reached=True, turn_error=0.0),
            2.8,
        )

        self.assertEqual(plan.state, ParkingState.SEARCH_CARS)
        self.assertEqual(plan.command.speed, planner.config.search_speed)
        self.assertEqual(plan.reason, "searching_for_parked_cars")
        self.assertEqual(waiting.state, ParkingState.SEARCH_CARS)
        self.assertEqual(waiting.command.steering, 0)
        self.assertEqual(turn.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turn.command.steering, planner.config.prealign_steering)

    def test_prealign_waits_for_two_lidar_cars_even_when_bev_path_exists(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_enabled=True,
                prealign_speed=35,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                prealign_gap_acquire_timeout_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        tracking = planner.update(geometry(), first_car_lidar(), 0.1)
        turn_point = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.2,
        )
        still_searching = planner.update(
            geometry(),
            LidarParkingObservation(
                timestamp=1.0,
                valid=True,
                observed_points=8,
                car_count=1,
                first_car_seen=True,
                first_car_confirmed=False,
                reason="one_parked_car",
            ),
            60.0,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(turn_point.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(still_searching.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turn_point.command.speed, 0)
        self.assertEqual(still_searching.command.speed, planner.config.prealign_speed)
        self.assertEqual(turn_point.command.steering, planner.config.prealign_steering)
        self.assertEqual(still_searching.command.steering, planner.config.prealign_steering)
        self.assertFalse(turn_point.command.brake)
        self.assertFalse(still_searching.command.brake)
        self.assertIn("prealign_waiting_for_lidar_cars", still_searching.reason)

    def test_prealign_missing_tracked_slot_keeps_moving(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        waiting = planner.update(
            ParkingGeometry(reason="not_found"),
            prealign_lidar(),
            0.2,
        )

        self.assertEqual(waiting.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(waiting.command.speed, planner.config.prealign_speed)
        self.assertEqual(waiting.command.steering, planner.config.prealign_steering)
        self.assertFalse(waiting.command.brake)
        self.assertEqual(waiting.reason, "prealign_waiting_for_yolo_cars:0/1")

    def test_prealign_visible_bev_path_only_needs_stable_confirmation(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        seen = planner.update(
            geometry(
                heading=44.0,
                lateral=-0.42,
                remaining=655.0,
                reason="lidar_slot_box",
                observed_cars=1,
            ),
            prealign_lidar(
                slot_heading_deg=44.0,
                entry_bearing_deg=46.0,
                distance_mm=1590.0,
            ),
            0.2,
        )

        self.assertEqual(seen.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(seen.command.speed, planner.config.prealign_speed)
        self.assertEqual(seen.command.steering, planner.config.prealign_steering)
        self.assertFalse(seen.command.brake)
        self.assertEqual(seen.reason, "prealign_bev_path_confirming:1/2")

    def test_prealign_allows_large_heading_for_reverse_path_correction(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        impossible = geometry(
            heading=70.0,
            lateral=-0.42,
            remaining=655.0,
            reason="lidar_slot_box",
            observed_cars=1,
        )
        lidar = prealign_lidar(
            slot_heading_deg=70.0,
            entry_bearing_deg=65.0,
            distance_mm=1590.0,
        )

        first = planner.update(impossible, lidar, 0.2)
        self.assertFalse(planner._lidar_y0_terminal_armed)
        second = planner.update(impossible, lidar, 0.3)

        self.assertEqual(first.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(first.reason, "prealign_bev_path_confirming:1/2")
        self.assertEqual(second.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(
            second.reason,
            "prealign_sensor_crosscheck_ready:reverse_path_armed",
        )
        self.assertTrue(planner._lidar_y0_terminal_armed)

    def test_prealign_uses_curve_reverse_when_box_path_is_feasible(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        lidar = prealign_lidar(
            slot_heading_deg=40.0,
            entry_bearing_deg=40.0,
            distance_mm=1200.0,
        )

        confirming = planner.update(
            geometry(
                heading=40.0,
                lateral=0.2,
                remaining=650.0,
                reason="lidar_slot_box",
                observed_cars=1,
            ),
            lidar,
            0.2,
        )
        ready = planner.update(
            geometry(
                heading=40.0,
                lateral=0.2,
                remaining=650.0,
                reason="lidar_slot_box",
                observed_cars=1,
            ),
            lidar,
            0.3,
        )

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(ready.reason, "prealign_sensor_crosscheck_ready:reverse_path_armed")
        self.assertTrue(ready.command.brake)

    def test_prealign_curve_uses_geometry_heading_instead_of_raw_lidar_heading(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        lidar = prealign_lidar(
            slot_heading_deg=65.0,
            entry_bearing_deg=20.0,
            distance_mm=1500.0,
        )
        slot_geometry = geometry(
            heading=40.0,
            lateral=0.2,
            remaining=650.0,
            reason="lidar_slot_box",
            observed_cars=1,
        )

        confirming = planner.update(slot_geometry, lidar, 0.2)
        ready = planner.update(slot_geometry, lidar, 0.3)

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(ready.reason, "prealign_sensor_crosscheck_ready:reverse_path_armed")
        self.assertTrue(ready.command.brake)

    def test_reverse_entry_switches_to_right_steering_after_left_prealign(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_steering=-150,
            )
        )
        path = ReversePath(
            found=True,
            points=((300.0, 570.0), (360.0, 500.0), (420.0, 320.0)),
            lookahead_point=(360.0, 500.0),
            curvature_per_px=-planner.path_generator.config.full_steering_curvature_per_px,
            reason="reverse_path_ready",
        )

        steering = planner._path_steering(path)

        self.assertEqual(planner.config.prealign_steering, -150)
        self.assertGreater(steering, 0)

    def test_reverse_entry_uses_minimum_visible_steering_for_small_curve(self):
        planner = self.make_planner()
        path = ReversePath(
            found=True,
            points=((300.0, 570.0), (320.0, 500.0), (350.0, 320.0)),
            lookahead_point=(320.0, 500.0),
            curvature_per_px=0.001,
            reason="reverse_path_ready",
        )

        steering = planner._entry_curve_steering(path)

        self.assertEqual(steering, -planner.config.reverse_entry_min_steering)

    def test_curve_reverse_keeps_maximum_right_when_local_path_changes_side(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        path_points_left = planner.update(
            geometry(
                heading=45.0,
                lateral=-0.60,
                remaining=700.0,
                reason="lidar_slot_box",
            ),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        path_points_right = planner.update(
            geometry(
                heading=30.0,
                lateral=0.60,
                remaining=680.0,
                reason="lidar_slot_box",
            ),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertIsNotNone(path_points_left.path)
        self.assertLess(path_points_left.path.curvature_per_px, 0.0)
        self.assertIsNotNone(path_points_right.path)
        self.assertGreater(path_points_right.path.curvature_per_px, 0.0)
        for plan in (path_points_left, path_points_right):
            self.assertEqual(plan.state, ParkingState.FOLLOW_ENTRY_CURVE)
            self.assertLess(plan.command.speed, 0)
            self.assertEqual(plan.command.steering, planner.config.max_steering)
            self.assertIn("following_entry_fixed_max_right", plan.reason)

    def test_curve_reverse_recomputes_magnitude_without_flipping_entry_side(self):
        # Curvature is regenerated every frame, but the armed entry-side sign
        # must not flip on a transient path change.
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                reverse_entry_continuous_steering=True,
                reverse_entry_release_heading_deg=1.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        left = planner.update(
            geometry(heading=45.0, lateral=-0.60, remaining=700.0, reason="lidar_slot_box"),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        right = planner.update(
            geometry(heading=30.0, lateral=0.60, remaining=680.0, reason="lidar_slot_box"),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(left.state, ParkingState.FOLLOW_ENTRY_CURVE)
        # Opposite-side paths → opposite curvature signs → opposite steering,
        # unlike fixed mode which pins both to the same constant max-right.
        self.assertLess(left.path.curvature_per_px, 0.0)
        self.assertGreater(right.path.curvature_per_px, 0.0)
        self.assertLess(left.command.steering, 0)
        self.assertLess(right.command.steering, 0)
        floor = planner.config.reverse_entry_min_steering
        self.assertGreaterEqual(abs(left.command.steering), floor)
        self.assertIn("following_entry_curve", left.reason)

    def test_single_car_left_virtual_slot_enters_with_maximum_right_steering(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=2,
                reverse_entry_continuous_steering=True,
                max_steering=150,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)
        planner._reverse_entry_virtual_left = True
        planner._reverse_entry_steering_direction = 1
        planner._reverse_entry_selection_mode = "single_car_left"

        plan = planner.update(
            geometry(
                heading=-35.0,
                lateral=0.5,
                remaining=700.0,
                reason="camera_virtual_slot",
                selection_mode="single_car_left",
            ),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        unwinding = planner.update(
            geometry(
                heading=-35.0,
                lateral=0.5,
                remaining=690.0,
                reason="camera_virtual_slot",
                selection_mode="single_car_left",
            ),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(plan.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertLess(plan.command.speed, 0)
        self.assertEqual(plan.command.steering, 150)
        self.assertIn("following_entry_virtual_left_max_right", plan.reason)
        self.assertLessEqual(
            abs(unwinding.command.steering),
            planner.config.reverse_entry_unwind_max_steering,
        )
        self.assertEqual(unwinding.command.steering, 0)
        self.assertIn("virtual_left_path_unwind", unwinding.reason)

        observed_line = ParkingLine(
            center_x=300.0,
            center_y=300.0,
            direction_x=0.0,
            direction_y=-1.0,
            length_px=200.0,
            residual_px=0.0,
            quality=1.0,
            point_count=100,
            mask_index=0,
        )
        real_three_line = replace(
            geometry(
                heading=-35.0,
                lateral=0.5,
                remaining=680.0,
                selection_mode="three_line_after_car",
                observed_lines=3,
            ),
            left=observed_line,
            right=replace(observed_line, mask_index=1),
            back=replace(observed_line, mask_index=2),
        )
        released = planner.update(real_three_line, lidar_gap(0.0, reached=True), 0.7)
        self.assertLess(released.command.steering, 0)

    def test_entry_path_loss_coasts_for_one_second(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                entry_path_loss_grace_s=1.0,
            )
        )
        self.arm_reverse(planner)
        planner.update(geometry(remaining=700.0), lidar_gap(0.0, reached=True), 0.5)
        lost = geometry(remaining=None)

        first = planner.update(lost, lidar_gap(0.0, reached=True), 0.6)
        second = planner.update(lost, lidar_gap(0.0, reached=True), 1.6)
        third = planner.update(lost, lidar_gap(0.0, reached=True), 1.61)

        self.assertLess(first.command.speed, 0)
        self.assertLess(second.command.speed, 0)
        self.assertEqual(abs(first.command.steering), planner.config.max_steering)
        self.assertEqual(third.command.speed, 0)
        self.assertIn("entry_curve_path_lost", third.reason)

    def test_entry_timeout_pauses_while_stopped_for_path_loss(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                entry_path_loss_grace_s=0.1,
                entry_curve_timeout_s=1.0,
            )
        )
        self.arm_reverse(planner)
        planner.update(geometry(remaining=700.0), lidar_gap(), 0.5)
        lost = geometry(remaining=None)
        planner.update(lost, lidar_gap(), 0.6)
        stopped = planner.update(lost, lidar_gap(), 0.71)
        still_stopped = planner.update(lost, lidar_gap(), 5.0)

        recovered = planner.update(
            geometry(remaining=680.0),
            lidar_gap(),
            5.1,
        )

        self.assertTrue(stopped.command.brake)
        self.assertTrue(still_stopped.command.brake)
        self.assertEqual(recovered.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertLess(recovered.command.speed, 0)

    def test_virtual_left_unwinds_and_coasts_across_selection_changes(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_heading_deg=12.0,
                reverse_entry_release_confirm_frames=2,
                entry_path_loss_grace_s=1.0,
                max_steering=150,
            )
        )
        self.arm_reverse(planner)
        planner._reverse_entry_virtual_left = True
        planner._reverse_entry_steering_direction = 1
        planner._reverse_entry_selection_mode = "single_car_left"

        release_pending = planner.update(
            geometry(
                heading=5.0,
                lateral=0.0,
                selection_mode="two_car_left_line",
                remaining=700.0,
            ),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        unwinding = planner.update(
            geometry(
                heading=5.0,
                lateral=0.0,
                selection_mode="two_car_left_line",
                remaining=690.0,
            ),
            lidar_gap(0.0, reached=True),
            0.6,
        )
        centered = planner.update(
            geometry(
                heading=3.0,
                lateral=0.0,
                selection_mode="line_only",
                remaining=680.0,
            ),
            lidar_gap(0.0, reached=True),
            0.7,
        )
        lost = geometry(remaining=None, selection_mode="line_only")
        coast_one = planner.update(lost, lidar_gap(0.0, reached=True), 0.8)
        coast_two = planner.update(lost, lidar_gap(0.0, reached=True), 1.8)
        stopped = planner.update(lost, lidar_gap(0.0, reached=True), 1.81)

        self.assertEqual(release_pending.command.steering, 150)
        self.assertIn("virtual_left_max_right", release_pending.reason)
        self.assertEqual(unwinding.command.steering, 0)
        self.assertIn("virtual_left_path_unwind", unwinding.reason)
        self.assertEqual(centered.state, ParkingState.FOLLOW_SLOT_CENTER)
        for plan in (coast_one, coast_two):
            self.assertLess(plan.command.speed, 0)
            self.assertEqual(plan.command.steering, centered.command.steering)
            self.assertIn("slot_center_path_coast", plan.reason)
        self.assertEqual(stopped.command.speed, 0)
        self.assertIn("slot_center_path_lost", stopped.reason)

    def test_curve_reverse_releases_only_after_stable_bev_centerline_alignment(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_heading_deg=12.0,
                reverse_entry_release_confirm_frames=3,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        first = planner.update(
            geometry(heading=12.0, lateral=0.2, remaining=650.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        second = planner.update(
            geometry(heading=10.0, lateral=0.2, remaining=640.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )
        aligned_once = planner.update(
            geometry(heading=8.0, lateral=0.1, remaining=630.0),
            lidar_gap(0.0, reached=True),
            0.7,
        )
        aligned_twice = planner.update(
            geometry(heading=5.0, lateral=0.1, remaining=620.0),
            lidar_gap(0.0, reached=True),
            0.8,
        )
        released = planner.update(
            geometry(heading=3.0, lateral=0.1, remaining=610.0),
            lidar_gap(0.0, reached=True),
            0.9,
        )

        for plan in (first, second, aligned_once, aligned_twice):
            self.assertEqual(plan.state, ParkingState.FOLLOW_ENTRY_CURVE)
        for plan in (first, second):
            self.assertGreaterEqual(
                abs(plan.command.steering),
                planner.config.reverse_entry_min_steering,
            )
        for plan in (aligned_once, aligned_twice):
            self.assertGreaterEqual(
                abs(plan.command.steering),
                planner.config.reverse_entry_min_steering,
            )
        self.assertEqual(released.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(released.command.speed, 0)
        self.assertEqual(
            released.reason,
            "following_slot_center:entry_aligned_released",
        )

    def test_fresh_aligned_obstacles_switch_to_trimmed_straight_reverse(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-20,
                reverse_aligned_speed=-40,
                aligned_confirm_frames=2,
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)
        planner.update(
            geometry(heading=5.0, lateral=0.1, remaining=500.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.5,
        )

        confirming = planner.update(
            geometry(heading=4.0, lateral=0.1, remaining=490.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.6,
        )
        straight = planner.update(
            geometry(heading=3.0, lateral=0.1, remaining=480.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.7,
        )

        self.assertEqual(confirming.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertEqual(straight.command.speed, -40)
        self.assertEqual(straight.command.steering, -20)
        self.assertEqual(
            straight.reason,
            "following_slot_center:aligned_straight",
        )

    def test_aligned_straight_reverse_parks_when_body_mid_is_inside(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-20,
                reverse_aligned_speed=-40,
                ultrasonic_inside_max_mm=600.0,
                ultrasonic_inside_confirm_frames=3,
                aligned_confirm_frames=1,
                aligned_reverse_min_s=0.4,
                aligned_reverse_max_s=2.5,
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)
        aligned_geometry = geometry(
            heading=3.0,
            lateral=0.1,
            remaining=480.0,
        )
        aligned_lidar = lidar_gap(
            0.0,
            reached=True,
            pair_observed=True,
        )
        planner.update(aligned_geometry, aligned_lidar, 0.5)
        planner.update(aligned_geometry, aligned_lidar, 0.6)

        first = planner.update(
            aligned_geometry,
            aligned_lidar,
            0.7,
            left_ultrasonic_mm=550.0,
            right_ultrasonic_mm=560.0,
        )
        second = planner.update(
            aligned_geometry,
            aligned_lidar,
            0.8,
            left_ultrasonic_mm=550.0,
            right_ultrasonic_mm=560.0,
        )
        parked = planner.update(
            aligned_geometry,
            aligned_lidar,
            1.0,
            left_ultrasonic_mm=550.0,
            right_ultrasonic_mm=560.0,
        )

        self.assertLess(first.command.speed, 0)
        self.assertLess(second.command.speed, 0)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(parked.reason, "body_mid_inside_and_aligned")
        self.assertTrue(parked.command.brake)

    def test_aligned_straight_reverse_has_bounded_fallback_duration(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-20,
                reverse_aligned_speed=-40,
                aligned_confirm_frames=1,
                aligned_reverse_min_s=0.4,
                aligned_reverse_max_s=2.5,
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)
        aligned_geometry = geometry(
            heading=3.0,
            lateral=0.1,
            remaining=480.0,
        )
        aligned_lidar = lidar_gap(
            0.0,
            reached=True,
            pair_observed=True,
        )
        planner.update(aligned_geometry, aligned_lidar, 0.5)
        straight = planner.update(aligned_geometry, aligned_lidar, 0.6)

        parked = planner.update(aligned_geometry, aligned_lidar, 3.1)

        self.assertEqual(straight.command.speed, -40)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(
            parked.reason,
            "aligned_straight_reverse_complete",
        )
        self.assertTrue(parked.command.brake)

    def test_curve_reverse_settles_maximum_right_before_moving(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.4,
                reverse_entry_continuous_steering=False,
                entry_curve_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        settling = planner.update(
            geometry(heading=45.0, lateral=-0.4, remaining=700.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        reversing = planner.update(
            geometry(heading=45.0, lateral=-0.4, remaining=690.0),
            lidar_gap(0.0, reached=True),
            0.9,
        )

        self.assertEqual(settling.command.speed, 0)
        self.assertEqual(settling.command.steering, planner.config.max_steering)
        self.assertEqual(settling.reason, "reverse_entry_bev_path:settling")
        self.assertLess(reversing.command.speed, 0)
        self.assertEqual(reversing.command.steering, planner.config.max_steering)

    def test_misaligned_entry_keeps_replanning_without_premature_correction(self):
        planner = self.make_correction_planner()
        self.arm_reverse(planner)
        off_center = geometry(
            heading=30.0,
            lateral=-0.45,
            remaining=700.0,
            reason="lidar_slot_box",
        )

        first_reverse = planner.update(off_center, lidar_gap(0.0, reached=True), 0.5)
        correction_start = planner.update(off_center, lidar_gap(0.0, reached=True), 0.6)
        correcting_forward = planner.update(off_center, lidar_gap(0.0, reached=True), 0.7)
        reverse_settle = planner.update(off_center, lidar_gap(0.0, reached=True), 1.2)
        correcting_reverse = planner.update(off_center, lidar_gap(0.0, reached=True), 1.3)
        aligned = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=600.0,
                reason="lidar_slot_box",
            ),
            lidar_gap(0.0, reached=True),
            1.4,
        )

        self.assertEqual(first_reverse.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(first_reverse.command.steering, planner.config.max_steering)
        for plan in (
            correction_start,
            correcting_forward,
            reverse_settle,
            correcting_reverse,
        ):
            self.assertEqual(plan.state, ParkingState.FOLLOW_ENTRY_CURVE)
            self.assertLess(plan.command.speed, 0)
            self.assertEqual(plan.command.steering, planner.config.max_steering)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(aligned.command.steering, 0)
        self.assertEqual(
            aligned.reason,
            "following_slot_center:entry_aligned_released",
        )

    def test_lidar_left_obstacle_uses_max_right_reverse_evasion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        avoiding = planner.update(
            geometry(),
            lidar_gap(unsafe=True, safety_x=-220.0),
            0.5,
        )
        resumed = planner.update(geometry(), lidar_gap(), 0.6)

        self.assertEqual(avoiding.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(avoiding.command.speed, -24)
        self.assertEqual(avoiding.command.steering, 150)
        self.assertIn("left_to_right", avoiding.reason)
        self.assertLess(resumed.command.speed, 0)

    def test_lidar_right_obstacle_uses_max_left_reverse_evasion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        avoiding = planner.update(
            geometry(),
            lidar_gap(unsafe=True, safety_x=220.0),
            0.5,
        )

        self.assertEqual(avoiding.command.speed, -24)
        self.assertEqual(avoiding.command.steering, -150)
        self.assertIn("right_to_left", avoiding.reason)

    def test_lidar_center_obstacle_uses_last_safe_low_speed_steering(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        held = planner.update(
            geometry(),
            lidar_gap(unsafe=True, safety_x=0.0),
            0.5,
        )
        resumed = planner.update(geometry(), lidar_gap(), 0.6)

        self.assertEqual(held.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(held.command.speed, -24)
        self.assertFalse(held.command.brake)
        self.assertIn("center_last_safe", held.reason)
        self.assertLess(resumed.command.speed, 0)

    def test_missing_lidar_never_allows_reverse_motion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(
            geometry(),
            LidarParkingObservation(reason="stale_scan"),
            0.5,
        )

        self.assertEqual(stopped.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(stopped.reason, "lidar_unavailable:stale_scan")

    def test_lidar_recovery_requires_three_distinct_valid_scans(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        planner.update(
            geometry(),
            LidarParkingObservation(reason="stale_scan"),
            0.5,
        )

        first = planner.update(
            geometry(), replace(lidar_gap(), timestamp=2.0), 0.6
        )
        duplicate = planner.update(
            geometry(), replace(lidar_gap(), timestamp=2.0), 0.7
        )
        second = planner.update(
            geometry(), replace(lidar_gap(), timestamp=3.0), 0.8
        )
        resumed = planner.update(
            geometry(), replace(lidar_gap(), timestamp=4.0), 0.9
        )

        self.assertEqual(first.reason, "lidar_resume_check:1/3")
        self.assertEqual(duplicate.reason, "lidar_resume_check:1/3")
        self.assertEqual(second.reason, "lidar_resume_check:2/3")
        self.assertLess(resumed.command.speed, 0)

    def test_legacy_rear_axle_positioning_corrects_in_both_directions(self):
        planner = self.make_planner()
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)

        forward = planner.update(geometry(), lidar_gap(entry_error=-150.0), 0.1)

        other = self.make_planner()
        other.start(0.0)
        other.update(geometry(), lidar_gap(), 0.0)
        reverse = other.update(geometry(), lidar_gap(entry_error=150.0), 0.1)

        self.assertGreater(forward.command.speed, 0)
        self.assertLess(reverse.command.speed, 0)

    def test_prealign_moves_forward_with_maximum_left_steering(self):
        planner = self.make_prealign_planner()
        entered = self.enter_prealign(planner)

        moving = planner.update(
            geometry(),
            prealign_lidar(slot_heading_deg=70.0, entry_bearing_deg=65.0),
            0.2,
        )

        self.assertEqual(entered.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.command.speed, 14)
        self.assertEqual(moving.command.steering, -150)

    def test_reverse_safety_box_does_not_stop_forward_prealign(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        moving = planner.update(
            geometry(observed_cars=1),
            prealign_lidar(slot_heading_deg=70.0, entry_bearing_deg=65.0),
            0.2,
        )
        still_moving = planner.update(
            geometry(observed_cars=1),
            prealign_lidar(70.0, 65.0, unsafe=True),
            0.3,
        )

        self.assertEqual(moving.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(still_moving.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertNotEqual(still_moving.state, ParkingState.EMERGENCY_STOP)

    def test_first_car_slows_then_starts_left_turn_before_gap_confirmation(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_approach_speed=10,
                first_car_straight_s=1.0,
                prealign_speed=35,
                prealign_steering=-150,
                prealign_steer_settle_s=0.4,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        creeping = planner.update(geometry(), first_car_lidar(), 0.1)
        settling = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-5.0),
            0.2,
        )
        delayed = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-5.0),
            1.0,
        )
        turning = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            1.2,
        )

        self.assertEqual(creeping.state, ParkingState.TRACK_GAP)
        self.assertEqual(creeping.command.speed, 10)
        self.assertEqual(settling.state, ParkingState.TRACK_GAP)
        self.assertEqual(settling.command.speed, 10)
        self.assertEqual(settling.command.steering, 0)
        self.assertEqual(settling.reason, "first_car_straight_delay")
        self.assertEqual(delayed.state, ParkingState.TRACK_GAP)
        self.assertEqual(delayed.command.speed, 10)
        self.assertEqual(delayed.command.steering, 0)
        self.assertEqual(turning.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turning.command.speed, 0)
        self.assertEqual(turning.command.steering, -150)
        self.assertEqual(turning.reason, "first_car_straight_elapsed:settling_max_left")

    def test_prealign_direct_reverse_requires_stable_heading_and_bearing(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        aligned_lidar = prealign_lidar(
            slot_heading_deg=5.0,
            entry_bearing_deg=8.0,
            distance_mm=900.0,
        )

        confirming = planner.update(geometry(observed_cars=1), aligned_lidar, 0.2)
        ready = planner.update(geometry(observed_cars=1), aligned_lidar, 0.3)

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(ready.reason, "prealign_sensor_crosscheck_ready:reverse_path_armed")
        self.assertTrue(ready.command.brake)

    def test_prealign_uses_car_selected_camera_slot_when_lidar_pose_is_bad(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        camera_slot = geometry(selection_mode="single_car_left", observed_cars=1)
        bad_lidar_pose = prealign_lidar(
            slot_heading_deg=90.0,
            entry_bearing_deg=90.0,
        )

        confirming = planner.update(camera_slot, bad_lidar_pose, 0.2)
        ready = planner.update(camera_slot, bad_lidar_pose, 0.3)

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(ready.reason, "prealign_sensor_crosscheck_ready:reverse_path_armed")
        self.assertTrue(ready.command.brake)

    def test_camera_slot_does_not_leave_search_before_lidar_car(self):
        planner = self.make_prealign_planner()
        planner.start(0.0)

        plan = planner.update(
            geometry(selection_mode="single_car_left"),
            LidarParkingObservation(valid=True, observed_points=20),
            0.1,
        )

        self.assertEqual(plan.state, ParkingState.SEARCH_CARS)
        self.assertEqual(plan.reason, "searching_for_parked_cars")
        self.assertEqual(plan.command.steering, 0)

    def test_camera_slot_cannot_finish_prealign_without_two_lidar_cars(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        camera_slot = geometry(selection_mode="single_car_right", observed_cars=1)
        lidar_without_gap = LidarParkingObservation(valid=True, observed_points=20)

        first = planner.update(camera_slot, lidar_without_gap, 0.2)
        second = planner.update(camera_slot, lidar_without_gap, 0.3)

        self.assertEqual(first.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(second.state, ParkingState.PREALIGN_LEFT)
        self.assertIn("prealign_waiting_for_lidar_cars", second.reason)

    def test_prealign_accepts_fresh_two_car_counts_without_a_gap_box(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        at_limit = planner.update(
            geometry(observed_cars=1),
            prealign_lidar(
                slot_heading_deg=20.0,
                first_car_bearing_deg=20.0,
                fresh_pair=False,
            ),
            0.2,
        )
        above_first = planner.update(
            geometry(observed_cars=1),
            prealign_lidar(
                slot_heading_deg=20.0,
                first_car_bearing_deg=20.0,
                fresh_pair=False,
            ),
            0.3,
        )
        above_second = planner.update(
            geometry(observed_cars=1),
            prealign_lidar(
                slot_heading_deg=20.0,
                first_car_bearing_deg=20.0,
                fresh_pair=False,
            ),
            0.4,
        )

        self.assertEqual(at_limit.state, ParkingState.PREALIGN_LEFT)
        self.assertIn("prealign_bev_path_confirming:1/2", at_limit.reason)
        self.assertEqual(above_first.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(above_second.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_prealign_requires_three_fresh_two_car_scans_and_resets_on_one(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.0,
                prealign_confirm_frames=1,
                prealign_lidar_confirm_scans=3,
            )
        )
        self.enter_prealign(planner)
        live_geometry = geometry(observed_cars=1)

        first = planner.update(
            live_geometry, replace(prealign_lidar(), timestamp=2.0), 0.2
        )
        second = planner.update(
            live_geometry, replace(prealign_lidar(), timestamp=3.0), 0.3
        )
        dropped = planner.update(
            live_geometry,
            replace(prealign_lidar(), timestamp=4.0, car_count=1),
            0.4,
        )
        planner.update(
            live_geometry, replace(prealign_lidar(), timestamp=5.0), 0.5
        )
        planner.update(
            live_geometry, replace(prealign_lidar(), timestamp=6.0), 0.6
        )
        armed = planner.update(
            live_geometry, replace(prealign_lidar(), timestamp=7.0), 0.7
        )

        self.assertIn("scans=1/3", first.reason)
        self.assertIn("scans=2/3", second.reason)
        self.assertIn("scans=0/3", dropped.reason)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(
            armed.reason,
            "prealign_sensor_crosscheck_ready:reverse_path_armed",
        )

    def test_prealign_requires_recent_yolo_car(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        plan = planner.update(
            geometry(observed_cars=0),
            prealign_lidar(slot_heading_deg=20.0),
            0.2,
        )

        self.assertEqual(plan.state, ParkingState.PREALIGN_LEFT)
        self.assertIn("prealign_waiting_for_yolo_cars", plan.reason)

    def test_prealign_holds_max_left_for_five_seconds_before_sensor_exit(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.4,
                prealign_min_drive_s=5.0,
                prealign_confirm_frames=1,
            )
        )
        self.enter_prealign(planner)
        ready_geometry = geometry(observed_lines=1, observed_cars=1)
        ready_lidar = prealign_lidar(first_car_bearing_deg=110.0)

        held = planner.update(ready_geometry, ready_lidar, 5.49)
        stopped = planner.update(ready_geometry, ready_lidar, 5.5)

        self.assertEqual(held.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(held.command.speed, planner.config.prealign_speed)
        self.assertEqual(held.command.steering, -150)
        self.assertIn("prealign_min_drive", held.reason)
        self.assertEqual(stopped.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(stopped.command.speed, 0)

    def test_prealign_requires_current_bev_path_after_lidar_gate_latches(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.0,
                prealign_confirm_frames=1,
                prealign_path_hold_frames=3,
            )
        )
        self.enter_prealign(planner)

        planner.update(
            ParkingGeometry(
                observed_car_count=1,
                reason="camera_path_unavailable",
            ),
            prealign_lidar(first_car_bearing_deg=110.0),
            0.2,
        )
        ready = planner.update(
            ParkingGeometry(reason="camera_frame_dropped"),
            prealign_lidar(first_car_bearing_deg=121.0),
            0.3,
        )

        self.assertEqual(ready.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.reason, "prealign_waiting_for_current_bev_path")

    def test_prealign_resets_lidar_gate_when_pair_is_lost(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.0,
                prealign_min_drive_s=5.0,
                prealign_confirm_frames=1,
            )
        )
        self.enter_prealign(planner)

        latched = planner.update(
            geometry(observed_lines=1, observed_cars=1),
            prealign_lidar(first_car_bearing_deg=100.0),
            1.0,
        )
        ready_after_pair_is_lost = planner.update(
            geometry(observed_lines=1, observed_cars=1),
            replace(
                prealign_lidar(first_car_bearing_deg=80.0, fresh_pair=False),
                timestamp=2.0,
                car_count=1,
            ),
            5.1,
        )

        self.assertIn("prealign_min_drive", latched.reason)
        self.assertEqual(ready_after_pair_is_lost.state, ParkingState.PREALIGN_LEFT)
        self.assertIn("prealign_waiting_for_lidar_cars", ready_after_pair_is_lost.reason)

    def test_prealign_requires_recent_yolo_car_cross_check(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.0,
                prealign_confirm_frames=1,
                prealign_yolo_min_cars=1,
            )
        )
        self.enter_prealign(planner)

        waiting = planner.update(
            geometry(observed_lines=1, observed_cars=0),
            prealign_lidar(first_car_bearing_deg=105.0),
            0.2,
        )
        ready = planner.update(
            geometry(observed_lines=1, observed_cars=1),
            prealign_lidar(first_car_bearing_deg=105.0),
            0.3,
        )

        self.assertEqual(waiting.reason, "prealign_waiting_for_yolo_cars:0/1")
        self.assertEqual(ready.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_virtual_left_entry_mode_stays_latched_when_camera_mode_drops(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_steer_settle_s=0.0,
                prealign_min_drive_s=1.0,
                prealign_confirm_frames=1,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
            )
        )
        self.enter_prealign(planner)
        stabilizer = ParkingGeometryDepthStabilizer(
            max_jump_px=160.0,
            reconfirm_frames=5,
        )
        virtual_left = stabilizer.update(
            geometry(
                selection_mode="single_car_left",
                observed_lines=1,
                observed_cars=1,
            )
        )
        latched = planner.update(
            virtual_left,
            prealign_lidar(first_car_bearing_deg=105.0),
            0.2,
        )
        dropped = stabilizer.update(
            geometry(
                remaining=318.0,
                selection_mode="line_only",
                observed_lines=0,
            )
        )
        armed = planner.update(
            virtual_left,
            replace(prealign_lidar(first_car_bearing_deg=105.0), timestamp=2.0),
            1.2,
        )
        planner.update(virtual_left, replace(lidar_gap(), timestamp=3.0), 1.3)
        reversing = planner.update(dropped, replace(lidar_gap(), timestamp=4.0), 1.4)

        self.assertIn("prealign_min_drive", latched.reason)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(reversing.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(reversing.command.steering, 150)
        self.assertIn("virtual_left_max_right", reversing.reason)

    def test_prealign_keeps_driving_until_current_bev_path_is_feasible(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        first = planner.update(
            ParkingGeometry(
                observed_line_count=1,
                observed_car_count=1,
                reason="lidar_slot_box_unavailable",
            ),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.2,
        )
        stopped = planner.update(
            ParkingGeometry(
                observed_line_count=1,
                observed_car_count=1,
                reason="lidar_slot_box_unavailable",
            ),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.3,
        )

        self.assertEqual(first.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(stopped.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(stopped.reason, "prealign_waiting_for_current_bev_path")
        self.assertGreater(stopped.command.speed, 0)

    def test_verify_slot_waits_stationary_for_bev_after_left_turn_stops(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        planner.update(
            ParkingGeometry(
                observed_line_count=1,
                observed_car_count=1,
                reason="lidar_slot_box_unavailable",
            ),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.2,
        )
        planner.update(
            geometry(observed_lines=1, observed_cars=1),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.3,
        )
        planner.update(
            geometry(observed_lines=1, observed_cars=1),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.35,
        )
        waiting = planner.update(
            ParkingGeometry(reason="lidar_slot_box_unavailable"),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.4,
        )

        self.assertEqual(waiting.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(waiting.reason, "reverse_start_waiting_for_live_path")
        self.assertEqual(waiting.command.speed, 0)

    def test_side_ultrasonic_emergency_stop_is_latched(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(
            geometry(),
            lidar_gap(),
            0.5,
            left_ultrasonic_mm=95.0,
            right_ultrasonic_mm=500.0,
        )
        still_stopped = planner.update(geometry(), lidar_gap(), 0.6)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(still_stopped.state, ParkingState.EMERGENCY_STOP)

    def test_side_ultrasonic_stop_reacquires_path_before_resuming(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        planner.update(
            geometry(),
            lidar_gap(),
            0.5,
            left_ultrasonic_mm=95.0,
            right_ultrasonic_mm=500.0,
        )

        first = planner.update(
            geometry(), lidar_gap(), 0.6,
            left_ultrasonic_mm=500.0, right_ultrasonic_mm=500.0,
        )
        second = planner.update(
            geometry(), lidar_gap(), 0.7,
            left_ultrasonic_mm=500.0, right_ultrasonic_mm=500.0,
        )
        reacquired = planner.update(
            geometry(), lidar_gap(), 0.8,
            left_ultrasonic_mm=500.0, right_ultrasonic_mm=500.0,
        )
        resumed = planner.update(
            geometry(), lidar_gap(), 0.9,
            left_ultrasonic_mm=500.0, right_ultrasonic_mm=500.0,
        )

        self.assertEqual(first.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(second.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(reacquired.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertTrue(reacquired.command.brake)
        self.assertIn("path_reacquired", reacquired.reason)
        self.assertLess(resumed.command.speed, 0)

    def test_side_ultrasonic_clear_uses_last_path_within_grace(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        planner.update(
            geometry(), lidar_gap(), 0.5,
            left_ultrasonic_mm=95.0, right_ultrasonic_mm=500.0,
        )

        for now in (0.6, 0.7, 0.8):
            waiting = planner.update(
                ParkingGeometry(), lidar_gap(), now,
                left_ultrasonic_mm=500.0, right_ultrasonic_mm=500.0,
            )

        self.assertEqual(waiting.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(
            waiting.reason,
            "side_ultrasonic_cleared:last_path_recovery",
        )

    def test_car_only_path_uses_seventy_percent_reverse_speed(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_continuous_steering=False,
                reverse_entry_speed=-60,
                car_only_speed_ratio=0.70,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        reversing = planner.update(
            geometry(
                heading=20.0,
                lateral=0.4,
                remaining=700.0,
                selection_mode="car_only_left",
                observed_lines=0,
                observed_cars=1,
            ),
            lidar_gap(),
            0.5,
        )

        self.assertEqual(reversing.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(reversing.command.speed, -42)

    def test_side_ultrasonic_is_ignored_during_forward_search(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(start_forward_s=0.0, search_timeout_s=100.0)
        )
        planner.start(0.0)

        moving = planner.update(
            geometry(),
            LidarParkingObservation(
                timestamp=1.0,
                valid=True,
                observed_points=10,
                reason="searching_for_parked_cars",
            ),
            0.1,
            left_ultrasonic_mm=95.0,
            right_ultrasonic_mm=95.0,
        )

        self.assertEqual(moving.state, ParkingState.SEARCH_CARS)
        self.assertGreater(moving.command.speed, 0)
        self.assertFalse(moving.command.brake)

    def test_front_ultrasonic_is_ignored_during_forward_search(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(start_forward_s=0.0, search_timeout_s=100.0)
        )
        planner.start(0.0)

        moving = planner.update(
            geometry(),
            LidarParkingObservation(
                timestamp=1.0,
                valid=True,
                observed_points=10,
                reason="searching_for_parked_cars",
            ),
            0.1,
            front_left_ultrasonic_mm=95.0,
            front_right_ultrasonic_mm=500.0,
        )

        self.assertEqual(moving.state, ParkingState.SEARCH_CARS)
        self.assertGreater(moving.command.speed, 0)
        self.assertFalse(moving.command.brake)

    def test_front_ultrasonic_does_not_block_reverse_motion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        reversing = planner.update(
            geometry(heading=20.0, lateral=0.4, remaining=700.0),
            lidar_gap(0.0, reached=True),
            0.5,
            front_left_ultrasonic_mm=95.0,
            front_right_ultrasonic_mm=95.0,
        )

        self.assertEqual(reversing.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertLess(reversing.command.speed, 0)

    def test_side_ultrasonic_p_control_is_bounded(self):
        planner = self.make_planner()

        self.assertEqual(planner._ultrasonic_correction(500.0, 500.0), 0)
        self.assertEqual(planner._ultrasonic_correction(400.0, 800.0), 35)
        self.assertEqual(planner._ultrasonic_correction(800.0, 400.0), -35)
        self.assertEqual(planner._ultrasonic_correction(None, 400.0), 0)

    def test_lidar_y0_one_side_clear_holds_then_exits_straight_for_four_seconds(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        left_car = CarCluster(
            point_count=8,
            x_min_mm=-900.0,
            x_max_mm=-500.0,
            y_back_min_mm=-100.0,
            y_back_max_mm=100.0,
        )
        right_car = CarCluster(
            point_count=8,
            x_min_mm=500.0,
            x_max_mm=900.0,
            y_back_min_mm=-100.0,
            y_back_max_mm=100.0,
        )
        three_line = geometry(
            heading=20.0,
            lateral=0.4,
            remaining=700.0,
            selection_mode="three_line_after_car",
            observed_lines=3,
        )
        occupied = replace(
            lidar_gap(),
            timestamp=2.0,
            side_car_clusters=(left_car, right_car),
        )
        one_side_clear = replace(
            lidar_gap(),
            timestamp=3.0,
            side_car_clusters=(right_car,),
        )
        planner._lidar_y0_terminal_armed = True

        not_armed = planner.update(
            three_line,
            replace(one_side_clear, timestamp=1.0),
            0.4,
        )
        live = planner.update(three_line, occupied, 0.5)
        yolo_lost = planner.update(ParkingGeometry(), replace(occupied, timestamp=2.5), 0.6)
        clear_once = planner.update(ParkingGeometry(), one_side_clear, 0.7)
        duplicate_scan = planner.update(ParkingGeometry(), one_side_clear, 0.8)
        clear_twice = planner.update(
            ParkingGeometry(),
            replace(one_side_clear, timestamp=4.0),
            0.9,
        )
        parked = planner.update(
            ParkingGeometry(),
            replace(one_side_clear, timestamp=5.0),
            1.0,
        )
        holding = planner.update(ParkingGeometry(), one_side_clear, 4.49)
        exiting = planner.update(ParkingGeometry(), one_side_clear, 4.51)
        exited = planner.update(ParkingGeometry(), one_side_clear, 8.52)

        self.assertLess(not_armed.command.speed, 0)
        self.assertLess(live.command.speed, 0)
        self.assertLess(yolo_lost.command.speed, 0)
        self.assertLess(clear_once.command.speed, 0)
        self.assertLess(duplicate_scan.command.speed, 0)
        self.assertLess(clear_twice.command.speed, 0)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(parked.reason, "lidar_y0_one_side_cleared")
        self.assertTrue(parked.command.brake)
        self.assertEqual(holding.state, ParkingState.PARKED)
        self.assertTrue(holding.command.brake)
        self.assertEqual(exiting.state, ParkingState.EXIT_STRAIGHT)
        self.assertGreater(exiting.command.speed, 0)
        self.assertEqual(exiting.command.steering, 0)
        self.assertEqual(exited.state, ParkingState.EXIT_DONE)
        self.assertTrue(exited.command.brake)


if __name__ == "__main__":
    unittest.main()
