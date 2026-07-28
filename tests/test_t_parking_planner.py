import unittest
import math
from dataclasses import replace

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.reverse_parking_path import (
    ReversePath,
    ReversePathConfig,
    ReversePathStatus,
)
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
    completion_candidate=False,
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
        vehicle_footprint_slot_local_mm=(
            (-300.0, 100.0),
            (300.0, 100.0),
            (300.0, 1100.0),
            (-300.0, 1100.0),
        ),
        park_completion_candidate=completion_candidate,
        park_completion_reason=(
            "footprint_inside_fixed_slot"
            if completion_candidate
            else "footprint_crosses_slot_entrance"
        ),
        confidence=0.9,
        reason=reason,
    )


def lidar_gap(
    entry_error=200.0,
    reached=False,
    unsafe=False,
    pair_observed=False,
    is_new_scan=True,
    scan_timestamp=1.0,
):
    return LidarParkingObservation(
        timestamp=scan_timestamp,
        is_new_scan=is_new_scan,
        valid=True,
        unsafe=unsafe,
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
    is_new_scan=True,
):
    slot_angle = math.radians(slot_heading_deg)
    bearing = math.radians(entry_bearing_deg)
    rear_axle_y = -300.0
    return LidarParkingObservation(
        timestamp=1.0,
        is_new_scan=is_new_scan,
        valid=True,
        observed_points=20,
        car_count=2,
        gap_found=True,
        gap_confirmed=True,
        gap_width_mm=1375.0,
        gap_center_x_right_mm=math.sin(bearing) * distance_mm,
        gap_center_y_back_mm=rear_axle_y + math.cos(bearing) * distance_mm,
        entry_target_y_back_mm=rear_axle_y,
        slot_depth_x_right=math.sin(slot_angle),
        slot_depth_y_back=math.cos(slot_angle),
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
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                reverse_entry_steer_settle_s=0.0,
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
                provisional_prealign_speed=8,
                prealign_speed=14,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                prealign_timeout_s=2.0,
                search_timeout_s=100.0,
                position_timeout_s=100.0,
            )
        )

    def make_correction_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
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
                reverse_entry_steer_settle_s=0.0,
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
        candidate = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=0.0,
                completion_candidate=True,
            ),
            lidar_gap(0.0, reached=True, scan_timestamp=1.0),
            0.6,
        )
        parked = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=0.0,
                completion_candidate=True,
            ),
            lidar_gap(0.0, reached=True, scan_timestamp=2.0),
            0.7,
        )
        hold = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=0.0,
                completion_candidate=True,
            ),
            lidar_gap(0.0, reached=True, scan_timestamp=3.0),
            3.5,
        )
        exit_right = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.7,
        )
        exit_straight = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            5.4,
        )
        exit_done = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            8.5,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(positioned.state, ParkingState.POSITION_REAR_AXLE)
        self.assertEqual(verifying.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(path_plan.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            path_plan.reason,
            "reverse_path_armed:curve:set_candidate_steer",
        )
        self.assertEqual(path_plan.command.speed, 0)
        self.assertEqual(
            path_plan.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIsNotNone(armed.path)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(
            aligned.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(
            aligned.reason,
            "following_entry_curve:candidate_steer",
        )
        self.assertEqual(candidate.state, ParkingState.PARK_CONFIRM)
        self.assertEqual(candidate.command.speed, 0)
        self.assertTrue(candidate.command.brake)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(
            parked.reason,
            "park_confirmed_on_first_stopped_scan",
        )
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

    def test_park_confirm_ignores_candidate_scan_duplicate(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        complete = geometry(
            heading=0.0,
            lateral=0.0,
            remaining=20.0,
            completion_candidate=True,
        )

        candidate = planner.update(
            complete,
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=10.0,
            ),
            0.5,
        )
        duplicate = planner.update(
            complete,
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=10.0,
            ),
            0.6,
        )
        confirmed = planner.update(
            complete,
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=11.0,
            ),
            0.7,
        )

        self.assertEqual(candidate.state, ParkingState.PARK_CONFIRM)
        self.assertTrue(candidate.command.brake)
        self.assertEqual(duplicate.state, ParkingState.PARK_CONFIRM)
        self.assertEqual(
            duplicate.reason,
            "park_confirm_waiting_for_first_stopped_scan",
        )
        self.assertEqual(confirmed.state, ParkingState.PARKED)

    def test_failed_first_park_confirm_scan_reacquires_before_resume(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        candidate = geometry(
            heading=0.0,
            lateral=0.0,
            remaining=20.0,
            completion_candidate=True,
        )
        planner.update(
            candidate,
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=10.0,
            ),
            0.5,
        )

        lost = planner.update(
            ParkingGeometry(reason="slot_map_depth_unobservable"),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=11.0,
            ),
            0.6,
        )
        reacquired = planner.update(
            geometry(
                heading=2.0,
                lateral=0.05,
                remaining=60.0,
                completion_candidate=False,
            ),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=12.0,
            ),
            0.7,
        )

        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertTrue(lost.command.brake)
        self.assertEqual(
            lost.reason,
            "park_confirm_slot_unavailable:reacquire",
        )
        self.assertEqual(
            reacquired.state,
            ParkingState.VERIFY_SLOT_BOX,
        )
        self.assertTrue(reacquired.command.brake)
        self.assertEqual(
            reacquired.reason,
            "slot_reacquired:resume_pending:verify_slot_box",
        )

    def test_verify_arms_first_unique_scan_without_plan_state(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)

        duplicate = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True, is_new_scan=False),
            0.3,
        )
        armed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True, is_new_scan=True),
            0.4,
        )

        self.assertEqual(duplicate.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            duplicate.reason,
            "waiting_for_unique_lidar_arm_scan",
        )
        self.assertEqual(armed.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            armed.reason,
            "reverse_path_armed:curve:set_candidate_steer",
        )
        self.assertIsNotNone(armed.path)
        self.assertEqual(armed.command.speed, 0)
        self.assertEqual(armed.command.steering, planner.config.max_steering)

    def test_failed_pre_reverse_check_resumes_forward_prealign(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        slot = geometry()
        lidar = prealign_lidar(
            slot_heading_deg=70.0,
            entry_bearing_deg=65.0,
            distance_mm=1500.0,
        )

        verifying = planner.update(slot, lidar, 0.2)
        unarmable = replace(slot, stop_target_y_px=700.0)
        resumed = planner.update(unarmable, lidar, 0.3)
        moving = planner.update(unarmable, lidar, 0.4)

        self.assertEqual(verifying.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(resumed.state, ParkingState.PREALIGN_LEFT)
        self.assertTrue(resumed.command.brake)
        self.assertIn("resume_prealign_left", resumed.reason)
        self.assertEqual(moving.state, ParkingState.PREALIGN_LEFT)
        self.assertGreater(moving.command.speed, 0)
        self.assertEqual(
            moving.command.steering,
            planner.config.prealign_steering,
        )

    def test_verify_reacquires_after_failed_unique_scan_then_arms(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        missing_geometry = ParkingGeometry(reason="lidar_slot_box_unavailable")
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)

        lost = planner.update(
            missing_geometry,
            lidar_gap(0.0, reached=True),
            0.3,
        )
        duplicate = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True, is_new_scan=False),
            0.4,
        )
        armed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        verified = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertEqual(lost.reason, "slot_arm_geometry_unavailable")
        self.assertEqual(
            duplicate.reason,
            "reacquire_waiting_for_unique_lidar_scan",
        )
        self.assertEqual(armed.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            armed.reason,
            "slot_reacquired:resume_pending:verify_slot_box",
        )
        self.assertEqual(verified.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            verified.reason,
            "reverse_path_armed:curve:set_candidate_steer",
        )

    def test_set_reverse_steer_without_arm_returns_to_verify_stopped(self):
        planner = self.make_planner()
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.state = ParkingState.SET_REVERSE_STEER
        planner._state_started_at = 0.0

        guarded = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.5,
        )

        self.assertEqual(guarded.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertTrue(guarded.command.brake)
        self.assertEqual(guarded.command.speed, 0)
        self.assertEqual(guarded.command.steering, 0)
        self.assertEqual(
            guarded.reason,
            "reverse_entry_not_armed:return_verify",
        )

    def test_pose_loss_while_setting_reverse_steer_requires_new_verify_arm(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        missing_geometry = ParkingGeometry(
            reason="lidar_slot_box_unavailable"
        )
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        armed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.3,
        )

        lost = planner.update(
            missing_geometry,
            lidar_gap(0.0, reached=True),
            0.4,
        )
        reacquired = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        rearmed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(armed.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertTrue(lost.command.brake)
        self.assertEqual(reacquired.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            reacquired.reason,
            "slot_reacquired:resume_pending:verify_slot_box",
        )
        self.assertEqual(rearmed.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            rearmed.reason,
            "reverse_path_armed:curve:set_candidate_steer",
        )

    def test_pose_loss_during_reverse_requires_new_verify_arm(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        missing_geometry = ParkingGeometry(
            reason="slot_map_depth_unobservable"
        )

        lost = planner.update(
            missing_geometry,
            lidar_gap(0.0, reached=True),
            0.5,
        )
        reacquired = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.6,
        )
        rearmed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.7,
        )

        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertTrue(lost.command.brake)
        self.assertEqual(
            lost.reason,
            "slot_pose_unavailable_during_reverse",
        )
        self.assertEqual(reacquired.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            reacquired.reason,
            "slot_reacquired:resume_pending:verify_slot_box",
        )
        self.assertEqual(rearmed.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            rearmed.reason,
            "reverse_path_armed:curve:set_candidate_steer",
        )

    def test_reacquire_timeout_after_reverse_uses_timed_finish(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                reverse_entry_steer_settle_s=0.0,
                slot_reacquire_timeout_s=0.2,
                recovery_finish_reverse_speed=-40,
                recovery_finish_curve_s=0.1,
                recovery_finish_total_s=0.4,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)
        lost_geometry = ParkingGeometry(
            reason="slot_map_depth_unobservable"
        )

        lost = planner.update(
            lost_geometry,
            lidar_gap(0.0, reached=True),
            0.5,
        )
        fallback = planner.update(
            lost_geometry,
            lidar_gap(0.0, reached=True),
            0.8,
        )
        still_reversing = planner.update(
            lost_geometry,
            lidar_gap(0.0, reached=True),
            1.3,
        )
        aligned = planner.update(
            geometry(heading=5.0),
            lidar_gap(0.0, reached=True),
            1.35,
        )
        candidate = planner.update(
            geometry(completion_candidate=True),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=2.0,
            ),
            1.4,
        )
        parked = planner.update(
            geometry(completion_candidate=True),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=3.0,
            ),
            1.5,
        )

        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertEqual(
            fallback.state,
            ParkingState.FINISH_REVERSE_TIMED,
        )
        self.assertEqual(fallback.command.speed, -40)
        self.assertEqual(
            fallback.command.steering,
            planner.config.max_steering,
        )
        self.assertFalse(fallback.command.brake)
        self.assertEqual(
            still_reversing.state,
            ParkingState.FINISH_REVERSE_TIMED,
        )
        self.assertEqual(still_reversing.command.speed, -40)
        self.assertEqual(
            still_reversing.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(
            still_reversing.reason,
            "timed_finish_waiting_for_slot_completion",
        )
        self.assertEqual(aligned.command.speed, -40)
        self.assertEqual(
            aligned.command.steering,
            planner.config.straight_steering_trim,
        )
        self.assertEqual(candidate.state, ParkingState.PARK_CONFIRM)
        self.assertTrue(candidate.command.brake)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(
            parked.reason,
            "park_confirmed_on_first_stopped_scan",
        )

    def test_collision_during_curve_returns_to_forward_realign(self):
        planner = self.make_prealign_planner()
        planner.start(0.0)
        planner.state = ParkingState.FOLLOW_ENTRY_CURVE
        planner._state_started_at = 0.0
        planner._preverify_state = ParkingState.PROVISIONAL_PREALIGN
        planner._reverse_entry_mode = "lidar_box_curve"
        planner._reverse_started = True
        planner._reverse_path_armed = True
        planner._armed_entry_steering = planner.config.max_steering
        planner._armed_reverse_path = ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            entry_steering_ratio=1.0,
            reason="full_arc_straight_ready",
        )
        collision = ReversePath(
            found=False,
            status=ReversePathStatus.COLLISION_RISK,
            entry_steering_ratio=1.0,
            reason="insufficient_side_clearance_for_reverse_path",
        )
        planner.path_generator.generate = lambda _geometry: collision

        stopped = planner.update(
            geometry(heading=40.0, remaining=650.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        moving_forward = planner.update(
            geometry(heading=40.0, remaining=650.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(
            stopped.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertTrue(stopped.command.brake)
        self.assertIn("forward_realign_settle", stopped.reason)
        self.assertFalse(planner._reverse_started)
        self.assertEqual(
            moving_forward.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertGreater(moving_forward.command.speed, 0)
        self.assertEqual(
            moving_forward.command.steering,
            planner.config.prealign_steering,
        )
        self.assertNotEqual(
            moving_forward.state,
            ParkingState.FINISH_REVERSE_TIMED,
        )

    def test_reacquire_timeout_before_reverse_resumes_forward_alignment(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        planner.config = replace(
            planner.config,
            slot_reacquire_timeout_s=0.2,
        )
        planner._enter(ParkingState.VERIFY_SLOT_BOX, 0.2)

        lost = planner.update(
            ParkingGeometry(reason="slot_map_depth_unobservable"),
            lidar_gap(0.0, reached=True),
            0.3,
        )
        resumed = planner.update(
            ParkingGeometry(reason="slot_map_depth_unobservable"),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(lost.state, ParkingState.REACQUIRE_SLOT)
        self.assertEqual(
            resumed.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertGreater(resumed.command.speed, 0)
        self.assertFalse(resumed.command.brake)
        self.assertEqual(
            resumed.reason,
            "slot_reacquire_timeout:resume_provisional_prealign",
        )

    def test_soft_deadline_finishes_an_already_armed_maneuver_before_four_minutes(
        self,
    ):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                reverse_entry_steer_settle_s=0.0,
                mission_soft_deadline_s=225.0,
                recovery_finish_total_s=1.5,
                park_hold_s=3.0,
                exit_turn_s=1.6,
                exit_straight_s=0.0,
                search_timeout_s=300.0,
                gap_tracking_timeout_s=300.0,
                position_timeout_s=300.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        fallback = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            225.0,
        )
        candidate = planner.update(
            geometry(completion_candidate=True),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=2.0,
            ),
            226.6,
        )
        parked = planner.update(
            geometry(completion_candidate=True),
            lidar_gap(
                0.0,
                reached=True,
                scan_timestamp=3.0,
            ),
            226.7,
        )
        exiting = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            229.8,
        )
        done = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            231.5,
        )

        self.assertEqual(
            fallback.state,
            ParkingState.FINISH_REVERSE_TIMED,
        )
        self.assertLess(fallback.command.speed, 0)
        self.assertEqual(candidate.state, ParkingState.PARK_CONFIRM)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(exiting.state, ParkingState.EXIT_RIGHT)
        self.assertEqual(done.state, ParkingState.EXIT_DONE)
        self.assertLess(231.5, 240.0)

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
        self.assertEqual(searching.reason, "waiting_for_lidar_scan")

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
            LidarParkingObservation(
                valid=True,
                observed_points=10,
                reason="searching_for_parked_cars",
            ),
            0.1,
        )
        waiting = planner.update(
            geometry(),
            LidarParkingObservation(reason="no_scan"),
            1.1,
        )

        self.assertGreater(rollout.command.speed, 0)
        self.assertEqual(rollout.command.steering, -10)
        self.assertEqual(waiting.command.speed, 0)
        self.assertEqual(waiting.command.steering, 0)
        self.assertEqual(waiting.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(waiting.reason, "lidar_unavailable:no_scan")
        self.assertEqual(planner._prealign_steering(), -150)
        self.assertEqual(planner._fixed_right_entry_steering(), 150)

    def test_unconfirmed_first_car_does_not_arm_prealign(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
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

    def test_start_rollout_drives_straight_even_with_immediate_first_car(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.8,
                provisional_prealign_speed=12,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.2,
        )
        after_rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.9,
        )
        after_delay = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            2.0,
        )

        self.assertEqual(rollout.state, ParkingState.SEARCH_CARS)
        self.assertEqual(rollout.command.speed, planner.config.search_speed)
        self.assertEqual(rollout.command.steering, 0)
        self.assertFalse(rollout.command.brake)
        self.assertEqual(rollout.reason, "start_forward_rollout")
        self.assertEqual(
            after_rollout.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(after_rollout.command.speed, 0)
        self.assertEqual(
            after_rollout.command.steering,
            planner.config.prealign_steering,
        )
        self.assertEqual(
            after_rollout.reason,
            "first_car_turn_reached:provisional_settle_max_left",
        )
        self.assertEqual(
            after_delay.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(
            after_delay.command.speed,
            planner.config.provisional_prealign_speed,
        )
        self.assertEqual(after_delay.command.steering, planner.config.prealign_steering)

    def test_non_right_lidar_clusters_do_not_trigger_prealign(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
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

    def test_prealign_keeps_searching_until_second_car_without_timeout(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                prealign_enabled=True,
                provisional_prealign_speed=12,
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
        self.assertEqual(
            turn_point.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(
            still_searching.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(turn_point.command.speed, 0)
        self.assertEqual(
            still_searching.command.speed,
            planner.config.provisional_prealign_speed,
        )
        self.assertEqual(turn_point.command.steering, planner.config.prealign_steering)
        self.assertEqual(still_searching.command.steering, planner.config.prealign_steering)
        self.assertFalse(turn_point.command.brake)
        self.assertFalse(still_searching.command.brake)
        self.assertEqual(
            still_searching.reason,
            "provisional_prealign_waiting_for_second_car",
        )

    def test_prealign_missing_tracked_slot_keeps_moving(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        waiting = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.2,
        )

        self.assertEqual(waiting.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(waiting.command.speed, planner.config.prealign_speed)
        self.assertEqual(waiting.command.steering, planner.config.prealign_steering)
        self.assertFalse(waiting.command.brake)
        self.assertEqual(waiting.reason, "prealign_waiting_for_tracked_slot")

    def test_provisional_prealign_stops_on_first_safe_path_candidate(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                provisional_prealign_speed=8,
                prealign_speed=35,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)
        planner.update(geometry(), first_car_lidar(), 0.1)
        entered = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.2,
        )

        stopped = planner.update(
            geometry(
                heading=40.0,
                lateral=0.2,
                remaining=650.0,
                reason="lidar_slot_box",
            ),
            prealign_lidar(
                slot_heading_deg=40.0,
                entry_bearing_deg=40.0,
                distance_mm=1200.0,
            ),
            0.3,
        )

        self.assertEqual(
            entered.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(entered.command.steering, -150)
        self.assertEqual(stopped.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(stopped.command.speed, 0)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(stopped.reason, "provisional_safe_path_ready")

    def test_prealign_visible_slot_box_must_be_centered_before_reverse_setup(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        seen = planner.update(
            geometry(
                heading=47.0,
                lateral=-0.42,
                remaining=655.0,
                reason="lidar_slot_box",
            ),
            prealign_lidar(
                slot_heading_deg=47.0,
                entry_bearing_deg=89.0,
                distance_mm=1590.0,
            ),
            0.2,
        )

        self.assertEqual(seen.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(seen.command.speed, planner.config.prealign_speed)
        self.assertEqual(seen.command.steering, planner.config.prealign_steering)
        self.assertFalse(seen.command.brake)
        self.assertIn("centerX=", seen.reason)

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
            ),
            lidar,
            0.2,
        )
        self.assertEqual(confirming.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            confirming.reason,
            "prealign_curve_reverse_ready:first_candidate",
        )
        self.assertTrue(confirming.command.brake)

    def test_full_lock_curve_entry_uses_heading_hysteresis_without_stopping(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        lidar = prealign_lidar(
            slot_heading_deg=60.0,
            entry_bearing_deg=45.0,
            distance_mm=1200.0,
        )
        saturated_path = ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            entry_steering_ratio=1.0,
            reason="full_arc_straight_ready",
        )
        planner.path_generator.generate = lambda _geometry: saturated_path

        blocked = planner.update(
            geometry(heading=50.1, reason="lidar_slot_box"),
            lidar,
            0.2,
        )
        still_blocked = planner.update(
            geometry(heading=49.0, reason="lidar_slot_box"),
            lidar,
            0.3,
        )
        released = planner.update(
            geometry(heading=47.9, reason="lidar_slot_box"),
            lidar,
            0.4,
        )

        self.assertEqual(blocked.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(blocked.command.speed, planner.config.prealign_speed)
        self.assertFalse(blocked.command.brake)
        self.assertIn("prealign_curve_entry_guard", blocked.reason)
        self.assertEqual(still_blocked.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(still_blocked.command.speed, planner.config.prealign_speed)
        self.assertEqual(released.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            released.reason,
            "prealign_curve_reverse_ready:first_candidate",
        )
        self.assertTrue(released.command.brake)

    def test_low_steering_curve_path_cannot_arm_fixed_full_lock_command(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        lidar = prealign_lidar(
            slot_heading_deg=60.0,
            entry_bearing_deg=45.0,
            distance_mm=1200.0,
        )
        planner.path_generator.generate = lambda _geometry: ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            entry_steering_ratio=0.8,
            reason="full_arc_straight_ready",
        )

        confirming = planner.update(
            geometry(heading=57.7, reason="lidar_slot_box"),
            lidar,
            0.2,
        )

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertGreater(confirming.command.speed, 0)
        self.assertIn(
            "prealign_full_lock_path_required",
            confirming.reason,
        )

    def test_point_nine_curve_path_is_blocked_until_heading_releases(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        lidar = prealign_lidar(
            slot_heading_deg=60.0,
            entry_bearing_deg=45.0,
            distance_mm=1200.0,
        )
        planner.path_generator.generate = lambda _geometry: ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            entry_steering_ratio=0.9,
            reason="full_arc_straight_ready",
        )

        blocked = planner.update(
            geometry(heading=50.1, reason="lidar_slot_box"),
            lidar,
            0.2,
        )

        self.assertEqual(blocked.state, ParkingState.PREALIGN_LEFT)
        self.assertIn("prealign_full_lock_path_required", blocked.reason)

    def test_reverse_entry_switches_to_right_steering_after_left_prealign(self):
        planner = self.make_prealign_planner()
        path = ReversePath(
            found=True,
            points=((300.0, 570.0), (360.0, 500.0), (420.0, 320.0)),
            lookahead_point=(360.0, 500.0),
            curvature_per_px=planner.path_generator.config.full_steering_curvature_per_px,
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

        self.assertEqual(steering, planner.config.reverse_entry_min_steering)

    def test_full_path_candidate_ratio_maps_to_configurable_steering(self):
        planner = self.make_planner()
        path = ReversePath(
            found=True,
            curvature_per_px=0.004,
            entry_steering_ratio=0.8,
            reason="full_arc_straight_ready",
        )

        steering = planner._candidate_entry_steering(path)

        self.assertEqual(steering, 120)

    def test_curve_arm_uses_maximum_right_steering_not_candidate_ratio(self):
        planner = self.make_planner()
        planner.path_generator.generate = lambda _geometry: ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            curvature_per_px=0.004,
            entry_steering_ratio=1.0,
            reason="full_arc_straight_ready",
        )
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)

        armed = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.3,
        )

        self.assertEqual(armed.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(
            armed.command.steering,
            planner.config.max_steering,
        )

    def test_curve_reverse_keeps_armed_candidate_when_local_path_changes_side(self):
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
            self.assertEqual(
                plan.reason,
                "following_entry_curve:candidate_steer",
            )

    def test_curve_reverse_release_uses_one_heading_scan_after_minimum_time(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_curve_s=0.3,
                reverse_entry_steering_release_s=0.4,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        self.arm_reverse(planner)

        curving = planner.update(
            geometry(heading=70.0, lateral=0.0, remaining=650.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        still_turning = planner.update(
            geometry(heading=70.0, lateral=0.0, remaining=640.0),
            lidar_gap(0.0, reached=True),
            0.81,
        )
        release_start = planner.update(
            geometry(heading=10.0, lateral=0.0, remaining=635.0),
            lidar_gap(0.0, reached=True),
            0.82,
        )
        releasing = planner.update(
            geometry(heading=8.0, lateral=0.0, remaining=630.0),
            lidar_gap(0.0, reached=True),
            1.02,
        )
        released = planner.update(
            geometry(heading=5.0, lateral=0.0, remaining=620.0),
            lidar_gap(0.0, reached=True),
            1.23,
        )

        self.assertEqual(curving.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(
            curving.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(
            still_turning.state,
            ParkingState.FOLLOW_ENTRY_CURVE,
        )
        self.assertEqual(
            release_start.state,
            ParkingState.RELEASE_ENTRY_STEER,
        )
        self.assertEqual(
            release_start.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(release_start.command.speed, 0)
        self.assertEqual(
            release_start.reason,
            "release_entry_steer:heading_ready_stop",
        )
        self.assertEqual(releasing.state, ParkingState.RELEASE_ENTRY_STEER)
        self.assertEqual(releasing.command.speed, 0)
        self.assertGreater(releasing.command.steering, 0)
        self.assertLess(
            releasing.command.steering,
            planner.config.max_steering,
        )
        self.assertEqual(released.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(released.command.speed, 0)
        self.assertEqual(
            released.reason,
            "following_slot_center:entry_steering_released",
        )

    def test_fresh_aligned_obstacles_switch_to_trimmed_straight_reverse(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-20,
                reverse_aligned_speed=-40,
                aligned_confirm_frames=2,
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_curve_s=0.0,
                reverse_entry_steering_release_s=0.0,
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

        planner.update(
            geometry(heading=4.0, lateral=0.1, remaining=490.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.6,
        )
        confirming = planner.update(
            geometry(heading=3.0, lateral=0.1, remaining=480.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.7,
        )
        straight = planner.update(
            geometry(heading=3.0, lateral=0.1, remaining=470.0),
            lidar_gap(0.0, reached=True, pair_observed=True),
            0.8,
        )

        self.assertEqual(confirming.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertEqual(straight.command.speed, -40)
        self.assertEqual(straight.command.steering, -20)
        self.assertEqual(
            straight.reason,
            "following_slot_center:aligned_straight",
        )

    def test_aligned_reverse_time_limit_waits_for_real_completion(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                straight_steering_trim=-20,
                reverse_aligned_speed=-40,
                aligned_confirm_frames=1,
                aligned_reverse_min_s=0.4,
                aligned_reverse_max_s=2.5,
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_curve_s=0.0,
                reverse_entry_steering_release_s=0.0,
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
        straight = planner.update(aligned_geometry, aligned_lidar, 0.7)

        still_reversing = planner.update(
            aligned_geometry,
            aligned_lidar,
            3.3,
        )

        self.assertEqual(straight.command.speed, -40)
        self.assertEqual(
            still_reversing.state,
            ParkingState.FOLLOW_SLOT_CENTER,
        )
        self.assertEqual(
            still_reversing.reason,
            "following_slot_center:aligned_waiting_for_completion",
        )
        self.assertEqual(still_reversing.command.speed, -40)
        self.assertFalse(still_reversing.command.brake)

    def test_curve_reverse_settles_maximum_right_before_moving(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                reverse_entry_steer_settle_s=0.4,
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

        self.assertEqual(settling.state, ParkingState.SET_REVERSE_STEER)
        self.assertEqual(settling.command.speed, 0)
        self.assertEqual(settling.command.steering, planner.config.max_steering)
        self.assertEqual(
            settling.reason,
            "reverse_entry_candidate_steer:settling",
        )
        self.assertEqual(reversing.state, ParkingState.FOLLOW_ENTRY_CURVE)
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
            2.0,
        )
        released = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=580.0,
                reason="lidar_slot_box",
            ),
            lidar_gap(0.0, reached=True),
            2.7,
        )

        self.assertEqual(first_reverse.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(first_reverse.command.steering, planner.config.max_steering)
        for plan in (
            correction_start,
            correcting_forward,
            reverse_settle,
            correcting_reverse,
        ):
            self.assertIn(
                plan.state,
                (
                    ParkingState.FOLLOW_ENTRY_CURVE,
                    ParkingState.RELEASE_ENTRY_STEER,
                ),
            )
            self.assertLess(plan.command.speed, 0)
            self.assertNotIn(
                plan.state,
                (
                    ParkingState.CORRECT_FORWARD,
                    ParkingState.CORRECT_REVERSE,
                ),
            )
        self.assertEqual(aligned.state, ParkingState.RELEASE_ENTRY_STEER)
        self.assertEqual(released.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(released.command.speed, 0)
        self.assertEqual(released.command.steering, 0)
        self.assertEqual(
            released.reason,
            "following_slot_center:entry_steering_released",
        )

    def test_lidar_obstacle_latches_emergency_stop(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(geometry(), lidar_gap(unsafe=True), 0.5)
        still_stopped = planner.update(geometry(), lidar_gap(), 0.6)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(still_stopped.state, ParkingState.EMERGENCY_STOP)

    def test_missing_lidar_never_allows_reverse_motion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(
            geometry(),
            LidarParkingObservation(reason="stale_scan"),
            0.5,
        )

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(stopped.reason, "lidar_unavailable:stale_scan")

    def test_lidar_loss_during_prealign_latches_emergency_stop(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        stopped = planner.update(
            geometry(),
            LidarParkingObservation(reason="lidar_error:disconnected"),
            0.2,
        )
        latched = planner.update(geometry(), prealign_lidar(), 0.3)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(
            stopped.reason,
            "lidar_unavailable:lidar_error:disconnected",
        )
        self.assertEqual(latched.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(latched.reason, "emergency_stop_latched")

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
            prealign_lidar(
                slot_heading_deg=70.0,
                entry_bearing_deg=89.0,
                distance_mm=1600.0,
            ),
            0.2,
        )

        self.assertEqual(entered.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.command.speed, 14)
        self.assertEqual(moving.command.steering, -150)

    def test_first_car_slows_then_starts_left_turn_before_gap_confirmation(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_approach_speed=10,
                provisional_prealign_speed=8,
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
        self.assertEqual(
            settling.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(settling.command.speed, 0)
        self.assertEqual(settling.command.steering, -150)
        self.assertEqual(
            settling.reason,
            "first_car_turn_reached:provisional_settle_max_left",
        )
        self.assertEqual(
            delayed.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(delayed.command.speed, 8)
        self.assertEqual(delayed.command.steering, -150)
        self.assertEqual(
            turning.state,
            ParkingState.PROVISIONAL_PREALIGN,
        )
        self.assertEqual(turning.command.speed, 8)
        self.assertEqual(turning.command.steering, -150)
        self.assertLess(
            turning.command.speed,
            planner.config.prealign_speed,
        )

    def test_direct_angles_select_mode_without_extra_confirmation_frames(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        aligned_lidar = prealign_lidar(
            slot_heading_deg=5.0,
            entry_bearing_deg=8.0,
            distance_mm=900.0,
        )

        verifying = planner.update(geometry(), aligned_lidar, 0.2)
        armed = planner.update(geometry(), aligned_lidar, 0.3)

        self.assertEqual(verifying.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(
            verifying.reason,
            "prealign_direct_reverse_ready:first_candidate",
        )
        self.assertEqual(armed.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertEqual(armed.reason, "reverse_path_armed:direct")
        self.assertTrue(armed.command.brake)

    def test_prealign_timeout_continues_when_path_is_not_feasible(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        planner.update(
            ParkingGeometry(reason="lidar_slot_box_unavailable"),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            0.2,
        )
        fallback = planner.update(
            ParkingGeometry(reason="lidar_slot_box_unavailable"),
            prealign_lidar(slot_heading_deg=85.0, entry_bearing_deg=80.0),
            2.3,
        )

        self.assertEqual(fallback.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(fallback.reason, "prealign_alignment_timeout:continuing")
        self.assertEqual(fallback.command.speed, planner.config.prealign_speed)
        self.assertFalse(fallback.command.brake)

if __name__ == "__main__":
    unittest.main()
