from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees, hypot
from typing import Optional

from ..estimation.parking_geometry import ParkingGeometry
from ..estimation.parking_lidar import LidarParkingObservation
from ..types import ControlCommand
from .reverse_parking_path import (
    ReverseParkingPathGenerator,
    ReversePath,
    ReversePathConfig,
    ReversePathStatus,
)


class ParkingState(str, Enum):
    IDLE = "idle" #시작 전 대기
    SEARCH_CARS = "search_cars" #첫번째 장애물 찾기
    TRACK_GAP = "track_gap" #첫번째 장애물을 추적하며 두번째 장애물과의 공간 찾기
    POSITION_REAR_AXLE = "position_rear_axle" #후륜을 주차 공간에 맞추는 작업 (선택적 실행)
    PROVISIONAL_PREALIGN = "provisional_prealign"
    PREALIGN_LEFT = "prealign_left" #좌조향 실행
    VERIFY_SLOT_BOX = "verify_slot_box" #라이다로 만든 주차 공간이 안정적인지 확인
    SET_REVERSE_STEER = "set_reverse_steer"
    FOLLOW_ENTRY_CURVE = "follow_entry_curve" #곡선 경로를 따라 후진
    RELEASE_ENTRY_STEER = "release_entry_steer"
    FOLLOW_SLOT_CENTER = "follow_slot_center" #직선 경로를 따라 후진
    CORRECT_FORWARD = "correct_forward" #오차가 크면 전진
    CORRECT_REVERSE = "correct_reverse" #다시 후진
    PARK_CONFIRM = "park_confirm"
    REACQUIRE_SLOT = "reacquire_slot"
    FINISH_REVERSE_TIMED = "finish_reverse_timed"
    PARKED = "parked"
    EXIT_RIGHT = "exit_right"
    EXIT_STRAIGHT = "exit_straight"
    EXIT_DONE = "exit_done"
    ABORTED = "aborted"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class ParkingPlannerConfig:
    search_speed: int = 42 #첫 차량을 찾으면서 직진하는 속도
    start_forward_s: float = 0.8 #초반에 무조건 직진하는 시간 (초)
    straight_steering_trim: int = 0
    gap_tracking_speed: int = 30 #두 차량 사이 공간을 추적할 때의 모터 속도
    position_speed: int = 22 #후륜축 중심을 공간 중심에 맞출 때 사용하는 속도
    first_car_preemptive_turn_enabled: bool = True
    first_car_approach_speed: int = 30 #첫 차량 감지 후 기준 위치까지 접근하는 속도
    prealign_enabled: bool = True
    provisional_prealign_speed: int = 35
    prealign_speed: int = 42 #최대 좌조향에서 전진하는 속도
    prealign_steering: int = -120 #사전 정렬 좌조향 명령
    prealign_steer_settle_s: float = 0.40 #좌조향 진행 직전 정지 상태로 바퀴만 움직이는 시간
    prealign_timeout_s: float = 6.0
    prealign_gap_acquire_timeout_s: float = 0.0 #두번째 공간을 기다리는 제한 시간
    prealign_slot_heading_tolerance_deg: float = 12.0
    prealign_entry_bearing_tolerance_deg: float = 12.0
    prealign_center_x_tolerance_mm: float = 180.0
    prealign_curve_center_x_tolerance_mm: float = 1400.0
    prealign_curve_heading_block_deg: float = 50.0
    prealign_curve_heading_release_deg: float = 48.0
    prealign_curve_saturated_steering_ratio: float = 1.0
    prealign_target_distance_min_mm: float = 900.0
    prealign_target_distance_max_mm: float = 2600.0
    reverse_entry_speed: int = -60 #곡선 후진 속도
    reverse_center_speed: int = -50 #정렬된 후 직선 후진 속도
    reverse_aligned_speed: int = -40
    aligned_reverse_min_s: float = 0.40
    aligned_reverse_max_s: float = 2.50
    reverse_entry_min_steering: int = 90 #곡선 진입 시 최소 조향
    reverse_entry_steer_settle_s: float = 0.40
    reverse_entry_curve_s: float = 0.30
    reverse_entry_release_heading_deg: float = 12.0
    reverse_entry_steering_release_s: float = 0.60
    correction_enabled: bool = True
    correction_forward_speed: int = 22
    correction_reverse_speed: int = -50
    correction_steering: int = 130
    correction_steer_settle_s: float = 0.25
    correction_forward_s: float = 0.70
    correction_reverse_s: float = 1.10
    correction_min_reverse_s: float = 0.80
    correction_depth_trigger_px: float = 760.0
    correction_heading_trigger_deg: float = 35.0
    correction_lateral_trigger_norm: float = 0.30
    correction_trigger_frames: int = 3
    correction_max_attempts: int = 3
    park_confirm_timeout_s: float = 2.0
    slot_reacquire_timeout_s: float = 0.60
    collision_realign_min_forward_s: float = 0.80
    recovery_finish_reverse_speed: int = -40
    recovery_finish_curve_s: float = 0.30
    recovery_finish_total_s: float = 1.50
    mission_soft_deadline_s: float = 225.0
    park_hold_s: float = 3.0
    exit_speed: int = 30
    exit_turn_steering: int = 80
    exit_turn_s: float = 1.6
    exit_straight_s: float = 0.0
    max_steering: int = 150 #후진 경로 추종 최대 조향 크기
    reverse_steering_sign: float = 1.0
    geometry_confidence_min: float = 0.20
    aligned_heading_deg: float = 8.0
    aligned_lateral_norm: float = 0.18
    aligned_confirm_frames: int = 4
    stop_depth_margin_px: float = 8.0
    search_timeout_s: float = 60.0
    gap_tracking_timeout_s: float = 20.0
    position_timeout_s: float = 10.0
    entry_curve_timeout_s: float = 16.0
    center_follow_timeout_s: float = 10.0


@dataclass(frozen=True)
class ParkingPlan:
    state: ParkingState
    command: ControlCommand
    reason: str
    path: Optional[ReversePath] = None


class TParkingPlanner:
    """LiDAR gap positioning followed by LiDAR-box-guided reverse parking."""

    def __init__(
        self,
        config: ParkingPlannerConfig = ParkingPlannerConfig(),
        path_config: ReversePathConfig = ReversePathConfig(),
    ):
        self.config = config
        self.path_generator = ReverseParkingPathGenerator(path_config)
        self.state = ParkingState.IDLE
        self._state_started_at = 0.0
        self._aligned_frames = 0
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._right_first_car_acquired = False
        self._prealign_gap_acquired_at: Optional[float] = None
        self._reverse_entry_mode = "lidar_box_curve"
        self._aligned_reverse_started_at: Optional[float] = None
        self._lidar_acquired = False
        self._park_candidate_scan_timestamp: Optional[float] = None
        self._reacquire_resume_state = ParkingState.FOLLOW_SLOT_CENTER
        self._preverify_state = ParkingState.PROVISIONAL_PREALIGN
        self._armed_reverse_path: Optional[ReversePath] = None
        self._armed_entry_steering = 0
        self._recovery_curve_steering = 0
        self._reverse_path_armed = False
        self._reverse_started = False
        self._mission_started_at = 0.0
        self._timed_finish_from_curve = False
        self._timed_finish_curve_released = False
        self._curve_entry_guard_active = False
        self._collision_realign_until: Optional[float] = None

    def start(self, now: float) -> bool:
        if self.state not in (
            ParkingState.IDLE,
            ParkingState.ABORTED,
            ParkingState.EMERGENCY_STOP,
            ParkingState.PARKED,
            ParkingState.EXIT_DONE,
        ):
            return False
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._right_first_car_acquired = False
        self._aligned_reverse_started_at = None
        self._lidar_acquired = False
        self._park_candidate_scan_timestamp = None
        self._reacquire_resume_state = ParkingState.FOLLOW_SLOT_CENTER
        self._preverify_state = ParkingState.PROVISIONAL_PREALIGN
        self._armed_reverse_path = None
        self._armed_entry_steering = 0
        self._recovery_curve_steering = 0
        self._reverse_path_armed = False
        self._reverse_started = False
        self._mission_started_at = now
        self._timed_finish_from_curve = False
        self._timed_finish_curve_released = False
        self._curve_entry_guard_active = False
        self._collision_realign_until = None
        self._enter(ParkingState.SEARCH_CARS, now)
        return True

    def reset(self, now: float = 0.0) -> None:
        self.state = ParkingState.IDLE
        self._state_started_at = now
        self._aligned_frames = 0
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._right_first_car_acquired = False
        self._prealign_gap_acquired_at = None
        self._reverse_entry_mode = "lidar_box_curve"
        self._aligned_reverse_started_at = None
        self._lidar_acquired = False
        self._park_candidate_scan_timestamp = None
        self._reacquire_resume_state = ParkingState.FOLLOW_SLOT_CENTER
        self._preverify_state = ParkingState.PROVISIONAL_PREALIGN
        self._armed_reverse_path = None
        self._armed_entry_steering = 0
        self._recovery_curve_steering = 0
        self._reverse_path_armed = False
        self._reverse_started = False
        self._mission_started_at = now
        self._timed_finish_from_curve = False
        self._timed_finish_curve_released = False
        self._curve_entry_guard_active = False
        self._collision_realign_until = None

    def update(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        enabled: bool = True,
    ) -> ParkingPlan:
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")
        # Parking is intentionally LiDAR-only.
        if lidar.valid:
            self._lidar_acquired = True
        elif self._lidar_acquired:
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("lidar_unavailable:%s" % lidar.reason)
        else:
            return self._stop("waiting_for_lidar_scan")
        if self.state == ParkingState.PARKED:
            if self._state_elapsed(now) >= max(0.0, self.config.park_hold_s):
                self._enter(ParkingState.EXIT_RIGHT, now)
                return self._exit_right_plan(now)
            return self._stop("parked_hold")
        if self.state == ParkingState.EXIT_DONE:
            return self._stop("exit_done")
        if self.state == ParkingState.ABORTED:
            return self._stop("parking_aborted")
        if self.state == ParkingState.EMERGENCY_STOP:
            return self._stop("emergency_stop_latched")
        if (
            self.config.mission_soft_deadline_s > 0.0
            and now - self._mission_started_at
            >= self.config.mission_soft_deadline_s
            and not lidar.unsafe
            and self.state
            not in (
                ParkingState.FINISH_REVERSE_TIMED,
                ParkingState.PARK_CONFIRM,
                ParkingState.PARKED,
                ParkingState.EXIT_RIGHT,
                ParkingState.EXIT_STRAIGHT,
                ParkingState.EXIT_DONE,
            )
        ):
            if self._reverse_started or self._reverse_path_armed:
                return self._begin_timed_finish(
                    geometry,
                    lidar,
                    now,
                    "mission_soft_deadline:finish_reverse",
                )
            return self._abort(now, "mission_deadline_without_armed_path")

        slot_and_reverse_states = (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.SET_REVERSE_STEER,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.RELEASE_ENTRY_STEER,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
            ParkingState.PARK_CONFIRM,
            ParkingState.REACQUIRE_SLOT,
            ParkingState.FINISH_REVERSE_TIMED,
        )
        if lidar.unsafe and self.state in slot_and_reverse_states:
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("lidar_safety_obstacle")

        if self.state in (
            ParkingState.PROVISIONAL_PREALIGN,
            ParkingState.PREALIGN_LEFT,
        ):
            if lidar.unsafe:
                self._enter(ParkingState.EMERGENCY_STOP, now)
                return self._stop("lidar_safety_obstacle_during_prealign")

        if self.state == ParkingState.EXIT_RIGHT:
            return self._exit_right_plan(now)

        if self.state == ParkingState.EXIT_STRAIGHT:
            if (
                self.config.exit_straight_s > 0.0
                and self._state_elapsed(now) >= self.config.exit_straight_s
            ):
                self._enter(ParkingState.EXIT_DONE, now)
                return self._stop("exit_complete")
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )

        if self.state == ParkingState.FINISH_REVERSE_TIMED:
            return self._timed_finish_plan(geometry, lidar, now)

        if self.state == ParkingState.PARK_CONFIRM:
            return self._park_confirm_plan(geometry, lidar, now)

        if self.state == ParkingState.REACQUIRE_SLOT:
            return self._reacquire_slot_plan(geometry, lidar, now)

        completion_monitor_states = (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.RELEASE_ENTRY_STEER,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
        )
        if (
            self.state in completion_monitor_states
            and lidar.is_new_scan
            and self._parking_completion_candidate(geometry)
        ):
            return self._begin_park_confirm(lidar, now)

        localized_reverse_states = (
            ParkingState.SET_REVERSE_STEER,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.RELEASE_ENTRY_STEER,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
        )
        if (
            self.state in localized_reverse_states
            and not self._full_geometry_usable(geometry)
        ):
            return self._begin_reacquire_slot(
                now,
                "slot_pose_unavailable_during_reverse",
                verify_after_reacquire=True,
            )

        if self.state == ParkingState.SEARCH_CARS:
            if self._expired(now, self.config.search_timeout_s):
                return self._abort(now, "parked_car_search_timeout")
            if self._state_elapsed(now) < max(0.0, self.config.start_forward_s):
                return self._drive(
                    self.config.search_speed,
                    self._straight_steering(),
                    "start_forward_rollout",
                )
            if not lidar.valid:
                return self._stop("waiting_for_lidar_scan")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_turn_reached
            ):
                self._right_first_car_acquired = True
                self._enter(ParkingState.PROVISIONAL_PREALIGN, now)
                if lidar.gap_confirmed:
                    self._prealign_gap_acquired_at = now
                return self._drive(
                    0,
                    self._prealign_steering(),
                    "first_car_turn_reached:provisional_settle_max_left",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_confirmed
            ):
                self._right_first_car_acquired = True
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_confirmed:creeping_to_turn_point",
                )
            if lidar.gap_confirmed:
                self._right_first_car_acquired = True
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            if lidar.gap_found or lidar.first_car_confirmed:
                self._right_first_car_acquired = True
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.gap_tracking_speed,
                    self._straight_steering(),
                    "tracking_parked_cars",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_seen
            ):
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_seen:waiting_for_confirmation",
                )
            return self._drive(
                self.config.search_speed,
                self._straight_steering(),
                "searching_for_parked_cars",
            )

        if self.state == ParkingState.TRACK_GAP:
            if self._expired(now, self.config.gap_tracking_timeout_s):
                return self._abort(now, "two_car_gap_timeout")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
            ):
                turn_ready = lidar.first_car_turn_reached or lidar.gap_confirmed
                if not turn_ready:
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_creeping_to_turn_point",
                    )
                if (
                    lidar.first_car_turn_reached
                    or lidar.gap_confirmed
                    or lidar.first_car_confirmed
                    or lidar.first_car_seen
                    or lidar.gap_found
                    or self._right_first_car_acquired
                ):
                    self._enter(ParkingState.PROVISIONAL_PREALIGN, now)
                    if lidar.gap_confirmed:
                        self._prealign_gap_acquired_at = now
                    return self._drive(
                        0,
                        self._prealign_steering(),
                        "first_car_turn_reached:provisional_settle_max_left",
                    )
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_temporarily_lost:creeping_to_turn_point",
                )
            if lidar.gap_confirmed:
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            return self._drive(
                self.config.gap_tracking_speed,
                self._straight_steering(),
                "confirming_two_car_gap",
            )

        if self.state == ParkingState.PROVISIONAL_PREALIGN:
            elapsed = now - self._state_started_at
            if self._collision_realign_until is not None:
                if now < self._collision_realign_until:
                    return self._provisional_prealign_drive(
                        elapsed,
                        "collision_risk_forward_realign",
                    )
                self._collision_realign_until = None
            if not lidar.gap_confirmed:
                if self._expired(now, self.config.prealign_gap_acquire_timeout_s):
                    return self._abort(now, "second_car_gap_acquire_timeout")
                return self._provisional_prealign_drive(
                    elapsed,
                    "provisional_prealign_waiting_for_second_car",
                )
            if (
                lidar.coasted
                or lidar.gap_center_x_right_mm is None
                or lidar.gap_center_y_back_mm is None
                or lidar.entry_target_y_back_mm is None
                or lidar.slot_depth_x_right is None
                or lidar.slot_depth_y_back is None
            ):
                return self._provisional_prealign_drive(
                    elapsed,
                    "provisional_prealign_waiting_for_slot_pose",
                )

            if self._prealign_gap_acquired_at is None:
                self._prealign_gap_acquired_at = now
            readiness = self._prealign_readiness(geometry, lidar)
            if readiness is None:
                return self._provisional_prealign_drive(
                    elapsed,
                    "provisional_prealign_invalid_slot_pose",
                )
            (
                slot_heading_deg,
                entry_bearing_deg,
                target_distance_mm,
                center_x_mm,
                direct_ready,
                path_ready,
                prealign_path,
            ) = readiness
            curve_entry_blocked = self._curve_entry_guard_blocks(
                geometry,
                direct_ready,
                prealign_path,
            )
            if path_ready and curve_entry_blocked:
                return self._provisional_prealign_drive(
                    elapsed,
                    self._curve_entry_guard_reason(geometry, prealign_path),
                    prealign_path,
                )
            if path_ready:
                self._clear_reverse_arm()
                self._preverify_state = self.state
                self._reverse_entry_mode = (
                    "direct_aligned" if direct_ready else "lidar_box_curve"
                )
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                reason = (
                    "provisional_direct_reverse_ready:first_candidate"
                    if direct_ready
                    else "provisional_safe_path_ready"
                )
                return self._stop(reason, prealign_path)

            reason = (
                "provisional_prealign head=%+.1f bearing=%+.1f "
                "centerX=%+.0fmm dist=%.0fmm path=%s:%s"
            ) % (
                slot_heading_deg,
                entry_bearing_deg,
                center_x_mm,
                target_distance_mm,
                (
                    prealign_path.status.value
                    if prealign_path is not None
                    else "unavailable"
                ),
                (
                    prealign_path.reason
                    if prealign_path is not None
                    else "geometry_unusable"
                ),
            )
            return self._provisional_prealign_drive(
                elapsed,
                reason,
                prealign_path,
            )

        if self.state == ParkingState.POSITION_REAR_AXLE:
            if self._expired(now, self.config.position_timeout_s):
                return self._abort(now, "rear_axle_position_timeout")
            if not lidar.valid or not lidar.gap_confirmed or lidar.entry_error_mm is None:
                return self._stop("rear_axle_waiting_for_gap")
            if lidar.entry_reached:
                if self.config.prealign_enabled:
                    self._enter(ParkingState.PREALIGN_LEFT, now)
                    self._prealign_gap_acquired_at = now
                    steering = self._prealign_steering()
                    return self._drive(
                        0,
                        steering,
                        "rear_axle_at_gap_center:settling_max_left",
                    )
                self._clear_reverse_arm()
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop("rear_axle_at_gap_center")
            direction = -1 if lidar.entry_error_mm > 0.0 else 1
            return self._drive(
                direction * abs(self.config.position_speed),
                self._straight_steering(),
                "correcting_rear_axle_to_gap",
            )

        if self.state == ParkingState.PREALIGN_LEFT:
            elapsed = now - self._state_started_at
            if self._collision_realign_until is not None:
                if now < self._collision_realign_until:
                    return self._prealign_drive(
                        elapsed,
                        "collision_risk_forward_realign",
                    )
                self._collision_realign_until = None
            if not lidar.gap_confirmed:
                if self._expired(now, self.config.prealign_gap_acquire_timeout_s):
                    return self._abort(now, "second_car_gap_acquire_timeout")
                return self._prealign_drive(
                    elapsed,
                    "prealign_left_waiting_for_second_car",
                )
            if (
                lidar.coasted
                or lidar.gap_center_x_right_mm is None
                or lidar.gap_center_y_back_mm is None
                or lidar.entry_target_y_back_mm is None
                or lidar.slot_depth_x_right is None
                or lidar.slot_depth_y_back is None
            ):
                return self._prealign_drive(
                    elapsed,
                    "prealign_waiting_for_tracked_slot",
                )

            if self._prealign_gap_acquired_at is None:
                self._prealign_gap_acquired_at = now

            readiness = self._prealign_readiness(geometry, lidar)
            if readiness is None:
                return self._prealign_drive(
                    elapsed,
                    "prealign_invalid_slot_pose",
                )
            (
                slot_heading_deg,
                entry_bearing_deg,
                target_distance_mm,
                center_x_mm,
                direct_ready,
                path_ready,
                prealign_path,
            ) = readiness
            curve_entry_blocked = self._curve_entry_guard_blocks(
                geometry,
                direct_ready,
                prealign_path,
            )
            if path_ready and curve_entry_blocked:
                return self._prealign_drive(
                    elapsed,
                    self._curve_entry_guard_reason(geometry, prealign_path),
                    prealign_path,
                )
            if path_ready:
                self._clear_reverse_arm()
                self._preverify_state = self.state
                self._reverse_entry_mode = (
                    "direct_aligned" if direct_ready else "lidar_box_curve"
                )
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                reason = (
                    "prealign_direct_reverse_ready:first_candidate"
                    if direct_ready
                    else "prealign_curve_reverse_ready:first_candidate"
                )
                return self._stop(reason, prealign_path)

            timed_out = (
                self.config.prealign_timeout_s > 0.0
                and now - self._prealign_gap_acquired_at
                >= self.config.prealign_timeout_s
            )
            if timed_out:
                return self._prealign_drive(
                    elapsed,
                    "prealign_alignment_timeout:continuing",
                    prealign_path,
                )

            reason = (
                "prealign_left head=%+.1f bearing=%+.1f "
                "centerX=%+.0fmm dist=%.0fmm path=%s:%s"
            ) % (
                slot_heading_deg,
                entry_bearing_deg,
                center_x_mm,
                target_distance_mm,
                (
                    prealign_path.status.value
                    if prealign_path is not None
                    else "unavailable"
                ),
                (
                    prealign_path.reason
                    if prealign_path is not None
                    else "geometry_unusable"
                ),
            )
            return self._prealign_drive(elapsed, reason, prealign_path)

        if self.state == ParkingState.VERIFY_SLOT_BOX:
            if not lidar.is_new_scan:
                return self._stop("waiting_for_unique_lidar_arm_scan")
            if not self._full_geometry_usable(geometry):
                return self._begin_reacquire_slot(
                    now,
                    "slot_arm_geometry_unavailable",
                )
            if lidar.coasted or geometry.coasted:
                return self._begin_reacquire_slot(
                    now,
                    "slot_arm_scan_not_fresh",
                )
            path = self.path_generator.generate(geometry)
            if not path.found or path.status != ReversePathStatus.READY:
                return self._resume_prealign_after_failed_arm(
                    now,
                    "reverse_path_not_armable:%s:%s"
                    % (path.status.value, path.reason),
                    path,
                )
            reverse_entry_mode = self._select_reverse_entry_mode(
                geometry,
                lidar,
            )
            if (
                reverse_entry_mode == "lidar_box_curve"
                and not self._curve_path_matches_fixed_steering(path)
            ):
                return self._resume_prealign_after_failed_arm(
                    now,
                    "reverse_path_not_armable:"
                    "full_lock_path_required:ratio=%.2f"
                    % abs(float(path.entry_steering_ratio)),
                    path,
                )
            self._reverse_entry_mode = reverse_entry_mode
            self._armed_reverse_path = path
            self._reverse_path_armed = True
            if self._reverse_entry_mode == "direct_aligned":
                self._armed_entry_steering = self._straight_steering()
                self._recovery_curve_steering = 0
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._stop("reverse_path_armed:direct", path)
            # The path candidate only decides whether the manoeuvre is
            # feasible. Once a curved reverse is armed, start with the known
            # maximum right lock; scaling this command by a 0.9 candidate was
            # leaving too little yaw authority in the real vehicle.
            self._armed_entry_steering = self._fixed_right_entry_steering()
            self._recovery_curve_steering = self._armed_entry_steering
            self._enter(ParkingState.SET_REVERSE_STEER, now)
            return self._drive(
                0,
                self._armed_entry_steering,
                "reverse_path_armed:curve:set_candidate_steer",
                path,
            )

        path = self.path_generator.generate(geometry)

        if self.state == ParkingState.SET_REVERSE_STEER:
            if self._expired(now, self.config.entry_curve_timeout_s):
                return self._abort(now, "reverse_steer_set_timeout")
            if not self._curve_reverse_arm_valid():
                self._clear_reverse_arm()
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop(
                    "reverse_entry_not_armed:return_verify",
                    path,
                )
            # The complete swept path was approved while stopped. Hold that
            # choice during steering settle so harmless scan jitter cannot
            # prevent the first reverse command indefinitely.
            armed_path = self._armed_reverse_path or path
            steering = self._armed_entry_steering
            if self._state_elapsed(now) < max(
                0.0,
                self.config.reverse_entry_steer_settle_s,
            ):
                return self._drive(
                    0,
                    steering,
                    "reverse_entry_candidate_steer:settling",
                    armed_path,
                )
            self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
            return self._drive(
                self.config.reverse_entry_speed,
                steering,
                "following_entry_curve:candidate_steer",
                armed_path,
            )

        if self.state == ParkingState.CORRECT_FORWARD:
            return self._correction_forward_plan(geometry, path, now)

        if self.state == ParkingState.CORRECT_REVERSE:
            return self._correction_reverse_plan(geometry, path, now)

        if self.state == ParkingState.FOLLOW_ENTRY_CURVE:
            if self._expired(now, self.config.entry_curve_timeout_s):
                return self._abort(now, "entry_curve_timeout")
            stop = self._stop_at_back_line(geometry, now, path)
            if stop is not None:
                return stop
            fixed_curve_mismatch = (
                self._reverse_entry_mode == "lidar_box_curve"
                and not self._curve_path_matches_fixed_steering(path)
            )
            if not path.found or fixed_curve_mismatch:
                self._aligned_frames = 0
                if (
                    path.status == ReversePathStatus.COLLISION_RISK
                    or fixed_curve_mismatch
                ):
                    return self._begin_forward_realign(
                        now,
                        "entry_curve_requires_forward_realign:%s:%s"
                        % (
                            path.status.value,
                            (
                                "full_lock_path_unavailable"
                                if fixed_curve_mismatch
                                else path.reason
                            ),
                        ),
                        path,
                    )
                return self._begin_reacquire_slot(
                    now,
                    "entry_curve_path_lost:%s:%s"
                    % (path.status.value, path.reason),
                    path,
                )
            if self._reverse_entry_mode == "lidar_box_curve":
                steering = self._armed_entry_steering
                release_ready = (
                    self._state_elapsed(now)
                    >= max(0.0, self.config.reverse_entry_curve_s)
                    and abs(geometry.heading_error_deg)
                    <= max(
                        0.0,
                        self.config.reverse_entry_release_heading_deg,
                    )
                )
                if release_ready:
                    self._armed_reverse_path = path
                    self._enter(ParkingState.RELEASE_ENTRY_STEER, now)
                    return self._drive(
                        0,
                        steering,
                        "release_entry_steer:heading_ready_stop",
                        path,
                    )
                return self._drive(
                    self.config.reverse_entry_speed,
                    steering,
                    "following_entry_curve:candidate_steer",
                    path,
                )
            aligned = self._slot_aligned(geometry)
            self._aligned_frames = self._aligned_frames + 1 if aligned else 0
            if self._aligned_frames >= max(1, self.config.aligned_confirm_frames):
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._drive(
                    self.config.reverse_center_speed,
                    self._path_steering(path),
                    "following_slot_center:local_target",
                    path,
                )
            return self._drive(
                self.config.reverse_entry_speed,
                self._entry_curve_steering(path),
                "following_entry_curve:%s" % self._reverse_entry_mode,
                path,
            )

        if self.state == ParkingState.RELEASE_ENTRY_STEER:
            if self._expired(now, self.config.entry_curve_timeout_s):
                return self._abort(now, "entry_steer_release_timeout")
            release_path = self._armed_reverse_path or path
            duration = max(
                0.0,
                self.config.reverse_entry_steering_release_s,
            )
            if duration <= 0.0 or self._state_elapsed(now) >= duration:
                if not path.found or path.status != ReversePathStatus.READY:
                    if path.status == ReversePathStatus.COLLISION_RISK:
                        return self._begin_forward_realign(
                            now,
                            "entry_steer_release_requires_forward_realign:%s"
                            % path.reason,
                            path,
                        )
                    return self._begin_reacquire_slot(
                        now,
                        "entry_steer_released_path_not_ready:%s:%s"
                        % (path.status.value, path.reason),
                        path,
                    )
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._drive(
                    self.config.reverse_center_speed,
                    self._straight_steering(),
                    "following_slot_center:entry_steering_released",
                    path,
                )
            progress = clip(self._state_elapsed(now) / duration, 0.0, 1.0)
            steering = self._released_entry_steering(path, progress)
            return self._drive(
                0,
                steering,
                "release_entry_steer_stopped:progress=%d%%"
                % round(progress * 100.0),
                release_path,
            )

        if self.state == ParkingState.FOLLOW_SLOT_CENTER:
            stop = self._stop_at_back_line(geometry, now, path)
            if stop is not None:
                return stop
            if not path.found:
                if path.status == ReversePathStatus.COLLISION_RISK:
                    return self._begin_forward_realign(
                        now,
                        "slot_center_requires_forward_realign:%s"
                        % path.reason,
                        path,
                    )
                return self._begin_reacquire_slot(
                    now,
                    "slot_center_path_lost:%s" % path.reason,
                    path,
                )
            if self._aligned_reverse_started_at is None:
                fresh_pair_aligned = (
                    lidar.gap_pair_observed
                    and lidar.car_count >= 2
                    and not lidar.coasted
                    and self._slot_aligned(geometry)
                )
                self._aligned_frames = (
                    self._aligned_frames + 1 if fresh_pair_aligned else 0
                )
                if self._aligned_frames >= max(
                    1,
                    self.config.aligned_confirm_frames,
                ):
                    self._aligned_reverse_started_at = now

            if self._aligned_reverse_started_at is not None:
                if self._expired(now, self.config.center_follow_timeout_s):
                    return self._abort(
                        now,
                        "aligned_reverse_slot_completion_timeout",
                    )
                aligned_elapsed = max(
                    0.0,
                    now - self._aligned_reverse_started_at,
                )
                minimum_elapsed = max(
                    0.0,
                    self.config.aligned_reverse_min_s,
                )
                configured_maximum = self.config.aligned_reverse_max_s
                maximum_elapsed = max(minimum_elapsed, configured_maximum)
                drive_reason = (
                    "following_slot_center:aligned_waiting_for_completion"
                    if (
                        configured_maximum > 0.0
                        and aligned_elapsed >= maximum_elapsed
                    )
                    else "following_slot_center:aligned_straight"
                )
                return self._drive(
                    self.config.reverse_aligned_speed,
                    self._straight_steering(),
                    drive_reason,
                    path,
                )
            if self._expired(now, self.config.center_follow_timeout_s):
                return self._abort(now, "slot_center_follow_timeout")
            if self._parking_correction_ready(geometry, now):
                return self._start_parking_correction(geometry, path, now)
            return self._drive(
                self.config.reverse_center_speed,
                self._path_steering(path),
                "following_slot_center:local_target",
                path,
            )

        return self._abort(now, "unknown_state")

    def _parking_completion_candidate(
        self,
        geometry: ParkingGeometry,
    ) -> bool:
        return (
            self._full_geometry_usable(geometry)
            and not geometry.coasted
            and len(geometry.vehicle_footprint_slot_local_mm) == 4
            and geometry.park_completion_candidate
        )

    def _begin_park_confirm(
        self,
        lidar: LidarParkingObservation,
        now: float,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        if self.state != ParkingState.REACQUIRE_SLOT:
            # A rejected completion candidate is already at slot depth; resume
            # center following rather than restarting the entry curve.
            self._reacquire_resume_state = ParkingState.FOLLOW_SLOT_CENTER
        self._enter(ParkingState.PARK_CONFIRM, now)
        self._park_candidate_scan_timestamp = float(lidar.timestamp)
        return self._stop(
            "park_completion_candidate:stopped_for_confirmation",
            path,
        )

    def _park_confirm_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        distinct_new_scan = (
            lidar.is_new_scan
            and self._park_candidate_scan_timestamp is not None
            and float(lidar.timestamp)
            != self._park_candidate_scan_timestamp
        )
        if not distinct_new_scan:
            if self._expired(now, self.config.park_confirm_timeout_s):
                return self._begin_reacquire_slot(
                    now,
                    "park_confirm_scan_timeout:reacquire",
                )
            return self._stop(
                "park_confirm_waiting_for_first_stopped_scan"
            )
        if (
            not self._full_geometry_usable(geometry)
            or geometry.coasted
        ):
            return self._begin_reacquire_slot(
                now,
                "park_confirm_slot_unavailable:reacquire",
                verify_after_reacquire=True,
            )
        if self._parking_completion_candidate(geometry):
            return self._finish_parking(
                now,
                "park_confirmed_on_first_stopped_scan",
            )
        return self._begin_reacquire_slot(
            now,
            "park_confirmation_rejected:reacquire",
        )

    def _begin_reacquire_slot(
        self,
        now: float,
        reason: str,
        path: Optional[ReversePath] = None,
        *,
        verify_after_reacquire: bool = False,
    ) -> ParkingPlan:
        state_before_reacquire = self.state
        if (
            state_before_reacquire
            in (
                ParkingState.SET_REVERSE_STEER,
                ParkingState.FOLLOW_ENTRY_CURVE,
                ParkingState.RELEASE_ENTRY_STEER,
            )
            and self._armed_entry_steering != 0
        ):
            # Path approval is intentionally cleared below, but a vehicle that
            # already started the entry curve must not forget which direction
            # its wheels were set when recovery falls back to timed motion.
            self._recovery_curve_steering = self._armed_entry_steering
        elif state_before_reacquire != ParkingState.REACQUIRE_SLOT:
            self._recovery_curve_steering = 0
        if verify_after_reacquire:
            self._reacquire_resume_state = ParkingState.VERIFY_SLOT_BOX
        elif self.state not in (
            ParkingState.PARK_CONFIRM,
            ParkingState.REACQUIRE_SLOT,
        ):
            self._reacquire_resume_state = (
                self._resume_state_after_reacquire(self.state)
            )
        if (
            verify_after_reacquire
            or state_before_reacquire
            in (
                ParkingState.VERIFY_SLOT_BOX,
                ParkingState.SET_REVERSE_STEER,
            )
        ):
            # A localization loss invalidates the path approval. Recovery only
            # restores observation; VERIFY_SLOT_BOX must explicitly ARM a new
            # path before any reverse command can resume.
            self._clear_reverse_arm()
        self._enter(ParkingState.REACQUIRE_SLOT, now)
        self._park_candidate_scan_timestamp = None
        return self._stop(reason, path)

    def _begin_forward_realign(
        self,
        now: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        """Back out of an unsafe reverse attempt and rebuild a safe entry.

        A fresh collision result is stronger evidence than a temporary pose
        loss. It must never fall through to blind timed reverse. Resetting
        ``_reverse_started`` makes the next attempt earn a new full-lock path
        approval before reverse can resume.
        """

        resume_state = (
            self._preverify_state
            if self._preverify_state
            in (
                ParkingState.PROVISIONAL_PREALIGN,
                ParkingState.PREALIGN_LEFT,
            )
            else ParkingState.PROVISIONAL_PREALIGN
        )
        self._clear_reverse_arm()
        self._recovery_curve_steering = 0
        self._reverse_started = False
        self._timed_finish_from_curve = False
        self._timed_finish_curve_released = False
        self._curve_entry_guard_active = False
        self._preverify_state = resume_state
        self._reacquire_resume_state = ParkingState.VERIFY_SLOT_BOX
        self._enter(resume_state, now)
        self._prealign_gap_acquired_at = now
        self._collision_realign_until = now + max(
            0.0,
            self.config.prealign_steer_settle_s,
        ) + max(
            0.0,
            self.config.collision_realign_min_forward_s,
        )
        return self._stop(
            "%s:forward_realign_settle" % reason,
            path,
        )

    def _resume_prealign_after_failed_arm(
        self,
        now: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        """Resume forward alignment when a stopped pre-reverse check fails."""

        if (
            not self.config.prealign_enabled
            or self._preverify_state
            not in (
                ParkingState.PROVISIONAL_PREALIGN,
                ParkingState.PREALIGN_LEFT,
            )
        ):
            return self._begin_reacquire_slot(now, reason, path)
        resume_state = self._preverify_state
        self._clear_reverse_arm()
        self._enter(resume_state, now)
        self._prealign_gap_acquired_at = now
        return self._stop(
            "%s:resume_%s" % (reason, resume_state.value),
            path,
        )

    def _reacquire_slot_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.slot_reacquire_timeout_s):
            if self._reverse_started:
                return self._begin_timed_finish(
                    geometry,
                    lidar,
                    now,
                    "slot_reacquire_timeout:finish_reverse",
                )
            resume_state = (
                self._preverify_state
                if self._preverify_state
                in (
                    ParkingState.PROVISIONAL_PREALIGN,
                    ParkingState.PREALIGN_LEFT,
                )
                else ParkingState.PROVISIONAL_PREALIGN
            )
            self._clear_reverse_arm()
            self._enter(resume_state, now)
            speed = (
                self.config.provisional_prealign_speed
                if resume_state == ParkingState.PROVISIONAL_PREALIGN
                else self.config.prealign_speed
            )
            return self._drive(
                abs(int(speed)),
                self._prealign_steering(),
                "slot_reacquire_timeout:resume_%s"
                % resume_state.value,
            )
        if not lidar.is_new_scan:
            return self._stop("reacquire_waiting_for_unique_lidar_scan")
        if (
            not self._full_geometry_usable(geometry)
            or geometry.coasted
        ):
            return self._stop("reacquiring_frozen_slot")
        if self._parking_completion_candidate(geometry):
            return self._begin_park_confirm(lidar, now)

        path = self.path_generator.generate(geometry)
        resume_state = self._reacquire_resume_state
        fixed_curve_mismatch = (
            resume_state
            in (
                ParkingState.FOLLOW_ENTRY_CURVE,
                ParkingState.RELEASE_ENTRY_STEER,
            )
            and not self._curve_path_matches_fixed_steering(path)
        )
        if (
            path.status == ReversePathStatus.COLLISION_RISK
            or fixed_curve_mismatch
        ):
            return self._begin_forward_realign(
                now,
                "reacquired_path_requires_forward_realign:%s"
                % (
                    "full_lock_path_unavailable"
                    if fixed_curve_mismatch
                    else path.reason
                ),
                path,
            )
        path_ready = (
            path.found
            and (
                resume_state != ParkingState.SET_REVERSE_STEER
                or path.status == ReversePathStatus.READY
            )
        )
        if not path_ready:
            return self._stop(
                "reacquired_slot_path_not_ready:%s:%s"
                % (path.status.value, path.reason),
                path,
            )
        self._enter(resume_state, now)
        return self._stop(
            "slot_reacquired:resume_pending:%s"
            % resume_state.value,
            path,
        )

    @staticmethod
    def _resume_state_after_reacquire(
        state: ParkingState,
    ) -> ParkingState:
        if state in (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.SET_REVERSE_STEER,
        ):
            return ParkingState.VERIFY_SLOT_BOX
        if state in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.RELEASE_ENTRY_STEER,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            return state
        return ParkingState.FOLLOW_SLOT_CENTER

    def _curve_reverse_arm_valid(self) -> bool:
        return (
            self._reverse_path_armed
            and self._armed_reverse_path is not None
            and self._armed_reverse_path.found
            and self._armed_reverse_path.status == ReversePathStatus.READY
            and self._curve_path_matches_fixed_steering(
                self._armed_reverse_path
            )
            and self._armed_entry_steering != 0
        )

    def _clear_reverse_arm(self) -> None:
        self._reverse_path_armed = False
        self._armed_reverse_path = None
        self._armed_entry_steering = 0

    def _begin_timed_finish(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        reason: str,
    ) -> ParkingPlan:
        self._timed_finish_from_curve = (
            self.state
            in (
                ParkingState.SET_REVERSE_STEER,
                ParkingState.FOLLOW_ENTRY_CURVE,
                ParkingState.RELEASE_ENTRY_STEER,
            )
            or self._reacquire_resume_state
            in (
                ParkingState.FOLLOW_ENTRY_CURVE,
                ParkingState.RELEASE_ENTRY_STEER,
            )
            or self._recovery_curve_steering != 0
        )
        self._timed_finish_curve_released = (
            not self._timed_finish_from_curve
        )
        self._enter(ParkingState.FINISH_REVERSE_TIMED, now)
        return self._timed_finish_plan(geometry, lidar, now, reason)

    def _timed_finish_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        reason: str = "timed_finish_reverse",
    ) -> ParkingPlan:
        elapsed = self._state_elapsed(now)
        nominal_total = max(0.0, self.config.recovery_finish_total_s)
        if (
            lidar.is_new_scan
            and self._parking_completion_candidate(geometry)
        ):
            return self._begin_park_confirm(
                lidar,
                now,
                self._armed_reverse_path,
            )

        # recovery_finish_total_s is only a nominal fallback duration. Time
        # passing is not evidence that the vehicle is inside the slot, so keep
        # reversing until the locked-slot footprint supplies a real completion
        # candidate. Retain a hard fault timeout to avoid unbounded blind
        # motion when localization can never recover.
        hard_timeout = max(
            nominal_total,
            max(0.0, self.config.center_follow_timeout_s),
        )
        if hard_timeout > 0.0 and elapsed >= hard_timeout:
            return self._abort(
                now,
                "timed_finish_slot_completion_timeout",
            )

        curve_duration = max(
            0.0,
            min(nominal_total, self.config.recovery_finish_curve_s),
        )
        fresh_geometry = (
            self._full_geometry_usable(geometry)
            and not geometry.coasted
        )
        if (
            self._timed_finish_from_curve
            and fresh_geometry
            and elapsed >= curve_duration
            and abs(geometry.heading_error_deg)
            <= max(
                0.0,
                self.config.reverse_entry_release_heading_deg,
            )
        ):
            self._timed_finish_curve_released = True
        curve_steering = (
            self._armed_entry_steering
            if self._armed_entry_steering != 0
            else self._recovery_curve_steering
        )
        steering = (
            curve_steering
            if (
                self._timed_finish_from_curve
                and curve_steering != 0
                and not self._timed_finish_curve_released
            )
            else self._straight_steering()
        )
        drive_reason = (
            "timed_finish_waiting_for_slot_completion"
            if elapsed >= nominal_total
            else reason
        )
        return self._drive(
            -abs(int(self.config.recovery_finish_reverse_speed)),
            steering,
            drive_reason,
            self._armed_reverse_path,
        )

    def _full_geometry_usable(self, geometry: ParkingGeometry) -> bool:
        return (
            geometry.found
            and geometry.has_side_pair
            and geometry.has_back_line
            and geometry.depth_remaining_px is not None
            and geometry.confidence >= self.config.geometry_confidence_min
        )

    def _slot_aligned(self, geometry: ParkingGeometry) -> bool:
        return (
            abs(geometry.heading_error_deg) <= self.config.aligned_heading_deg
            and abs(geometry.lateral_error_norm) <= self.config.aligned_lateral_norm
        )

    def _parking_correction_ready(
        self,
        geometry: ParkingGeometry,
        now: float,
    ) -> bool:
        if not self.config.correction_enabled:
            self._misaligned_frames = 0
            return False
        if (
            self.config.correction_max_attempts > 0
            and self._correction_attempts >= self.config.correction_max_attempts
        ):
            self._misaligned_frames = 0
            return False
        if self._state_elapsed(now) < max(0.0, self.config.correction_min_reverse_s):
            self._misaligned_frames = 0
            return False
        if not self._full_geometry_usable(geometry):
            self._misaligned_frames = 0
            return False
        if (
            self.config.correction_depth_trigger_px > 0.0
            and geometry.depth_remaining_px is not None
            and geometry.depth_remaining_px > self.config.correction_depth_trigger_px
        ):
            self._misaligned_frames = 0
            return False
        misaligned = (
            abs(geometry.heading_error_deg)
            >= abs(self.config.correction_heading_trigger_deg)
            or abs(geometry.lateral_error_norm)
            >= abs(self.config.correction_lateral_trigger_norm)
        )
        self._misaligned_frames = self._misaligned_frames + 1 if misaligned else 0
        return self._misaligned_frames >= max(1, self.config.correction_trigger_frames)

    def _start_parking_correction(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        self._correction_attempts += 1
        self._misaligned_frames = 0
        self._correction_reverse_steering = self._correction_steering(geometry, path)
        self._enter(ParkingState.CORRECT_FORWARD, now)
        return self._drive(
            0,
            -self._correction_reverse_steering,
            "parking_correction_forward:settling",
            path,
        )

    def _correction_forward_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        if not path.found:
            if path.status == ReversePathStatus.COLLISION_RISK:
                return self._begin_forward_realign(
                    now,
                    "correction_forward_requires_realign:%s"
                    % path.reason,
                    path,
                )
            return self._begin_reacquire_slot(
                now,
                "correction_forward_path_lost:%s" % path.reason,
                path,
            )
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        elapsed = self._state_elapsed(now)
        steering = -self._correction_reverse_steering
        if steering == 0:
            steering = -self._correction_steering(geometry, path)
        if elapsed < max(0.0, self.config.correction_steer_settle_s):
            return self._drive(0, steering, "parking_correction_forward:settling", path)
        if elapsed >= (
            max(0.0, self.config.correction_steer_settle_s)
            + max(0.0, self.config.correction_forward_s)
        ):
            self._correction_reverse_steering = self._correction_steering(geometry, path)
            self._enter(ParkingState.CORRECT_REVERSE, now)
            return self._drive(
                0,
                self._correction_reverse_steering,
                "parking_correction_reverse:settling",
                path,
            )
        return self._drive(
            abs(int(self.config.correction_forward_speed)),
            steering,
            "parking_correction_forward",
            path,
        )

    def _correction_reverse_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        if not path.found:
            if path.status == ReversePathStatus.COLLISION_RISK:
                return self._begin_forward_realign(
                    now,
                    "correction_reverse_requires_forward_realign:%s"
                    % path.reason,
                    path,
                )
            return self._begin_reacquire_slot(
                now,
                "correction_reverse_path_lost:%s" % path.reason,
                path,
            )
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        steering = self._correction_steering(geometry, path)
        self._correction_reverse_steering = steering
        if self._slot_aligned(geometry):
            self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
            return self._drive(
                self.config.reverse_center_speed,
                self._straight_steering(),
                "parking_correction_aligned:straight",
                path,
            )
        elapsed = self._state_elapsed(now)
        if elapsed < max(0.0, self.config.correction_steer_settle_s):
            return self._drive(0, steering, "parking_correction_reverse:settling", path)
        if elapsed >= (
            max(0.0, self.config.correction_steer_settle_s)
            + max(0.0, self.config.correction_reverse_s)
        ):
            self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
            return self._drive(
                self.config.reverse_entry_speed,
                self._entry_curve_steering(path),
                "parking_correction_reverse_complete:resume_entry_curve",
                path,
            )
        return self._drive(
            -abs(int(self.config.correction_reverse_speed)),
            steering,
            "parking_correction_reverse",
            path,
        )

    def _lidar_slot_box_ready(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
    ) -> bool:
        return (
            lidar.gap_confirmed
            and self._full_geometry_usable(geometry)
            and geometry.reason in ("lidar_slot_box", "lidar_slot_box_hold")
        )

    def _stop_at_back_line(
        self,
        geometry: ParkingGeometry,
        now: float,
        path: ReversePath,
    ) -> Optional[ParkingPlan]:
        if not self._full_geometry_usable(geometry):
            return self._begin_reacquire_slot(
                now,
                "reverse_waiting_for_full_geometry",
                path,
            )
        if geometry.depth_remaining_px <= self.config.stop_depth_margin_px:
            return self._finish_parking(
                now,
                "back_clearance_reached:timed_finish:%s"
                % geometry.park_completion_reason,
                path,
            )
        return None

    def _finish_parking(
        self,
        now: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        self._park_candidate_scan_timestamp = None
        self._enter(ParkingState.PARKED, now)
        return self._stop(reason, path)

    def _path_steering(
        self,
        path: ReversePath,
        minimum_abs: int = 0,
    ) -> int:
        full_scale = max(1e-9, self.path_generator.config.full_steering_curvature_per_px)
        normalized = clip(path.curvature_per_px / full_scale, -1.0, 1.0)
        raw = self.config.reverse_steering_sign * self.config.max_steering * normalized
        steering = round(raw)
        if abs(raw) > 1e-9 and minimum_abs > 0:
            minimum = min(abs(int(minimum_abs)), abs(int(self.config.max_steering)))
            if abs(steering) < minimum:
                steering = minimum if raw > 0.0 else -minimum
        return int(clip(steering, -self.config.max_steering, self.config.max_steering))

    def _entry_curve_steering(
        self,
        path: ReversePath,
    ) -> int:
        return self._path_steering(
            path,
            minimum_abs=abs(int(self.config.reverse_entry_min_steering)),
        )

    def _candidate_entry_steering(self, path: ReversePath) -> int:
        ratio = clip(abs(path.entry_steering_ratio), 0.0, 1.0)
        if ratio <= 1e-9 or abs(path.curvature_per_px) <= 1e-9:
            return self._entry_curve_steering(path)
        direction = (
            1
            if self.config.reverse_steering_sign
            * path.curvature_per_px
            >= 0.0
            else -1
        )
        maximum = abs(int(self.config.max_steering))
        minimum = min(
            abs(int(self.config.reverse_entry_min_steering)),
            maximum,
        )
        magnitude = max(minimum, round(maximum * ratio))
        return int(direction * min(maximum, magnitude))

    def _fixed_right_entry_steering(self) -> int:
        direction = 1 if self.config.reverse_steering_sign >= 0.0 else -1
        return direction * abs(int(self.config.max_steering))

    def _released_entry_steering(
        self,
        path: ReversePath,
        progress: float,
    ) -> int:
        ratio = clip(progress, 0.0, 1.0)
        entry = self._armed_entry_steering
        target = self._straight_steering()
        return int(round(entry + (target - entry) * ratio))

    def _correction_steering(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
    ) -> int:
        steering = self._path_steering(
            path,
            minimum_abs=abs(int(self.config.correction_steering)),
        )
        if steering != 0:
            return steering
        direction = self._geometry_steering_direction(geometry)
        magnitude = min(
            abs(int(self.config.correction_steering)),
            abs(int(self.config.max_steering)),
        )
        return int(direction * magnitude)

    @staticmethod
    def _geometry_steering_direction(geometry: ParkingGeometry) -> int:
        if abs(geometry.heading_error_deg) > 1.0:
            return 1 if geometry.heading_error_deg > 0.0 else -1
        if abs(geometry.lateral_error_norm) > 0.02:
            return 1 if geometry.lateral_error_norm > 0.0 else -1
        return 1

    def _prealign_steering(self) -> int:
        return int(self.config.prealign_steering)

    def _straight_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.straight_steering_trim, -limit, limit))

    def _prealign_drive(
        self,
        elapsed: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        steering = self._prealign_steering()
        if elapsed < max(0.0, self.config.prealign_steer_settle_s):
            return self._drive(
                0,
                steering,
                "steering_settle:" + reason,
                path,
            )
        return self._drive(
            self.config.prealign_speed,
            steering,
            reason,
            path,
        )

    def _provisional_prealign_drive(
        self,
        elapsed: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        steering = self._prealign_steering()
        if elapsed < max(0.0, self.config.prealign_steer_settle_s):
            return self._drive(
                0,
                steering,
                "steering_settle:" + reason,
                path,
            )
        return self._drive(
            self.config.provisional_prealign_speed,
            steering,
            reason,
            path,
        )

    def _exit_speed(self) -> int:
        return abs(int(self.config.exit_speed))

    def _exit_right_plan(
        self,
        now: float,
    ) -> ParkingPlan:
        if self._state_elapsed(now) >= max(0.0, self.config.exit_turn_s):
            if self.config.exit_straight_s <= 0.0:
                self._enter(ParkingState.EXIT_DONE, now)
                return self._stop("exit_complete")
            self._enter(ParkingState.EXIT_STRAIGHT, now)
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )
        return self._drive(
            self._exit_speed(),
            int(self.config.exit_turn_steering),
            "exit_right_turn",
        )

    def _prealign_readiness(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
    ) -> Optional[
        tuple[
            float,
            float,
            float,
            float,
            bool,
            bool,
            Optional[ReversePath],
        ]
    ]:
        metrics = self._prealign_metrics(lidar)
        if metrics is None:
            return None
        (
            slot_heading_deg,
            entry_bearing_deg,
            target_distance_mm,
            center_x_mm,
        ) = metrics
        prealign_path = (
            self.path_generator.generate(geometry)
            if self._full_geometry_usable(geometry)
            else None
        )
        direct_ready = (
            abs(slot_heading_deg)
            <= self.config.prealign_slot_heading_tolerance_deg
            and abs(entry_bearing_deg)
            <= self.config.prealign_entry_bearing_tolerance_deg
            and abs(center_x_mm)
            <= self.config.prealign_center_x_tolerance_mm
            and self.config.prealign_target_distance_min_mm
            <= target_distance_mm
            <= self.config.prealign_target_distance_max_mm
        )
        path_ready = (
            self._full_geometry_usable(geometry)
            and not lidar.coasted
            and abs(center_x_mm)
            <= self.config.prealign_curve_center_x_tolerance_mm
            and self.config.prealign_target_distance_min_mm
            <= target_distance_mm
            <= self.config.prealign_target_distance_max_mm
            and prealign_path is not None
            and prealign_path.found
            and prealign_path.status == ReversePathStatus.READY
        )
        return (
            slot_heading_deg,
            entry_bearing_deg,
            target_distance_mm,
            center_x_mm,
            direct_ready,
            path_ready,
            prealign_path,
        )

    def _curve_entry_guard_blocks(
        self,
        geometry: ParkingGeometry,
        direct_ready: bool,
        path: Optional[ReversePath],
    ) -> bool:
        """Block curve entries not validated for the command sent to the car."""

        if direct_ready or path is None:
            self._curve_entry_guard_active = False
            return False

        block_heading = max(
            0.0,
            float(self.config.prealign_curve_heading_block_deg),
        )
        release_heading = clip(
            float(self.config.prealign_curve_heading_release_deg),
            0.0,
            block_heading,
        )
        if not self._curve_path_matches_fixed_steering(path):
            self._curve_entry_guard_active = False
            return True
        heading = abs(float(geometry.heading_error_deg))

        if block_heading <= 0.0:
            self._curve_entry_guard_active = False
            return False
        if self._curve_entry_guard_active:
            if heading <= release_heading:
                self._curve_entry_guard_active = False
        elif heading > block_heading:
            self._curve_entry_guard_active = True
        return self._curve_entry_guard_active

    def _curve_entry_guard_reason(
        self,
        geometry: ParkingGeometry,
        path: Optional[ReversePath],
    ) -> str:
        ratio = (
            abs(float(path.entry_steering_ratio))
            if path is not None
            else 0.0
        )
        if (
            path is not None
            and not self._curve_path_matches_fixed_steering(path)
        ):
            return (
                "prealign_full_lock_path_required:"
                "plannedRatio=%.2f executedRatio=1.00:continuing"
            ) % ratio
        return (
            "prealign_curve_entry_guard:"
            "head=%+.1f steerRatio=%.2f:continuing"
        ) % (geometry.heading_error_deg, ratio)

    def _curve_path_matches_fixed_steering(
        self,
        path: ReversePath,
    ) -> bool:
        # Local/camera compatibility paths do not expose calibrated steering
        # candidates. Metric LiDAR full paths do, and curved reverse always
        # executes at maximum right lock.
        if path.reason != "full_arc_straight_ready":
            return True
        required_ratio = clip(
            float(self.config.prealign_curve_saturated_steering_ratio),
            0.0,
            1.0,
        )
        return (
            abs(float(path.entry_steering_ratio))
            >= required_ratio - 1e-6
        )

    def _select_reverse_entry_mode(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
    ) -> str:
        metrics = self._prealign_metrics(lidar)
        if metrics is None:
            direct = self._slot_aligned(geometry)
        else:
            (
                slot_heading_deg,
                entry_bearing_deg,
                target_distance_mm,
                center_x_mm,
            ) = metrics
            direct = (
                abs(slot_heading_deg)
                <= self.config.prealign_slot_heading_tolerance_deg
                and abs(entry_bearing_deg)
                <= self.config.prealign_entry_bearing_tolerance_deg
                and abs(center_x_mm)
                <= self.config.prealign_center_x_tolerance_mm
                and self.config.prealign_target_distance_min_mm
                <= target_distance_mm
                <= self.config.prealign_target_distance_max_mm
            )
        return "direct_aligned" if direct else "lidar_box_curve"

    @staticmethod
    def _prealign_metrics(
        lidar: LidarParkingObservation,
    ) -> Optional[tuple[float, float, float, float]]:
        values = (
            lidar.gap_center_x_right_mm,
            lidar.gap_center_y_back_mm,
            lidar.entry_target_y_back_mm,
            lidar.slot_depth_x_right,
            lidar.slot_depth_y_back,
        )
        if any(value is None for value in values):
            return None
        center_x = float(lidar.gap_center_x_right_mm)
        center_y = float(lidar.gap_center_y_back_mm)
        rear_axle_y = float(lidar.entry_target_y_back_mm)
        depth_x = float(lidar.slot_depth_x_right)
        depth_y = float(lidar.slot_depth_y_back)
        depth_length = hypot(depth_x, depth_y)
        if depth_length <= 1e-9:
            return None
        depth_x /= depth_length
        depth_y /= depth_length
        target_x = center_x
        target_y = center_y - rear_axle_y
        target_distance = hypot(target_x, target_y)
        if target_distance <= 1e-9:
            return None
        # Both angles are measured from the vehicle-rear direction (+y_back).
        # Positive values mean the target/slot is still on the vehicle-right.
        slot_heading = degrees(atan2(depth_x, depth_y))
        entry_bearing = degrees(atan2(target_x, target_y))
        return slot_heading, entry_bearing, target_distance, center_x

    def _expired(self, now: float, timeout_s: float) -> bool:
        return timeout_s > 0.0 and now - self._state_started_at >= timeout_s

    def _state_elapsed(self, now: float) -> float:
        return max(0.0, now - self._state_started_at)

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now
        self._aligned_frames = 0
        self._prealign_gap_acquired_at = None
        self._aligned_reverse_started_at = None

    def _abort(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ABORTED, now)
        return self._stop(reason)

    def _stop(self, reason: str, path: Optional[ReversePath] = None) -> ParkingPlan:
        return ParkingPlan(
            self.state,
            ControlCommand.stop(reason),
            reason,
            path,
        )

    def _drive(
        self,
        speed: int,
        steering: int,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        if speed < 0:
            self._reverse_started = True
        return ParkingPlan(
            self.state,
            ControlCommand(speed=speed, steering=steering, brake=False, reason=reason),
            reason,
            path,
        )


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
