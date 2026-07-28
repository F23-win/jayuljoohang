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
)


class ParkingState(str, Enum):
    IDLE = "idle" #시작 전 대기
    SEARCH_CARS = "search_cars" #첫번째 장애물 찾기
    TRACK_GAP = "track_gap" #첫번째 장애물을 추적하며 두번째 장애물과의 공간 찾기
    POSITION_REAR_AXLE = "position_rear_axle" #후륜을 주차 공간에 맞추는 작업 (선택적 실행)
    PREALIGN_LEFT = "prealign_left" #좌조향 실행
    VERIFY_SLOT_BOX = "verify_slot_box" #라이다로 만든 주차 공간이 안정적인지 확인
    # Backward-compatible alias for old replay integrations.
    VERIFY_PARKING_LINES = "verify_slot_box"
    PLAN_REVERSE_PATH = "plan_reverse_path" #첫 짧은 후진 목표점이 유효한지 확인
    FOLLOW_ENTRY_CURVE = "follow_entry_curve" #곡선 경로를 따라 후진
    FOLLOW_SLOT_CENTER = "follow_slot_center" #직선 경로를 따라 후진
    CORRECT_FORWARD = "correct_forward" #오차가 크면 전진
    CORRECT_REVERSE = "correct_reverse" #다시 후진
    PARKED = "parked"
    EXIT_RIGHT = "exit_right"
    EXIT_STRAIGHT = "exit_straight"
    EXIT_DONE = "exit_done"
    ABORTED = "aborted"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class ParkingPlannerConfig:
    search_speed: int = 42 #첫 차량을 찾으면서 직진하는 속도
    start_forward_s: float = 5.0 #스페이스바 시작 후 차량 검출을 무시하고 직진하는 시간 (초)
    initial_search_steering_trim: int = 0
    straight_steering_trim: int = 0
    gap_tracking_speed: int = 30 #두 차량 사이 공간을 추적할 때의 모터 속도
    position_speed: int = 22 #후륜축 중심을 공간 중심에 맞출 때 사용하는 속도
    first_car_preemptive_turn_enabled: bool = True
    first_car_approach_speed: int = 30 #첫 차량 감지 후 기준 위치까지 접근하는 속도
    detection_slow_hold_s: float = 1.5
    first_car_straight_s: float = 1.6 #첫 차량 기준 위치 도달 후 추가 직진 시간 (초)
    prealign_enabled: bool = True
    prealign_speed: int = 42 #최대 좌조향에서 전진하는 속도
    prealign_steering: int = -150 #사전 정렬 좌조향 명령
    prealign_steer_settle_s: float = 0.40 #좌조향 진행 직전 정지 상태로 바퀴만 움직이는 시간
    prealign_min_drive_s: float = 0.0 #최대 좌조향으로 실제 전진해야 하는 최소 시간
    prealign_timeout_s: float = 6.0
    prealign_curve_grace_s: float = 4.0 #timeout 이후 curve_ready 재시도를 몇 초 더 봐줄지, 넘으면 abort
    prealign_gap_acquire_timeout_s: float = 0.0 #두번째 공간을 기다리는 제한 시간
    prealign_slot_heading_tolerance_deg: float = 12.0
    prealign_entry_bearing_tolerance_deg: float = 12.0
    prealign_center_x_tolerance_mm: float = 180.0
    prealign_curve_entry_bearing_tolerance_deg: float = 45.0 #주차칸과의 각도
    prealign_curve_center_x_tolerance_mm: float = 1400.0
    prealign_target_distance_min_mm: float = 900.0
    prealign_target_distance_max_mm: float = 2600.0
    prealign_confirm_frames: int = 3 #정렬 조건을 만족해야 하는 프레임 수
    prealign_path_hold_frames: int = 3
    prealign_lidar_confirm_scans: int = 1
    prealign_yolo_min_cars: int = 1
    prealign_yolo_hold_frames: int = 3
    prealign_heading_overshoot_deg: float = 25.0
    ultrasonic_kp_steering_per_mm: float = 0.23
    ultrasonic_max_correction: int = 35
    ultrasonic_emergency_mm: float = 100.0
    ultrasonic_max_valid_mm: float = 2500.0
    ultrasonic_stale_after_s: float = 0.8
    ultrasonic_inside_max_mm: float = 600.0
    ultrasonic_inside_confirm_frames: int = 3
    ultrasonic_resume_confirm_frames: int = 3
    lidar_resume_confirm_scans: int = 3
    car_only_speed_ratio: float = 0.70
    lidar_evasion_speed: int = -24
    lidar_evasion_center_deadband_mm: float = 80.0
    reverse_entry_speed: int = -32 #곡선 후진 속도
    reverse_center_speed: int = -22 #정렬된 후 직선 후진 속도
    reverse_aligned_speed: int = -40
    aligned_reverse_min_s: float = 0.40
    aligned_reverse_max_s: float = 2.50
    reverse_entry_min_steering: int = 90 #곡선 진입 시 최소 조향
    reverse_entry_steer_settle_s: float = 0.40
    reverse_entry_release_heading_deg: float = 12.0
    reverse_entry_release_confirm_frames: int = 3
    reverse_entry_unwind_max_steering: int = 100
    entry_path_loss_grace_s: float = 1.5
    lidar_y0_band_half_width_mm: float = 150.0
    lidar_y0_clear_confirm_scans: int = 3
    lidar_y0_hold_s: float = 3.5
    lidar_y0_exit_s: float = 4.0
    # lidar_box_curve 진입에서 max 고정 조향을 heading 풀릴 때까지 유지하는 대신,
    # 매 프레임 BEV 경로 곡률로 조향을 재계산해 계속 자세제어한다. reverse_entry_min_steering
    # 바닥은 유지되므로 진입 초반엔 여전히 세게 꺾는다. False면 기존 고정 조향으로 되돌아간다.
    reverse_entry_continuous_steering: bool = True
    correction_enabled: bool = True
    correction_forward_speed: int = 22
    correction_reverse_speed: int = -28
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
    park_hold_s: float = 3.0
    exit_speed: int = 30
    exit_turn_steering: int = 80
    exit_turn_s: float = 1.6
    exit_straight_s: float = 0.0
    exit_right_min_clearance_mm: float = 180.0
    max_steering: int = 150 #후진 경로 추종 최대 조향 크기
    geometry_confidence_min: float = 0.20
    aligned_heading_deg: float = 8.0
    aligned_lateral_norm: float = 0.18
    aligned_confirm_frames: int = 4
    stop_depth_margin_px: float = 8.0
    verify_hold_s: float = 0.6
    search_timeout_s: float = 60.0
    gap_tracking_timeout_s: float = 20.0
    position_timeout_s: float = 10.0
    verify_timeout_s: float = 5.0
    path_timeout_s: float = 4.0
    path_confirm_frames: int = 3
    entry_curve_timeout_s: float = 16.0
    center_follow_timeout_s: float = 10.0


@dataclass(frozen=True)
class ParkingPlan:
    state: ParkingState
    command: ControlCommand
    reason: str
    path: Optional[ReversePath] = None
    body_mid_inside: bool = False


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
        self._waiting_for_first_car_edge = False
        self._first_car_turn_reached_at: Optional[float] = None
        self._last_search_detection_at: Optional[float] = None
        self._prealign_aligned_frames = 0
        self._prealign_last_path: Optional[ReversePath] = None
        self._prealign_path_age_frames = 0
        self._prealign_yolo_car_hold_frames = 0
        self._prealign_lidar_gate_latched = False
        self._prealign_lidar_confirm_scans = 0
        self._prealign_lidar_last_timestamp: Optional[float] = None
        self._prealign_gap_acquired_at: Optional[float] = None
        self._reverse_path_confirm_frames = 0
        self._reverse_entry_mode = "lidar_box_curve"
        self._reverse_entry_virtual_left = False
        self._reverse_entry_steering_direction = 0
        self._reverse_entry_selection_mode = ""
        self._last_entry_path: Optional[ReversePath] = None
        self._last_entry_car_only = False
        self._entry_path_loss_frames = 0
        self._entry_path_lost_at: Optional[float] = None
        self._entry_timeout_paused_at: Optional[float] = None
        self._reverse_entry_unwinding = False
        self._last_entry_steering = 0
        self._reverse_start_pending = False
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False
        self._aligned_reverse_started_at: Optional[float] = None
        self._lidar_y0_terminal_armed = False
        self._lidar_y0_both_sides_seen = False
        self._lidar_y0_clear_scans = 0
        self._lidar_y0_last_timestamp: Optional[float] = None
        self._straight_exit_after_lidar_y0_clear = False
        self._emergency_source = ""
        self._emergency_resume_state: Optional[ParkingState] = None
        self._ultrasonic_resume_frames = 0
        self._ultrasonic_clear_started_at: Optional[float] = None
        self._lidar_recovery_required = False
        self._lidar_resume_scans = 0
        self._lidar_resume_last_timestamp: Optional[float] = None

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
        self._waiting_for_first_car_edge = False
        self._first_car_turn_reached_at = None
        self._last_search_detection_at = None
        self._prealign_lidar_gate_latched = False
        self._prealign_lidar_confirm_scans = 0
        self._prealign_lidar_last_timestamp = None
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False
        self._aligned_reverse_started_at = None
        self._lidar_y0_terminal_armed = False
        self._lidar_y0_both_sides_seen = False
        self._lidar_y0_clear_scans = 0
        self._lidar_y0_last_timestamp = None
        self._straight_exit_after_lidar_y0_clear = False
        self._reverse_entry_mode = "lidar_box_curve"
        self._reverse_entry_virtual_left = False
        self._reverse_entry_steering_direction = 0
        self._reverse_entry_selection_mode = ""
        self._last_entry_path = None
        self._last_entry_car_only = False
        self._entry_path_loss_frames = 0
        self._entry_path_lost_at = None
        self._entry_timeout_paused_at = None
        self._reverse_entry_unwinding = False
        self._last_entry_steering = 0
        self._reverse_start_pending = False
        self._emergency_source = ""
        self._emergency_resume_state = None
        self._ultrasonic_resume_frames = 0
        self._ultrasonic_clear_started_at = None
        self._lidar_recovery_required = False
        self._lidar_resume_scans = 0
        self._lidar_resume_last_timestamp = None
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
        self._waiting_for_first_car_edge = False
        self._first_car_turn_reached_at = None
        self._last_search_detection_at = None
        self._prealign_aligned_frames = 0
        self._prealign_last_path = None
        self._prealign_path_age_frames = 0
        self._prealign_yolo_car_hold_frames = 0
        self._prealign_lidar_gate_latched = False
        self._prealign_lidar_confirm_scans = 0
        self._prealign_lidar_last_timestamp = None
        self._prealign_gap_acquired_at = None
        self._reverse_path_confirm_frames = 0
        self._reverse_entry_mode = "lidar_box_curve"
        self._reverse_entry_virtual_left = False
        self._reverse_entry_steering_direction = 0
        self._reverse_entry_selection_mode = ""
        self._last_entry_path = None
        self._last_entry_car_only = False
        self._entry_path_loss_frames = 0
        self._entry_path_lost_at = None
        self._entry_timeout_paused_at = None
        self._reverse_entry_unwinding = False
        self._last_entry_steering = 0
        self._reverse_start_pending = False
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False
        self._aligned_reverse_started_at = None
        self._lidar_y0_terminal_armed = False
        self._lidar_y0_both_sides_seen = False
        self._lidar_y0_clear_scans = 0
        self._lidar_y0_last_timestamp = None
        self._straight_exit_after_lidar_y0_clear = False
        self._emergency_source = ""
        self._emergency_resume_state = None
        self._ultrasonic_resume_frames = 0
        self._ultrasonic_clear_started_at = None
        self._lidar_recovery_required = False
        self._lidar_resume_scans = 0
        self._lidar_resume_last_timestamp = None

    @property
    def prealign_confirmed_frames(self) -> int:
        return self._prealign_aligned_frames

    def update(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        enabled: bool = True,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        front_left_ultrasonic_mm: Optional[float] = None,
        front_right_ultrasonic_mm: Optional[float] = None,
    ) -> ParkingPlan:
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")
        if self.state in (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.PREALIGN_LEFT,
        ):
            if (
                geometry.selection_mode == "single_car_left"
                or geometry.observed_car_count == 1
            ):
                self._reverse_entry_virtual_left = True
                self._reverse_entry_selection_mode = "single_car_left"
            elif not self._reverse_entry_selection_mode:
                if geometry.selection_mode in (
                    "two_car",
                    "two_car_left_line",
                    "two_car_right_line",
                ):
                    self._reverse_entry_selection_mode = geometry.selection_mode
                elif lidar.gap_confirmed and lidar.car_count >= 2:
                    self._reverse_entry_selection_mode = "two_car"
        self._update_body_mid_inside(left_ultrasonic_mm, right_ultrasonic_mm)
        if self.state not in (
            ParkingState.IDLE,
            ParkingState.ABORTED,
            ParkingState.EXIT_DONE,
        ):
            lidar_hold_reason = self._lidar_recovery_hold_reason(lidar)
            if lidar_hold_reason is not None:
                return self._stop(lidar_hold_reason, self._last_entry_path)
        if self.state == ParkingState.PARKED:
            hold_s = (
                self.config.lidar_y0_hold_s
                if self._straight_exit_after_lidar_y0_clear
                else self.config.park_hold_s
            )
            if self._state_elapsed(now) >= max(0.0, hold_s):
                if self._straight_exit_after_lidar_y0_clear:
                    self._enter(ParkingState.EXIT_STRAIGHT, now)
                    return self._drive(
                        self._exit_speed(),
                        self._straight_steering(),
                        "lidar_y0_clear_exit_straight",
                    )
                self._enter(ParkingState.EXIT_RIGHT, now)
                return self._exit_right_plan(now)
            return self._stop(
                "lidar_y0_clear_parked_hold"
                if self._straight_exit_after_lidar_y0_clear
                else "parked_hold"
            )
        if self.state == ParkingState.EXIT_DONE:
            return self._stop("exit_done")
        if self.state == ParkingState.ABORTED:
            return self._stop("parking_aborted")
        if self.state == ParkingState.EMERGENCY_STOP:
            if self._emergency_source != "side_ultrasonic":
                return self._stop("emergency_stop_latched")
            ultrasonic_clear = (
                self._usable_ultrasonic(left_ultrasonic_mm)
                and self._usable_ultrasonic(right_ultrasonic_mm)
                and float(left_ultrasonic_mm) > self.config.ultrasonic_emergency_mm
                and float(right_ultrasonic_mm) > self.config.ultrasonic_emergency_mm
            )
            if ultrasonic_clear:
                if self._ultrasonic_clear_started_at is None:
                    self._ultrasonic_clear_started_at = now
            else:
                self._ultrasonic_clear_started_at = None
            self._ultrasonic_resume_frames = (
                self._ultrasonic_resume_frames + 1 if ultrasonic_clear else 0
            )
            required_clear_frames = max(
                1,
                self.config.ultrasonic_resume_confirm_frames,
            )
            if self._ultrasonic_resume_frames < required_clear_frames:
                return self._stop(
                    "side_ultrasonic_resume_check:%d/%d"
                    % (self._ultrasonic_resume_frames, required_clear_frames)
                )
            recovery_path = (
                self.path_generator.generate(geometry)
                if self._full_geometry_usable(geometry) and not geometry.coasted
                else None
            )
            recovery_is_live = recovery_path is not None and recovery_path.found
            if not recovery_is_live:
                clear_started_at = self._ultrasonic_clear_started_at
                stale_path_available = (
                    self._last_entry_path is not None
                    and clear_started_at is not None
                    and now - clear_started_at
                    <= max(0.0, self.config.entry_path_loss_grace_s)
                )
                if not stale_path_available:
                    return self._stop(
                        "side_ultrasonic_cleared:waiting_for_live_path",
                        recovery_path,
                    )
                recovery_path = self._last_entry_path
            resume_state = self._emergency_resume_state
            if resume_state not in (
                ParkingState.FOLLOW_ENTRY_CURVE,
                ParkingState.FOLLOW_SLOT_CENTER,
                ParkingState.CORRECT_REVERSE,
            ):
                resume_state = ParkingState.FOLLOW_ENTRY_CURVE
            self._last_entry_path = recovery_path
            if recovery_is_live:
                self._last_entry_car_only = self._car_only_guidance(geometry)
            self._entry_path_loss_frames = 0
            self._emergency_source = ""
            self._emergency_resume_state = None
            self._ultrasonic_resume_frames = 0
            self._enter(resume_state, now)
            self._entry_path_lost_at = (
                None if recovery_is_live else self._ultrasonic_clear_started_at
            )
            self._ultrasonic_clear_started_at = None
            return self._stop(
                (
                    "side_ultrasonic_cleared:path_reacquired"
                    if recovery_is_live
                    else "side_ultrasonic_cleared:last_path_recovery"
                ),
                recovery_path,
            )

        reverse_ultrasonic_states = (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_REVERSE,
        )
        if (
            self.state in reverse_ultrasonic_states
            and (
                self._ultrasonic_emergency(left_ultrasonic_mm)
                or self._ultrasonic_emergency(right_ultrasonic_mm)
            )
        ):
            self._emergency_source = "side_ultrasonic"
            self._emergency_resume_state = self.state
            self._ultrasonic_resume_frames = 0
            self._ultrasonic_clear_started_at = None
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("side_ultrasonic_distance<=%.0fmm" % self.config.ultrasonic_emergency_mm)

        slot_and_reverse_states = (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
        )
        if lidar.unsafe and self.state in slot_and_reverse_states:
            safety_x = lidar.safety_center_x_right_mm
            deadband = max(0.0, self.config.lidar_evasion_center_deadband_mm)
            if (
                self.state in reverse_ultrasonic_states
                and safety_x is not None
                and abs(safety_x) >= deadband
            ):
                steering = (
                    abs(int(self.config.max_steering))
                    if safety_x < 0.0
                    else -abs(int(self.config.max_steering))
                )
                return self._drive(
                    -abs(int(self.config.lidar_evasion_speed)),
                    steering,
                    "lidar_evasion:%s:x=%+.0fmm"
                    % (
                        "left_to_right" if safety_x < 0.0 else "right_to_left",
                        safety_x,
                    ),
                )
            if self.state in reverse_ultrasonic_states:
                steering = self._last_entry_steering or self._fixed_right_entry_steering()
                return self._drive(
                    -abs(int(self.config.lidar_evasion_speed)),
                    steering,
                    "lidar_evasion:center_last_safe:steer=%+d" % steering,
                    self._last_entry_path,
                )
            return self._stop("lidar_safety_center_hold")

        if self.state == ParkingState.EXIT_RIGHT:
            return self._exit_right_plan(now)

        if self.state == ParkingState.EXIT_STRAIGHT:
            straight_s = (
                self.config.lidar_y0_exit_s
                if self._straight_exit_after_lidar_y0_clear
                else self.config.exit_straight_s
            )
            if (
                straight_s > 0.0
                and self._state_elapsed(now) >= straight_s
            ):
                self._enter(ParkingState.EXIT_DONE, now)
                return self._stop("exit_complete")
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )

        if self.state == ParkingState.SEARCH_CARS:
            if self._expired(now, self.config.search_timeout_s):
                return self._abort(now, "parked_car_search_timeout")
            # The launch trim compensates hardware drift only until the first
            # car has ever been seen; a later detection dropout must not revive it.
            object_detected = lidar.first_car_seen
            if object_detected:
                self._last_search_detection_at = now
            if self._state_elapsed(now) < max(0.0, self.config.start_forward_s):
                return self._drive(
                    self.config.search_speed,
                    (
                        self._initial_search_steering()
                        if self._last_search_detection_at is None
                        else self._straight_steering()
                    ),
                    "start_forward_rollout",
                )
            # Camera car masks guide the BEV slot, but only a geometrically
            # eligible right-side LiDAR cluster may change driving state.
            if not lidar.valid:
                return self._stop("waiting_for_lidar_scan")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_turn_reached
            ):
                self._right_first_car_acquired = True
                self._first_car_turn_reached_at = now
                if self.config.first_car_straight_s <= 0.0:
                    self._enter(ParkingState.PREALIGN_LEFT, now)
                    return self._drive(
                        0,
                        self._prealign_steering(),
                        "first_car_edge_aligned:settling_max_left",
                    )
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_straight_delay",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_confirmed
            ):
                self._right_first_car_acquired = True
                self._waiting_for_first_car_edge = True
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_confirmed:creeping_to_turn_point",
                )
            # 선회(preemptive) 모드가 켜져 있으면 gap이 먼저 확정돼도 legacy
            # POSITION_REAR_AXLE(후진 정렬)로 새지 않고 TRACK_GAP→PREALIGN_LEFT
            # 선회 흐름으로 흘려보낸다. first_car가 선형성/포인트 게이트를 통과
            # 못한 채 gap만 먼저 확정되던 케이스에서 후진만 하다 gap을 잃고
            # 멈추던 문제(0725_163722)를 막는다.
            preemptive_turn = (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
            )
            if lidar.gap_confirmed and not preemptive_turn:
                self._right_first_car_acquired = True
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            if lidar.gap_found or lidar.gap_confirmed or lidar.first_car_confirmed:
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
            if (
                self._last_search_detection_at is not None
                and now - self._last_search_detection_at
                <= max(0.0, self.config.detection_slow_hold_s)
            ):
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "right_lidar_detected:slow_hold",
                )
            return self._drive(
                self.config.search_speed,
                (
                    self._initial_search_steering()
                    if self._last_search_detection_at is None
                    else self._straight_steering()
                ),
                "searching_for_parked_cars",
            )

        if self.state == ParkingState.TRACK_GAP:
            if self._expired(now, self.config.gap_tracking_timeout_s):
                return self._abort(now, "two_car_gap_timeout")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
            ):
                turn_ready = lidar.first_car_turn_reached or (
                    lidar.gap_confirmed and not self._waiting_for_first_car_edge
                )
                if turn_ready and self._first_car_turn_reached_at is None:
                    self._first_car_turn_reached_at = now
                if self._first_car_turn_reached_at is None:
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_creeping_to_turn_point",
                    )
                if now - self._first_car_turn_reached_at < max(
                    0.0,
                    self.config.first_car_straight_s,
                ):
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_straight_delay",
                    )
                if (
                    lidar.first_car_turn_reached
                    or lidar.gap_confirmed
                    or lidar.first_car_confirmed
                    or lidar.first_car_seen
                    or lidar.gap_found
                    or self._right_first_car_acquired
                ):
                    self._enter(ParkingState.PREALIGN_LEFT, now)
                    if lidar.gap_confirmed:
                        self._prealign_gap_acquired_at = now
                    return self._drive(
                        0,
                        self._prealign_steering(),
                        (
                            "first_car_edge_aligned:settling_max_left"
                            if self.config.first_car_straight_s <= 0.0
                            else "first_car_straight_elapsed:settling_max_left"
                        ),
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
            fresh_lidar_cars = (
                lidar.valid
                and lidar.car_count >= 2
            )
            if lidar.timestamp != self._prealign_lidar_last_timestamp:
                self._prealign_lidar_last_timestamp = lidar.timestamp
                self._prealign_lidar_confirm_scans = (
                    self._prealign_lidar_confirm_scans + 1
                    if fresh_lidar_cars
                    else 0
                )
                self._prealign_lidar_gate_latched = (
                    self._prealign_lidar_confirm_scans
                    >= max(1, self.config.prealign_lidar_confirm_scans)
                )
            prealign_path = (
                self.path_generator.generate(geometry)
                if self._full_geometry_usable(geometry)
                else None
            )
            current_path_ready = (
                self._camera_guidance_usable(geometry)
                and not geometry.coasted
                and prealign_path is not None
                and prealign_path.found
            )
            if current_path_ready:
                self._prealign_last_path = prealign_path
                self._prealign_path_age_frames = 0
            else:
                self._prealign_path_age_frames += 1
            held_path = (
                self._prealign_last_path
                if self._prealign_path_age_frames
                <= max(0, self.config.prealign_path_hold_frames)
                else None
            )
            if geometry.observed_car_count >= max(
                1,
                self.config.prealign_yolo_min_cars,
            ):
                self._prealign_yolo_car_hold_frames = max(
                    1,
                    self.config.prealign_yolo_hold_frames,
                )
            else:
                self._prealign_yolo_car_hold_frames = max(
                    0,
                    self._prealign_yolo_car_hold_frames - 1,
                )
            yolo_ready = self._prealign_yolo_car_hold_frames > 0
            minimum_drive_end = (
                max(0.0, self.config.prealign_steer_settle_s)
                + max(0.0, self.config.prealign_min_drive_s)
            )
            if elapsed < minimum_drive_end:
                return self._prealign_drive(
                    elapsed,
                    "prealign_min_drive:%.2f/%.2f"
                    % (
                        max(0.0, elapsed - self.config.prealign_steer_settle_s),
                        max(0.0, self.config.prealign_min_drive_s),
                    ),
                    held_path,
                )
            ready = (
                self._prealign_lidar_gate_latched
                and yolo_ready
                and current_path_ready
                and bool(self._reverse_entry_selection_mode)
            )
            self._prealign_aligned_frames = (
                self._prealign_aligned_frames + 1 if ready else 0
            )
            if self._prealign_aligned_frames >= max(
                1,
                self.config.prealign_confirm_frames,
            ):
                self._reverse_entry_mode = "lidar_box_curve"
                initial_steering = self._fixed_right_entry_steering() if (
                    self._reverse_entry_virtual_left
                    or not self.config.reverse_entry_continuous_steering
                ) else (
                    self._entry_curve_steering(held_path)
                    if held_path is not None
                    else 0
                )
                self._reverse_entry_steering_direction = (
                    1 if initial_steering > 0 else -1 if initial_steering < 0 else 0
                )
                self._last_entry_steering = initial_steering
                self._lidar_y0_terminal_armed = True
                self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
                self._last_entry_path = prealign_path
                self._last_entry_car_only = self._car_only_guidance(geometry)
                self._entry_path_loss_frames = 0
                self._entry_path_lost_at = None
                self._reverse_start_pending = True
                return self._stop(
                    "prealign_sensor_crosscheck_ready:reverse_path_armed",
                    prealign_path,
                )
            if ready:
                reason = "prealign_bev_path_confirming:%d/%d" % (
                    self._prealign_aligned_frames,
                    max(1, self.config.prealign_confirm_frames),
                )
            elif not self._prealign_lidar_gate_latched:
                reason = "prealign_waiting_for_lidar_cars:%d/2 scans=%d/%d" % (
                    lidar.car_count,
                    self._prealign_lidar_confirm_scans,
                    max(1, self.config.prealign_lidar_confirm_scans),
                )
            elif not yolo_ready:
                reason = "prealign_waiting_for_yolo_cars:%d/%d" % (
                    geometry.observed_car_count,
                    max(1, self.config.prealign_yolo_min_cars),
                )
            elif not current_path_ready:
                reason = "prealign_waiting_for_current_bev_path"
            elif not self._reverse_entry_selection_mode:
                reason = "prealign_waiting_for_target_slot"
            else:
                reason = "prealign_sensor_crosscheck_pending"
            return self._prealign_drive(
                elapsed,
                reason,
                held_path,
            )

        if self.state == ParkingState.VERIFY_SLOT_BOX:
            if self._expired(now, self.config.verify_timeout_s):
                return self._abort(now, "lidar_slot_box_verify_timeout")
            if not self._full_geometry_usable(geometry):
                return self._stop("waiting_for_lidar_slot_box")
            if now - self._state_started_at < self.config.verify_hold_s:
                return self._stop("lidar_slot_box_verify_hold")
            self._enter(ParkingState.PLAN_REVERSE_PATH, now)
            return self._stop("lidar_slot_box_verified")

        path = self.path_generator.generate(geometry)

        lidar_y0_plan = self._lidar_y0_terminal_plan(geometry, lidar, now, path)
        if lidar_y0_plan is not None:
            return lidar_y0_plan

        if self.state == ParkingState.PLAN_REVERSE_PATH:
            if self._expired(now, self.config.path_timeout_s):
                return self._abort(now, "reverse_path_timeout:%s" % path.reason)
            if not path.found:
                self._reverse_path_confirm_frames = max(
                    0,
                    self._reverse_path_confirm_frames - 1,
                )
                return self._stop(
                    "waiting_for_reverse_path:%s confirm=%d/%d"
                    % (
                        path.reason,
                        self._reverse_path_confirm_frames,
                        max(1, self.config.path_confirm_frames),
                    ),
                    path,
                )
            self._reverse_path_confirm_frames += 1
            if self._reverse_path_confirm_frames < max(
                1,
                self.config.path_confirm_frames,
            ):
                return self._stop(
                    "reverse_path_confirming:%d/%d"
                    % (
                        self._reverse_path_confirm_frames,
                        max(1, self.config.path_confirm_frames),
                    ),
                    path,
                )
            if self._reverse_entry_steering_direction == 0:
                initial_steering = (
                    self._fixed_right_entry_steering()
                    if (
                        self._reverse_entry_virtual_left
                        or not self.config.reverse_entry_continuous_steering
                    )
                    else self._entry_curve_steering(path)
                )
                self._reverse_entry_steering_direction = (
                    1 if initial_steering > 0 else -1 if initial_steering < 0 else 0
                )
                self._last_entry_steering = initial_steering
            self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
            self._last_entry_path = path
            self._last_entry_car_only = self._car_only_guidance(geometry)
            self._entry_path_loss_frames = 0
            self._entry_path_lost_at = None
            return self._stop("reverse_path_armed", path)

        if self.state == ParkingState.CORRECT_FORWARD:
            return self._correction_forward_plan(geometry, path, now)

        if self.state == ParkingState.CORRECT_REVERSE:
            return self._correction_reverse_plan(geometry, path, now)

        if self.state == ParkingState.FOLLOW_ENTRY_CURVE:
            path_coasting = False
            car_only_guidance = self._car_only_guidance(geometry)
            guidance_live = (
                path.found
                and self._full_geometry_usable(geometry)
                and not geometry.coasted
            )
            if self._reverse_start_pending:
                if not guidance_live:
                    self._pause_path_timeout(now)
                    return self._stop(
                        "reverse_start_waiting_for_live_path",
                        self._last_entry_path,
                    )
                self._reverse_start_pending = False
            if guidance_live:
                self._resume_path_timeout(now)
                self._last_entry_path = path
                self._last_entry_car_only = car_only_guidance
                self._entry_path_loss_frames = 0
                self._entry_path_lost_at = None
            else:
                self._aligned_frames = 0
                self._entry_path_loss_frames += 1
                if (
                    self._last_entry_path is not None
                    and self._entry_path_loss_elapsed(now)
                    <= max(0.0, self.config.entry_path_loss_grace_s)
                ):
                    path = self._last_entry_path
                    path_coasting = True
                    car_only_guidance = self._last_entry_car_only
                else:
                    self._entry_heading_ready_frames = 0
                    reason = geometry.reason if path.found else path.reason
                    self._pause_path_timeout(now)
                    return self._stop("entry_curve_path_lost:%s" % reason, path)
            if (
                not self._lidar_y0_terminal_armed
                and self._expired(now, self.config.entry_curve_timeout_s)
            ):
                return self._abort(now, "entry_curve_timeout")
            entry_speed = self._reverse_speed(
                self.config.reverse_entry_speed,
                car_only_guidance,
            )
            stop = (
                None
                if path_coasting or car_only_guidance or self._lidar_y0_terminal_armed
                else self._stop_at_back_line(geometry, now, path)
            )
            if stop is not None:
                return stop
            if self._reverse_entry_mode == "lidar_box_curve":
                elapsed = self._state_elapsed(now)
                # 기존: heading 풀릴 때까지 max 고정 조향 유지. 개선: 매 프레임 경로
                # 곡률로 재계산해 계속 자세제어(진입 초반 min_steering 바닥은 유지).
                virtual_left_entry = self._reverse_entry_virtual_left
                required_frames = max(
                    1,
                    self.config.reverse_entry_release_confirm_frames,
                )
                if elapsed < max(0.0, self.config.reverse_entry_steer_settle_s):
                    steering = (
                        self._fixed_right_entry_steering()
                        if virtual_left_entry
                        or not self.config.reverse_entry_continuous_steering
                        else self._entry_curve_steering(
                            path,
                            left_ultrasonic_mm,
                            right_ultrasonic_mm,
                        )
                    )
                    if self._reverse_entry_steering_direction:
                        steering = self._reverse_entry_steering_direction * abs(steering)
                    self._last_entry_steering = steering
                    self._entry_heading_ready_frames = 0
                    return self._drive(
                        0,
                        steering,
                        "reverse_entry_bev_path:settling",
                        path,
                    )

                if virtual_left_entry and not self._reverse_entry_unwinding:
                    release_ready = guidance_live and not path_coasting
                    self._entry_heading_ready_frames = (
                        self._entry_heading_ready_frames + 1
                        if release_ready
                        else 0
                    )
                    if self._entry_heading_ready_frames < required_frames:
                        steering = self._fixed_right_entry_steering()
                        self._last_entry_steering = steering
                        return self._drive(
                            entry_speed,
                            steering,
                            (
                                "following_entry_virtual_left_max_right:"
                                "heading=%+.1f steer=%+d confirm=%d/%d"
                            )
                            % (
                                geometry.heading_error_deg,
                                steering,
                                self._entry_heading_ready_frames,
                                required_frames,
                            ),
                            path,
                        )
                    self._reverse_entry_unwinding = True
                    self._entry_heading_ready_frames = 0

                if path_coasting:
                    steering = self._last_entry_steering
                elif virtual_left_entry and self._reverse_entry_unwinding:
                    steering = self._path_steering(
                        path,
                        left_ultrasonic_mm,
                        right_ultrasonic_mm,
                    )
                    unwind_limit = min(
                        abs(int(self.config.max_steering)),
                        abs(int(self.config.reverse_entry_unwind_max_steering)),
                    )
                    steering = int(clip(steering, -unwind_limit, unwind_limit))
                    real_three_line = (
                        geometry.selection_mode == "three_line_after_car"
                        and not geometry.coasted
                        and all(
                            line is not None and line.mask_index >= 0
                            for line in (geometry.left, geometry.right, geometry.back)
                        )
                    )
                    if steering < 0 and not real_three_line:
                        steering = 0
                elif self.config.reverse_entry_continuous_steering:
                    steering = self._entry_curve_steering(
                        path,
                        left_ultrasonic_mm,
                        right_ultrasonic_mm,
                    )
                    if self._reverse_entry_steering_direction:
                        steering = self._reverse_entry_steering_direction * abs(steering)
                else:
                    steering = self._fixed_right_entry_steering()
                if not path_coasting:
                    self._last_entry_steering = steering

                heading_ready = (
                    not path_coasting
                    and not car_only_guidance
                    and self._slot_aligned(geometry)
                )
                self._entry_heading_ready_frames = (
                    self._entry_heading_ready_frames + 1
                    if heading_ready
                    else 0
                )
                if self._entry_heading_ready_frames >= required_frames:
                    center_steering = self._path_steering(
                        path,
                        left_ultrasonic_mm,
                        right_ultrasonic_mm,
                    )
                    self._last_entry_steering = center_steering
                    self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                    return self._drive(
                        self._reverse_speed(
                            self.config.reverse_center_speed,
                            car_only_guidance,
                        ),
                        center_steering,
                        "following_slot_center:entry_aligned_released",
                        path,
                    )
                phase = (
                    "virtual_left_path_unwind"
                    if virtual_left_entry and self._reverse_entry_unwinding
                    else "curve"
                    if self.config.reverse_entry_continuous_steering
                    else "fixed_max_right"
                )
                if path_coasting:
                    phase += "_coast:%.2f/%.2fs" % (
                        self._entry_path_loss_elapsed(now),
                        max(0.0, self.config.entry_path_loss_grace_s),
                    )
                return self._drive(
                    entry_speed,
                    steering,
                    (
                        "following_entry_%s:heading=%+.1f steer=%+d confirm=%d/%d"
                    )
                    % (
                        phase,
                        geometry.heading_error_deg,
                        steering,
                        self._entry_heading_ready_frames,
                        required_frames,
                    ),
                    path,
                )
            aligned = not car_only_guidance and self._slot_aligned(geometry)
            self._aligned_frames = self._aligned_frames + 1 if aligned else 0
            if self._aligned_frames >= max(1, self.config.aligned_confirm_frames):
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._drive(
                    self._reverse_speed(
                        self.config.reverse_center_speed,
                        car_only_guidance,
                    ),
                    self._path_steering(
                        path,
                        left_ultrasonic_mm,
                        right_ultrasonic_mm,
                    ),
                    "following_slot_center:local_target",
                    path,
                )
            return self._drive(
                entry_speed,
                self._entry_curve_steering(
                    path,
                    left_ultrasonic_mm,
                    right_ultrasonic_mm,
                ),
                "following_entry_curve:%s" % self._reverse_entry_mode,
                path,
            )

        if self.state == ParkingState.FOLLOW_SLOT_CENTER:
            car_only_guidance = self._car_only_guidance(geometry)
            guidance_live = (
                path.found
                and self._full_geometry_usable(geometry)
                and not geometry.coasted
            )
            if guidance_live:
                self._resume_path_timeout(now)
                stop = (
                    None
                    if car_only_guidance or self._lidar_y0_terminal_armed
                    else self._stop_at_back_line(geometry, now, path)
                )
                if stop is not None:
                    return stop
                self._last_entry_path = path
                self._last_entry_car_only = car_only_guidance
                self._entry_path_loss_frames = 0
                self._entry_path_lost_at = None
                center_steering = self._path_steering(
                    path,
                    left_ultrasonic_mm,
                    right_ultrasonic_mm,
                )
                self._last_entry_steering = center_steering
            else:
                self._entry_path_loss_frames += 1
                if (
                    self._last_entry_path is None
                    or self._entry_path_loss_elapsed(now)
                    > max(0.0, self.config.entry_path_loss_grace_s)
                ):
                    self._pause_path_timeout(now)
                    return self._stop("slot_center_path_lost:%s" % path.reason, path)
                path = self._last_entry_path
                car_only_guidance = self._last_entry_car_only
                center_steering = self._last_entry_steering
                return self._drive(
                    self._reverse_speed(
                        self.config.reverse_center_speed,
                        car_only_guidance,
                    ),
                    center_steering,
                    "following_slot_center_path_coast:%.2f/%.2fs"
                    % (
                        self._entry_path_loss_elapsed(now),
                        max(0.0, self.config.entry_path_loss_grace_s),
                    ),
                    path,
                )
            if self._aligned_reverse_started_at is None:
                fresh_pair_aligned = (
                    not car_only_guidance
                    and not self._lidar_y0_terminal_armed
                    and lidar.gap_pair_observed
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
                aligned_elapsed = max(
                    0.0,
                    now - self._aligned_reverse_started_at,
                )
                minimum_elapsed = max(
                    0.0,
                    self.config.aligned_reverse_min_s,
                )
                if (
                    self._body_mid_inside
                    and aligned_elapsed >= minimum_elapsed
                ):
                    return self._finish_parking(
                        now,
                        "body_mid_inside_and_aligned",
                        path,
                    )
                configured_maximum = self.config.aligned_reverse_max_s
                maximum_elapsed = max(minimum_elapsed, configured_maximum)
                if configured_maximum > 0.0 and aligned_elapsed >= maximum_elapsed:
                    return self._finish_parking(
                        now,
                        "aligned_straight_reverse_complete",
                        path,
                    )
                self._last_entry_steering = self._straight_steering()
                return self._drive(
                    self._reverse_speed(
                        self.config.reverse_aligned_speed,
                        car_only_guidance,
                    ),
                    self._last_entry_steering,
                    "following_slot_center:aligned_straight",
                    path,
                )
            if (
                not self._lidar_y0_terminal_armed
                and self._expired(now, self.config.center_follow_timeout_s)
            ):
                return self._abort(now, "slot_center_follow_timeout")
            if (
                not car_only_guidance
                and not self._lidar_y0_terminal_armed
                and self._parking_correction_ready(geometry, now)
            ):
                return self._start_parking_correction(geometry, path, now)
            return self._drive(
                self._reverse_speed(
                    self.config.reverse_center_speed,
                    car_only_guidance,
                ),
                center_steering,
                "following_slot_center:local_target",
                path,
            )

        return self._abort(now, "unknown_state")

    def _full_geometry_usable(self, geometry: ParkingGeometry) -> bool:
        return (
            geometry.found
            and geometry.has_side_pair
            and geometry.has_back_line
            and geometry.depth_remaining_px is not None
            and geometry.confidence >= self.config.geometry_confidence_min
        )

    @staticmethod
    def _car_only_guidance(geometry: ParkingGeometry) -> bool:
        return geometry.selection_mode in ("car_only_left", "car_only_right")

    def _lidar_y0_terminal_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        path: ReversePath,
    ) -> Optional[ParkingPlan]:
        if self.state not in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            return None
        if not self._lidar_y0_terminal_armed:
            return None

        band = max(0.0, self.config.lidar_y0_band_half_width_mm)
        deadband = max(0.0, self.config.lidar_evasion_center_deadband_mm)
        at_y0 = tuple(
            cluster
            for cluster in lidar.side_car_clusters
            if cluster.y_back_min_mm <= band
            and cluster.y_back_max_mm >= -band
        )
        left_seen = any(
            cluster.center_x_right_mm <= -deadband for cluster in at_y0
        )
        right_seen = any(
            cluster.center_x_right_mm >= deadband for cluster in at_y0
        )
        if left_seen and right_seen:
            self._lidar_y0_both_sides_seen = True

        if lidar.timestamp == self._lidar_y0_last_timestamp:
            return None
        self._lidar_y0_last_timestamp = lidar.timestamp
        if not self._lidar_y0_both_sides_seen:
            return None

        self._lidar_y0_clear_scans = (
            self._lidar_y0_clear_scans + 1
            if not left_seen or not right_seen
            else 0
        )
        if self._lidar_y0_clear_scans < max(
            1,
            self.config.lidar_y0_clear_confirm_scans,
        ):
            return None

        self._straight_exit_after_lidar_y0_clear = True
        self._enter(ParkingState.PARKED, now)
        return self._stop("lidar_y0_one_side_cleared", path)

    def _reverse_speed(self, speed: int, car_only_guidance: bool) -> int:
        if not car_only_guidance:
            return int(speed)
        ratio = clip(self.config.car_only_speed_ratio, 0.0, 1.0)
        return int(round(float(speed) * ratio))

    @staticmethod
    def _camera_guidance_usable(geometry: ParkingGeometry) -> bool:
        # Heading error is corrected while reversing along the generated path;
        # requiring near-alignment here can prevent reverse entry from starting.
        return geometry.found and geometry.has_side_pair

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
            return self._stop("correction_forward_path_lost:%s" % path.reason, path)
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
            return self._stop("correction_reverse_path_lost:%s" % path.reason, path)
        car_only_guidance = self._car_only_guidance(geometry)
        stop = (
            None
            if car_only_guidance
            else self._stop_at_back_line(geometry, now, path)
        )
        if stop is not None:
            return stop
        steering = self._correction_steering(geometry, path)
        self._correction_reverse_steering = steering
        if not car_only_guidance and self._slot_aligned(geometry):
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
                self._reverse_speed(
                    self.config.reverse_entry_speed,
                    car_only_guidance,
                ),
                self._entry_curve_steering(path),
                "parking_correction_reverse_complete:resume_entry_curve",
                path,
            )
        return self._drive(
            self._reverse_speed(
                -abs(int(self.config.correction_reverse_speed)),
                car_only_guidance,
            ),
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
            return self._stop("reverse_waiting_for_full_geometry", path)
        if geometry.depth_remaining_px <= self.config.stop_depth_margin_px:
            if not geometry.vehicle_fully_inside:
                return self._stop("back_clearance_reached_vehicle_not_fully_inside", path)
            if not self._slot_aligned(geometry):
                return self._stop("back_clearance_reached_vehicle_not_aligned", path)
            return self._finish_parking(
                now,
                "vehicle_fully_inside_and_aligned",
                path,
            )
        return None

    def _finish_parking(
        self,
        now: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        self._enter(ParkingState.PARKED, now)
        return self._stop(reason, path)

    def _path_steering(
        self,
        path: ReversePath,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        minimum_abs: int = 0,
    ) -> int:
        full_scale = max(1e-9, self.path_generator.config.full_steering_curvature_per_px)
        normalized = clip(path.curvature_per_px / full_scale, -1.0, 1.0)
        # The raw rear-camera BEV x-axis is opposite the vehicle's physical x-axis.
        # Firmware steering is fixed: positive=right, negative=left.
        raw = -self.config.max_steering * normalized
        steering = round(raw)
        if abs(raw) > 1e-9 and minimum_abs > 0:
            minimum = min(abs(int(minimum_abs)), abs(int(self.config.max_steering)))
            if abs(steering) < minimum:
                steering = minimum if raw > 0.0 else -minimum
        steering += self._ultrasonic_correction(
            left_ultrasonic_mm,
            right_ultrasonic_mm,
        )
        return int(clip(steering, -self.config.max_steering, self.config.max_steering))

    def _entry_curve_steering(
        self,
        path: ReversePath,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
    ) -> int:
        return self._path_steering(
            path,
            left_ultrasonic_mm,
            right_ultrasonic_mm,
            minimum_abs=abs(int(self.config.reverse_entry_min_steering)),
        )

    def _fixed_right_entry_steering(self) -> int:
        return abs(int(self.config.max_steering))

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

    def _ultrasonic_correction(
        self,
        left_mm: Optional[float],
        right_mm: Optional[float],
    ) -> int:
        if not self._usable_ultrasonic(left_mm) or not self._usable_ultrasonic(right_mm):
            return 0
        correction = self.config.ultrasonic_kp_steering_per_mm * (
            float(right_mm) - float(left_mm)
        )
        limit = abs(self.config.ultrasonic_max_correction)
        return int(round(clip(correction, -limit, limit)))

    def _ultrasonic_emergency(self, value_mm: Optional[float]) -> bool:
        return (
            self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.ultrasonic_emergency_mm
        )

    def _update_body_mid_inside(
        self,
        left_mm: Optional[float],
        right_mm: Optional[float],
    ) -> None:
        if self._body_mid_inside:
            return
        if self.state not in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_REVERSE,
        ):
            self._body_mid_inside_frames = 0
            return
        threshold = max(0.0, self.config.ultrasonic_inside_max_mm)
        detected = (
            threshold > 0.0
            and self._usable_ultrasonic(left_mm)
            and self._usable_ultrasonic(right_mm)
            and float(left_mm) <= threshold
            and float(right_mm) <= threshold
        )
        self._body_mid_inside_frames = (
            self._body_mid_inside_frames + 1 if detected else 0
        )
        if self._body_mid_inside_frames >= max(
            1,
            self.config.ultrasonic_inside_confirm_frames,
        ):
            self._body_mid_inside = True

    def _usable_ultrasonic(self, value_mm: Optional[float]) -> bool:
        return (
            value_mm is not None
            and 0.0 < float(value_mm) <= self.config.ultrasonic_max_valid_mm
        )

    def _prealign_steering(self) -> int:
        return int(self.config.prealign_steering)

    def _straight_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.straight_steering_trim, -limit, limit))

    def _initial_search_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.initial_search_steering_trim, -limit, limit))

    def _prealign_drive(
        self,
        elapsed: float,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        steering = self._prealign_steering()
        if elapsed < max(0.0, self.config.prealign_steer_settle_s):
            return self._drive(0, steering, "steering_settle:" + reason, path)
        return self._drive(self.config.prealign_speed, steering, reason, path)

    def _exit_speed(self) -> int:
        return abs(int(self.config.exit_speed))

    def _exit_right_plan(
        self,
        now: float,
    ) -> ParkingPlan:
        if self._state_elapsed(now) >= max(0.0, self.config.exit_turn_s):
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

    def _lidar_recovery_hold_reason(
        self,
        lidar: LidarParkingObservation,
    ) -> Optional[str]:
        if not lidar.valid:
            self._lidar_recovery_required = True
            self._lidar_resume_scans = 0
            self._lidar_resume_last_timestamp = None
            if self.state == ParkingState.PREALIGN_LEFT:
                self._prealign_lidar_gate_latched = False
                self._prealign_lidar_confirm_scans = 0
                self._prealign_lidar_last_timestamp = None
            return "lidar_unavailable:%s" % lidar.reason

        if not self._lidar_recovery_required:
            return None

        if lidar.timestamp != self._lidar_resume_last_timestamp:
            self._lidar_resume_last_timestamp = lidar.timestamp
            self._lidar_resume_scans += 1
        required_scans = max(1, self.config.lidar_resume_confirm_scans)
        if self._lidar_resume_scans < required_scans:
            return "lidar_resume_check:%d/%d" % (
                self._lidar_resume_scans,
                required_scans,
            )

        self._lidar_recovery_required = False
        self._lidar_resume_scans = 0
        self._lidar_resume_last_timestamp = None
        return None

    def _expired(self, now: float, timeout_s: float) -> bool:
        return timeout_s > 0.0 and now - self._state_started_at >= timeout_s

    def _state_elapsed(self, now: float) -> float:
        return max(0.0, now - self._state_started_at)

    def _entry_path_loss_elapsed(self, now: float) -> float:
        if self._entry_path_lost_at is None:
            self._entry_path_lost_at = now
        return max(0.0, now - self._entry_path_lost_at)

    def _pause_path_timeout(self, now: float) -> None:
        if self._entry_timeout_paused_at is None:
            self._entry_timeout_paused_at = now

    def _resume_path_timeout(self, now: float) -> None:
        if self._entry_timeout_paused_at is None:
            return
        self._state_started_at += max(0.0, now - self._entry_timeout_paused_at)
        self._entry_timeout_paused_at = None

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now
        self._aligned_frames = 0
        self._prealign_aligned_frames = 0
        self._prealign_last_path = None
        self._prealign_path_age_frames = 0
        self._prealign_yolo_car_hold_frames = 0
        self._prealign_gap_acquired_at = None
        self._reverse_path_confirm_frames = 0
        self._entry_heading_ready_frames = 0
        self._entry_path_lost_at = None
        self._entry_timeout_paused_at = None
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
            self._body_mid_inside,
        )

    def _drive(
        self,
        speed: int,
        steering: int,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        return ParkingPlan(
            self.state,
            ControlCommand(speed=speed, steering=steering, brake=False, reason=reason),
            reason,
            path,
            self._body_mid_inside,
        )


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
