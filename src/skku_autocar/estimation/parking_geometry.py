from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, hypot
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParkingLine:
    """A straight rear-BEV line fitted from a mask or projected from LiDAR."""

    center_x: float
    center_y: float
    direction_x: float
    direction_y: float
    length_px: float
    residual_px: float
    quality: float
    point_count: int
    mask_index: int = -1

    @property
    def angle_deg(self) -> float:
        angle = degrees(atan2(self.direction_y, self.direction_x)) % 180.0
        return angle

    def point_at_y(self, y: float) -> Optional[Tuple[float, float]]:
        if abs(self.direction_y) < 1e-6:
            return None
        scale = (y - self.center_y) / self.direction_y
        return (
            self.center_x + scale * self.direction_x,
            y,
        )

    def supports(self, point: Tuple[float, float], margin_px: float) -> bool:
        dx = point[0] - self.center_x
        dy = point[1] - self.center_y
        along = abs(dx * self.direction_x + dy * self.direction_y)
        return along <= self.length_px / 2.0 + margin_px


@dataclass(frozen=True)
class ParkingGeometry:
    found: bool = False
    has_side_pair: bool = False
    has_back_line: bool = False
    left: Optional[ParkingLine] = None
    right: Optional[ParkingLine] = None
    back: Optional[ParkingLine] = None
    lateral_error_px: float = 0.0
    lateral_error_norm: float = 0.0
    heading_error_deg: float = 0.0
    depth_to_back_px: Optional[float] = None
    depth_remaining_px: Optional[float] = None
    slot_width_px: Optional[float] = None
    vehicle_x_px: float = 0.0
    vehicle_y_px: float = 0.0
    slot_center_x_px: Optional[float] = None
    slot_center_y_px: Optional[float] = None
    slot_direction_x: float = 0.0
    slot_direction_y: float = -1.0
    back_center_x_px: Optional[float] = None
    back_center_y_px: Optional[float] = None
    stop_target_x_px: Optional[float] = None
    stop_target_y_px: Optional[float] = None
    vehicle_inside_ratio: float = 0.0
    vehicle_fully_inside: bool = False
    confidence: float = 0.0
    observed_line_count: int = 0
    observed_car_count: int = 0
    coasted: bool = False
    selection_mode: str = "line_only"
    reason: str = "not_found"


@dataclass(frozen=True)
class ParkingGeometryConfig:
    min_line_pixels: int = 30
    max_fit_points: int = 5000
    max_line_residual_px: float = 10.0
    parallel_tolerance_deg: float = 15.0
    perpendicular_tolerance_deg: float = 18.0
    slot_width_min_px: float = 60.0
    slot_width_max_px: float = 360.0
    expected_slot_width_px: float = 180.0
    intersection_margin_px: float = 45.0
    vehicle_center_x_ratio: float = 0.50
    vehicle_reference_y_ratio: float = 0.95
    control_target_y_ratio: float = 0.70
    desired_back_clearance_px: float = 50.0
    virtual_back_y_ratio: float = 0.32
    min_geometry_confidence: float = 0.20
    min_confirm_frames: int = 3
    jump_reconfirm_frames: int = 5
    max_jump_hold_frames: int = 3
    max_coast_frames: int = 3
    smooth_alpha: float = 0.35
    max_lateral_jump_px: float = 90.0
    max_heading_jump_deg: float = 35.0
    max_depth_jump_px: float = 140.0
    merged_depth_reconfirm_frames: int = 5
    merged_depth_candidate_tolerance_px: float = 45.0


class ParkingGeometryDepthStabilizer:
    """Reconfirm large final back-line jumps after camera/LiDAR fusion."""

    def __init__(
        self,
        max_jump_px: float = 140.0,
        reconfirm_frames: int = 5,
        candidate_tolerance_px: float = 45.0,
    ):
        self.max_jump_px = max(0.0, float(max_jump_px))
        self.reconfirm_frames = max(1, int(reconfirm_frames))
        self.candidate_tolerance_px = max(0.0, float(candidate_tolerance_px))
        self._last: Optional[ParkingGeometry] = None
        self._candidate_depth: Optional[float] = None
        self._candidate_frames = 0

    def reset(self) -> None:
        self._last = None
        self._candidate_depth = None
        self._candidate_frames = 0

    def update(self, geometry: ParkingGeometry) -> ParkingGeometry:
        depth = geometry.depth_remaining_px
        if not geometry.has_back_line or depth is None:
            self._candidate_depth = None
            self._candidate_frames = 0
            return geometry
        if self._last is None or self._last.depth_remaining_px is None:
            self._accept(geometry)
            return geometry
        previous_depth = self._last.depth_remaining_px
        if abs(depth - previous_depth) <= self.max_jump_px:
            self._accept(geometry)
            return geometry
        if (
            self._candidate_depth is None
            or abs(depth - self._candidate_depth) > self.candidate_tolerance_px
        ):
            self._candidate_depth = depth
            self._candidate_frames = 1
        else:
            self._candidate_depth = depth
            self._candidate_frames += 1
        if self._candidate_frames >= self.reconfirm_frames:
            self._accept(geometry)
            return geometry
        previous = self._last
        # Side direction and back target must come from one accepted frame.
        # Mixing a new side pose with the held target can put it ahead of the car.
        return replace(
            previous,
            observed_car_count=geometry.observed_car_count,
            observed_line_count=geometry.observed_line_count,
            coasted=True,
            reason="%s|depth_reconfirming:%d/%d" % (
                geometry.reason,
                self._candidate_frames,
                self.reconfirm_frames,
            ),
        )

    def _accept(self, geometry: ParkingGeometry) -> None:
        self._last = geometry
        self._candidate_depth = None
        self._candidate_frames = 0


class ParkingGeometryEstimator:
    """Resolve separate ``line`` masks into the topology of a parking bay.

    Coordinates are rear-camera BEV pixels: x grows to vehicle-right, y grows
    toward the vehicle.  The desired reverse direction is therefore negative y.
    Angles are compared only after the ground-plane BEV transform.
    """

    def __init__(self, config: ParkingGeometryConfig = ParkingGeometryConfig()):
        self.config = config
        self._candidate: Optional[ParkingGeometry] = None
        self._confirm_frames = 0
        self._last: Optional[ParkingGeometry] = None
        self._coast_frames = 0
        self._reconfirming = False
        self._jump_hold_frames = 0
        self._seen_two_cars = False
        self._two_car_confirm_frames = 0

    def reset(self) -> None:
        self._candidate = None
        self._confirm_frames = 0
        self._last = None
        self._coast_frames = 0
        self._reconfirming = False
        self._jump_hold_frames = 0
        self._seen_two_cars = False
        self._two_car_confirm_frames = 0

    def select_masks(
        self,
        line_masks: Sequence[Any],
        car_masks: Sequence[Any],
    ) -> Tuple[Tuple[Any, ...], str]:
        selected, mode = select_parking_line_masks(
            line_masks,
            car_masks,
            after_two_cars=self._seen_two_cars,
        )
        self._two_car_confirm_frames = (
            self._two_car_confirm_frames + 1 if mode == "two_car" else 0
        )
        if self._two_car_confirm_frames >= max(1, self.config.min_confirm_frames):
            self._seen_two_cars = True

        locked_two_car_gap = (
            self._seen_two_cars
            and len(car_masks) < 2
            and self._last is not None
            and self._last.selection_mode in (
                "two_car",
                "two_car_left_line",
                "two_car_right_line",
            )
        )
        if locked_two_car_gap and self._coast_frames < self.config.max_coast_frames:
            # Do not rebuild a stable two-car bay from a one-frame YOLO dropout.
            # Returning no masks makes estimate() coast the last locked geometry.
            return (), "locked_gap_coast"
        if locked_two_car_gap:
            # The dropout outlived the hold window.  Forget the stale bay before
            # accepting the survivor-based fallback selected above.
            self._candidate = None
            self._confirm_frames = 0
            self._last = None
            self._coast_frames = 0
            self._reconfirming = False
            self._jump_hold_frames = 0
        return selected, mode

    def estimate(
        self,
        masks: Sequence[Any],
        confidence: float = 1.0,
        selection_mode: str = "line_only",
        observed_car_count: int = 0,
    ) -> ParkingGeometry:
        raw = replace(
            self._estimate_raw(masks, confidence, selection_mode),
            observed_car_count=max(0, observed_car_count),
        )
        if not raw.found:
            self._candidate = None
            self._confirm_frames = 0
            if self._last is not None and self._coast_frames < self.config.max_coast_frames:
                self._coast_frames += 1
                decay = max(0.0, 1.0 - self._coast_frames / float(self.config.max_coast_frames + 1))
                return replace(
                    self._last,
                    confidence=self._last.confidence * decay,
                    observed_car_count=raw.observed_car_count,
                    coasted=True,
                    selection_mode=(
                        "locked_gap_coast"
                        if selection_mode == "locked_gap_coast"
                        else self._last.selection_mode
                    ),
                    reason=(
                        "coast:locked_two_car_gap"
                        if selection_mode == "locked_gap_coast"
                        else "coast:%s" % raw.reason
                    ),
                )
            self._last = None
            self._coast_frames = 0
            self._reconfirming = False
            self._jump_hold_frames = 0
            return raw

        jumped = self._candidate is None or self._is_jump(self._candidate, raw)
        if jumped:
            if self._last is not None:
                if self._reconfirming:
                    self._jump_hold_frames += 1
                else:
                    self._reconfirming = True
            self._candidate = raw
            self._confirm_frames = 1
        else:
            self._candidate = raw
            self._confirm_frames += 1

        required_frames = max(
            1,
            self.config.jump_reconfirm_frames
            if self._reconfirming
            else self.config.min_confirm_frames,
        )
        if self._confirm_frames < required_frames:
            if self._last is not None and self._reconfirming:
                if not jumped:
                    self._jump_hold_frames += 1
                if self._jump_hold_frames <= max(
                    0,
                    self.config.max_jump_hold_frames,
                ):
                    return replace(
                        self._last,
                        observed_car_count=raw.observed_car_count,
                        coasted=True,
                        reason="coast:reconfirming_jump:%d/%d hold=%d/%d"
                        % (
                            self._confirm_frames,
                            required_frames,
                            self._jump_hold_frames,
                            max(0, self.config.max_jump_hold_frames),
                        ),
                    )
            return replace(
                raw,
                found=False,
                confidence=raw.confidence * self._confirm_frames / required_frames,
                reason="confirming:%d/%d" % (
                    self._confirm_frames,
                    required_frames,
                ),
            )

        result = self._smooth(self._last, raw)
        self._last = result
        self._coast_frames = 0
        self._reconfirming = False
        self._jump_hold_frames = 0
        return result

    def _estimate_raw(
        self,
        masks: Sequence[Any],
        detection_confidence: float,
        selection_mode: str,
    ) -> ParkingGeometry:
        lines = [
            line
            for line in (
                self._fit_line(mask, mask_index)
                for mask_index, mask in enumerate(masks)
            )
            if line is not None
        ]
        line_count = len(lines)
        if lines and selection_mode in (
            "single_car_left",
            "single_car_right",
            "two_car_left_line",
            "two_car_right_line",
        ):
            anchor = lines[0]
            if abs(anchor.direction_y) < 0.55:
                anchor = replace(
                    anchor,
                    direction_x=0.0,
                    direction_y=-1.0,
                    length_px=max(anchor.length_px, self._shape[0] * 0.65),
                )
            left, right = self._virtual_pair(anchor, selection_mode)
            best_back = None
            best_back_score = -1.0
            for candidate in lines[1:]:
                perpendicular_error = abs(
                    90.0 - axial_angle_difference(anchor.angle_deg, candidate.angle_deg)
                )
                if perpendicular_error > self.config.perpendicular_tolerance_deg:
                    continue
                left_corner = line_intersection(left, candidate)
                right_corner = line_intersection(right, candidate)
                if left_corner is None or right_corner is None:
                    continue
                support_count = sum(
                    (
                        left.supports(left_corner, self.config.intersection_margin_px),
                        right.supports(right_corner, self.config.intersection_margin_px),
                        candidate.supports(left_corner, self.config.intersection_margin_px),
                        candidate.supports(right_corner, self.config.intersection_margin_px),
                    )
                )
                if support_count < 3:
                    continue
                score = candidate.quality + support_count / 4.0
                if score > best_back_score:
                    best_back = candidate
                    best_back_score = score
            if best_back is None:
                best_back = self._virtual_back(left, right, anchor)
            return self._build_geometry(
                left,
                right,
                (anchor.direction_x, anchor.direction_y),
                self.config.expected_slot_width_px,
                detection_confidence,
                0.80,
                line_count,
                selection_mode,
                back=best_back,
            )
        if line_count < 2:
            return ParkingGeometry(observed_line_count=line_count, reason="need_two_lines")

        pair_candidates = []
        for first_index in range(line_count - 1):
            for second_index in range(first_index + 1, line_count):
                first = lines[first_index]
                second = lines[second_index]
                angle_error = axial_angle_difference(first.angle_deg, second.angle_deg)
                if angle_error > self.config.parallel_tolerance_deg:
                    continue
                direction = average_direction(first, second)
                separation = parallel_line_distance(first, second, direction)
                if not (
                    self.config.slot_width_min_px
                    <= separation
                    <= self.config.slot_width_max_px
                ):
                    continue
                parallel_score = 1.0 - angle_error / max(1e-6, self.config.parallel_tolerance_deg)
                width_span = max(
                    1.0,
                    self.config.slot_width_max_px - self.config.slot_width_min_px,
                )
                width_score = max(
                    0.0,
                    1.0 - abs(separation - self.config.expected_slot_width_px) / width_span,
                )
                score = 0.55 * parallel_score + 0.45 * width_score
                pair_candidates.append(
                    (score, first_index, second_index, direction, separation)
                )

        if not pair_candidates:
            return ParkingGeometry(observed_line_count=line_count, reason="no_parallel_side_pair")

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        best_topology = None
        for pair_score, first_index, second_index, direction, separation in pair_candidates:
            first = lines[first_index]
            second = lines[second_index]
            for back_index, back in enumerate(lines):
                if back_index in (first_index, second_index):
                    continue
                perpendicular_error = abs(
                    90.0 - axial_angle_difference(direction_angle(direction), back.angle_deg)
                )
                if perpendicular_error > self.config.perpendicular_tolerance_deg:
                    continue
                first_corner = line_intersection(first, back)
                second_corner = line_intersection(second, back)
                if first_corner is None or second_corner is None:
                    continue
                support_count = sum(
                    (
                        first.supports(first_corner, self.config.intersection_margin_px),
                        second.supports(second_corner, self.config.intersection_margin_px),
                        back.supports(first_corner, self.config.intersection_margin_px),
                        back.supports(second_corner, self.config.intersection_margin_px),
                    )
                )
                if support_count < 3:
                    continue
                support_score = support_count / 4.0
                perpendicular_score = 1.0 - perpendicular_error / max(
                    1e-6, self.config.perpendicular_tolerance_deg
                )
                topology_score = (
                    0.45 * pair_score
                    + 0.35 * perpendicular_score
                    + 0.20 * support_score
                )
                candidate = (
                    topology_score,
                    first,
                    second,
                    back,
                    direction,
                    separation,
                )
                if best_topology is None or candidate[0] > best_topology[0]:
                    best_topology = candidate

        if best_topology is not None:
            topology_score, first, second, back, direction, separation = best_topology
            return self._build_geometry(
                first,
                second,
                direction,
                separation,
                detection_confidence,
                topology_score,
                line_count,
                selection_mode,
                back=back,
            )

        pair_score, first_index, second_index, direction, separation = pair_candidates[0]
        first = lines[first_index]
        second = lines[second_index]
        return self._build_geometry(
            first,
            second,
            direction,
            separation,
            detection_confidence,
            pair_score * 0.65,
            line_count,
            selection_mode,
            back=self._virtual_back(first, second, first),
        )

    def _virtual_pair(
        self,
        line: ParkingLine,
        selection_mode: str,
    ) -> Tuple[ParkingLine, ParkingLine]:
        right_dx = -line.direction_y * self.config.expected_slot_width_px
        right_dy = line.direction_x * self.config.expected_slot_width_px
        if selection_mode in ("single_car_left", "two_car_right_line"):
            return replace(line, center_x=line.center_x - right_dx, center_y=line.center_y - right_dy, mask_index=-1), line
        return line, replace(line, center_x=line.center_x + right_dx, center_y=line.center_y + right_dy, mask_index=-1)

    def _virtual_back(
        self,
        left: ParkingLine,
        right: ParkingLine,
        anchor: ParkingLine,
    ) -> ParkingLine:
        return ParkingLine(
            center_x=(left.center_x + right.center_x) / 2.0,
            center_y=self._shape[0] * self.config.virtual_back_y_ratio,
            direction_x=-anchor.direction_y,
            direction_y=anchor.direction_x,
            length_px=self.config.expected_slot_width_px * 1.25,
            residual_px=0.0,
            quality=min(left.quality, right.quality) * 0.85,
            point_count=0,
            mask_index=-1,
        )

    def _build_geometry(
        self,
        first: ParkingLine,
        second: ParkingLine,
        direction: Tuple[float, float],
        separation: float,
        detection_confidence: float,
        topology_score: float,
        line_count: int,
        selection_mode: str,
        back: Optional[ParkingLine],
    ) -> ParkingGeometry:
        import numpy as np

        if direction[1] > 0.0:
            direction = (-direction[0], -direction[1])
        height, width = self._shape
        target_y = height * self.config.control_target_y_ratio
        first_point = point_on_average_direction(first, direction, target_y)
        second_point = point_on_average_direction(second, direction, target_y)
        if first_point is None or second_point is None:
            return ParkingGeometry(observed_line_count=line_count, reason="side_pair_horizontal")

        if first_point[0] <= second_point[0]:
            left, right = first, second
            left_point, right_point = first_point, second_point
        else:
            left, right = second, first
            left_point, right_point = second_point, first_point

        center_x = (left_point[0] + right_point[0]) / 2.0
        vehicle_center_x = width * self.config.vehicle_center_x_ratio
        lateral_error = center_x - vehicle_center_x
        lateral_norm = lateral_error / max(1.0, separation / 2.0)
        heading_error = degrees(atan2(direction[0], -direction[1]))

        center_line = ParkingLine(
            center_x=(left.center_x + right.center_x) / 2.0,
            center_y=(left.center_y + right.center_y) / 2.0,
            direction_x=direction[0],
            direction_y=direction[1],
            length_px=(left.length_px + right.length_px) / 2.0,
            residual_px=(left.residual_px + right.residual_px) / 2.0,
            quality=(left.quality + right.quality) / 2.0,
            point_count=left.point_count + right.point_count,
            mask_index=-1,
        )

        depth = None
        remaining = None
        back_center_x = None
        back_center_y = None
        stop_target_x = None
        stop_target_y = None
        if back is not None:
            back_center = line_intersection(center_line, back)
            if back_center is not None:
                vehicle_reference_y = height * self.config.vehicle_reference_y_ratio
                depth = vehicle_reference_y - back_center[1]
                remaining = depth - self.config.desired_back_clearance_px
                back_center_x = back_center[0]
                back_center_y = back_center[1]
                # direction points from the bay mouth toward the back line.
                # Move opposite that vector to keep the rear reference point a
                # configured clearance in front of the painted back line.
                stop_target_x = back_center[0] - direction[0] * self.config.desired_back_clearance_px
                stop_target_y = back_center[1] - direction[1] * self.config.desired_back_clearance_px

        line_quality = float(np.mean([left.quality, right.quality] + ([back.quality] if back else [])))
        confidence = clip(
            float(detection_confidence) * line_quality * topology_score,
            0.0,
            1.0,
        )
        found = confidence >= self.config.min_geometry_confidence
        return ParkingGeometry(
            found=found,
            has_side_pair=True,
            has_back_line=back is not None and depth is not None,
            left=left,
            right=right,
            back=back,
            lateral_error_px=lateral_error,
            lateral_error_norm=clip(lateral_norm, -2.0, 2.0),
            heading_error_deg=heading_error,
            depth_to_back_px=depth,
            depth_remaining_px=remaining,
            slot_width_px=separation,
            vehicle_x_px=vehicle_center_x,
            vehicle_y_px=height * self.config.vehicle_reference_y_ratio,
            slot_center_x_px=center_line.center_x,
            slot_center_y_px=center_line.center_y,
            slot_direction_x=direction[0],
            slot_direction_y=direction[1],
            back_center_x_px=back_center_x,
            back_center_y_px=back_center_y,
            stop_target_x_px=stop_target_x,
            stop_target_y_px=stop_target_y,
            confidence=confidence,
            observed_line_count=line_count,
            selection_mode=selection_mode,
            reason=(
                ("parking_bay" if back is not None else "side_pair")
                if found
                else "low_confidence"
            ),
        )

    def _fit_line(self, mask: Any, mask_index: int = -1) -> Optional[ParkingLine]:
        import numpy as np

        array = np.asarray(mask)
        self._shape = array.shape[:2]
        ys, xs = np.nonzero(array > 0)
        count = len(xs)
        if count < self.config.min_line_pixels:
            return None
        if count > self.config.max_fit_points:
            step = max(1, count // self.config.max_fit_points)
            xs = xs[::step]
            ys = ys[::step]

        points = np.column_stack((xs.astype(float), ys.astype(float)))
        center = np.mean(points, axis=0)
        centered = points - center
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        if direction[1] > 0.0 or (abs(direction[1]) < 1e-6 and direction[0] < 0.0):
            direction = -direction
        projections = centered @ direction
        perpendicular = centered - np.outer(projections, direction)
        distances = np.linalg.norm(perpendicular, axis=1)
        residual = float(np.sqrt(np.mean(distances ** 2)))
        if residual > self.config.max_line_residual_px:
            return None
        length = float(np.max(projections) - np.min(projections))
        if length < 5.0:
            return None
        quality = clip(1.0 - residual / max(1e-6, self.config.max_line_residual_px), 0.0, 1.0)
        return ParkingLine(
            center_x=float(center[0]),
            center_y=float(center[1]),
            direction_x=float(direction[0]),
            direction_y=float(direction[1]),
            length_px=length,
            residual_px=residual,
            quality=quality,
            point_count=count,
            mask_index=mask_index,
        )

    def _smooth(
        self,
        previous: Optional[ParkingGeometry],
        current: ParkingGeometry,
    ) -> ParkingGeometry:
        if previous is None:
            return current
        alpha = clip(self.config.smooth_alpha, 0.0, 1.0)

        def blend(old: float, new: float) -> float:
            return alpha * new + (1.0 - alpha) * old

        def blend_optional(old: Optional[float], new: Optional[float]) -> Optional[float]:
            if old is None or new is None:
                return new
            return blend(old, new)

        depth = current.depth_to_back_px
        remaining = current.depth_remaining_px
        if previous.depth_to_back_px is not None and depth is not None:
            depth = blend(previous.depth_to_back_px, depth)
        if previous.depth_remaining_px is not None and remaining is not None:
            remaining = blend(previous.depth_remaining_px, remaining)
        direction_x = blend(previous.slot_direction_x, current.slot_direction_x)
        direction_y = blend(previous.slot_direction_y, current.slot_direction_y)
        direction_length = hypot(direction_x, direction_y)
        if direction_length > 1e-9:
            direction_x /= direction_length
            direction_y /= direction_length
        return replace(
            current,
            lateral_error_px=blend(previous.lateral_error_px, current.lateral_error_px),
            lateral_error_norm=blend(previous.lateral_error_norm, current.lateral_error_norm),
            heading_error_deg=blend(previous.heading_error_deg, current.heading_error_deg),
            depth_to_back_px=depth,
            depth_remaining_px=remaining,
            slot_center_x_px=blend_optional(previous.slot_center_x_px, current.slot_center_x_px),
            slot_center_y_px=blend_optional(previous.slot_center_y_px, current.slot_center_y_px),
            slot_direction_x=direction_x,
            slot_direction_y=direction_y,
            back_center_x_px=blend_optional(previous.back_center_x_px, current.back_center_x_px),
            back_center_y_px=blend_optional(previous.back_center_y_px, current.back_center_y_px),
            stop_target_x_px=blend_optional(previous.stop_target_x_px, current.stop_target_x_px),
            stop_target_y_px=blend_optional(previous.stop_target_y_px, current.stop_target_y_px),
            confidence=blend(previous.confidence, current.confidence),
            coasted=False,
        )

    def _is_jump(self, previous: ParkingGeometry, current: ParkingGeometry) -> bool:
        if abs(current.lateral_error_px - previous.lateral_error_px) > self.config.max_lateral_jump_px:
            return True
        if (
            abs(axial_signed_difference(current.heading_error_deg, previous.heading_error_deg))
            > self.config.max_heading_jump_deg
        ):
            return True
        if previous.depth_to_back_px is not None and current.depth_to_back_px is not None:
            if abs(current.depth_to_back_px - previous.depth_to_back_px) > self.config.max_depth_jump_px:
                return True
        return False


def select_parking_line_masks(
    line_masks: Sequence[Any],
    car_masks: Sequence[Any],
    after_two_cars: bool = False,
) -> Tuple[Tuple[Any, ...], str]:
    """Select a virtual left bay until a real two-car gap is confirmed."""

    lines = tuple(line_masks)
    if len(lines) == 0 or len(car_masks) == 0:
        return lines, "line_only"

    car_bounds = [bounds for mask in car_masks if (bounds := _mask_x_bounds(mask))]
    line_anchors = [(_mask_top_anchor_x(mask), mask) for mask in lines]
    line_anchors = [(anchor, mask) for anchor, mask in line_anchors if anchor is not None]
    if not car_bounds or not line_anchors:
        return lines, "line_only"

    image_width = int(getattr(lines[0], "shape", (0, 0))[1])
    image_center = image_width / 2.0
    cars = sorted(car_bounds, key=lambda bounds: (bounds[0] + bounds[1]) / 2.0)

    if len(cars) >= 2:
        candidates = []
        for left_car, right_car in zip(cars, cars[1:]):
            left_center = (left_car[0] + left_car[1]) / 2.0
            right_center = (right_car[0] + right_car[1]) / 2.0
            selected = tuple(
                (anchor, mask)
                for anchor, mask in line_anchors
                if left_center <= anchor <= right_center
            )
            if selected:
                gap_center = (left_car[1] + right_car[0]) / 2.0
                candidates.append((abs(gap_center - image_center), gap_center, selected))
        if candidates:
            _, gap_center, selected = min(candidates, key=lambda item: item[0])
            masks = tuple(mask for _, mask in selected)
            if _has_parallel_mask_pair(masks):
                return masks, "two_car"
            anchor, mask = min(selected, key=lambda item: abs(item[0] - gap_center))
            return (mask,) + _perpendicular_line_masks(mask, lines), (
                "two_car_left_line"
                if anchor <= gap_center
                else "two_car_right_line"
            )

    if len(cars) == 1:
        car_center = (cars[0][0] + cars[0][1]) / 2.0
        # Before a two-car gap is known, the first car always anchors a virtual
        # bay on its left.  Keep that identity even while it traverses the image.
        # After a real gap was seen, use screen position only as a long-dropout
        # survivor fallback: left survivor -> gap right, right survivor -> gap left.
        slot_is_right_of_car = after_two_cars and car_center <= image_center
        candidates = [
            (anchor, mask)
            for anchor, mask in line_anchors
            if (anchor > car_center) == slot_is_right_of_car
        ]
        if candidates:
            anchor, mask = min(
                candidates,
                key=lambda item: abs(item[0] - car_center),
            )
            return (mask,) + _perpendicular_line_masks(mask, lines), (
                "single_car_right" if slot_is_right_of_car else "single_car_left"
            )

    return lines, "line_only"


def filter_parking_car_masks(
    car_masks: Sequence[Any],
    min_bottom_ratio: float = 0.20,
) -> Tuple[Any, ...]:
    """Keep road-level, car-shaped masks and reject upright people."""

    import numpy as np

    result = []
    for mask in car_masks:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        if (
            width < height * 1.05
            or ys.max() < mask.shape[0] * min(1.0, max(0.0, min_bottom_ratio))
        ):
            continue
        result.append(mask)
    return tuple(result)


def _perpendicular_line_masks(anchor: Any, masks: Sequence[Any]) -> Tuple[Any, ...]:
    anchor_angle = _mask_axis_angle_deg(anchor)
    if anchor_angle is None:
        return ()
    result = []
    for mask in masks:
        if mask is anchor:
            continue
        angle = _mask_axis_angle_deg(mask)
        if angle is not None and abs(90.0 - axial_angle_difference(anchor_angle, angle)) <= 35.0:
            result.append(mask)
    return tuple(result)


def _has_parallel_mask_pair(masks: Sequence[Any]) -> bool:
    angles = [angle for mask in masks if (angle := _mask_axis_angle_deg(mask)) is not None]
    return any(
        axial_angle_difference(first, second) <= 35.0
        for index, first in enumerate(angles[:-1])
        for second in angles[index + 1 :]
    )


def _mask_axis_angle_deg(mask: Any) -> Optional[float]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    center = np.mean(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    return degrees(atan2(float(direction[1]), float(direction[0]))) % 180.0


def _mask_x_bounds(mask: Any) -> Optional[Tuple[float, float]]:
    import numpy as np

    _, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(xs.max())


def _mask_top_anchor_x(mask: Any) -> Optional[float]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    top = float(np.percentile(ys, 10.0))
    band = xs[ys <= top + max(2.0, mask.shape[0] * 0.04)]
    return float(np.median(band if len(band) else xs))


def merge_camera_back_line(
    geometry: ParkingGeometry,
    camera_geometry: ParkingGeometry,
) -> ParkingGeometry:
    """Prefer the camera-detected painted back line over the LiDAR-inferred one.

    LiDAR gets no return off painted lines, so its back edge is only a
    geometric guess (fixed depth from the two bordering cars). The camera/YOLO
    line detector sees the actual paint. Both pipelines share the same BEV
    pixel frame and ``vehicle_reference_y_ratio`` convention, so the camera's
    depth values are directly usable in place of the LiDAR ones.
    """

    if not camera_geometry.has_back_line:
        return geometry
    return replace(
        geometry,
        has_back_line=True,
        back=camera_geometry.back,
        depth_to_back_px=camera_geometry.depth_to_back_px,
        depth_remaining_px=camera_geometry.depth_remaining_px,
        back_center_x_px=camera_geometry.back_center_x_px,
        back_center_y_px=camera_geometry.back_center_y_px,
        stop_target_x_px=camera_geometry.stop_target_x_px,
        stop_target_y_px=camera_geometry.stop_target_y_px,
    )


def merge_camera_slot_guidance(
    geometry: ParkingGeometry,
    camera_geometry: ParkingGeometry,
) -> ParkingGeometry:
    """Use car-selected YOLO side lines while retaining LiDAR safety/depth."""

    merged = replace(
        merge_camera_back_line(geometry, camera_geometry),
        observed_car_count=camera_geometry.observed_car_count,
        observed_line_count=camera_geometry.observed_line_count,
    )
    if (
        not camera_geometry.found
        or not camera_geometry.has_side_pair
        or camera_geometry.left is None
        or camera_geometry.right is None
    ):
        return merged

    back_center_x = merged.back_center_x_px
    back_center_y = merged.back_center_y_px
    stop_target_x = merged.stop_target_x_px
    stop_target_y = merged.stop_target_y_px
    depth_to_back = merged.depth_to_back_px
    depth_remaining = merged.depth_remaining_px

    if (
        merged.back is not None
        and camera_geometry.slot_center_x_px is not None
        and camera_geometry.slot_center_y_px is not None
    ):
        center_line = ParkingLine(
            center_x=camera_geometry.slot_center_x_px,
            center_y=camera_geometry.slot_center_y_px,
            direction_x=camera_geometry.slot_direction_x,
            direction_y=camera_geometry.slot_direction_y,
            length_px=max(
                camera_geometry.left.length_px,
                camera_geometry.right.length_px,
            ),
            residual_px=0.0,
            quality=1.0,
            point_count=2,
        )
        intersection = line_intersection(center_line, merged.back)
        if intersection is not None:
            clearance = (
                hypot(
                    merged.back_center_x_px - merged.stop_target_x_px,
                    merged.back_center_y_px - merged.stop_target_y_px,
                )
                if merged.back_center_x_px is not None
                and merged.back_center_y_px is not None
                and merged.stop_target_x_px is not None
                and merged.stop_target_y_px is not None
                else 0.0
            )
            direction = (
                camera_geometry.slot_direction_x,
                camera_geometry.slot_direction_y,
            )
            back_center_x, back_center_y = intersection
            stop_target_x = back_center_x - direction[0] * clearance
            stop_target_y = back_center_y - direction[1] * clearance
            vehicle_to_back = (
                back_center_x - camera_geometry.vehicle_x_px,
                back_center_y - camera_geometry.vehicle_y_px,
            )
            vehicle_to_stop = (
                stop_target_x - camera_geometry.vehicle_x_px,
                stop_target_y - camera_geometry.vehicle_y_px,
            )
            depth_to_back = (
                vehicle_to_back[0] * direction[0]
                + vehicle_to_back[1] * direction[1]
            )
            depth_remaining = (
                vehicle_to_stop[0] * direction[0]
                + vehicle_to_stop[1] * direction[1]
            )

    return replace(
        merged,
        found=True,
        has_side_pair=True,
        left=camera_geometry.left,
        right=camera_geometry.right,
        lateral_error_px=camera_geometry.lateral_error_px,
        lateral_error_norm=camera_geometry.lateral_error_norm,
        heading_error_deg=camera_geometry.heading_error_deg,
        slot_width_px=camera_geometry.slot_width_px,
        slot_center_x_px=camera_geometry.slot_center_x_px,
        slot_center_y_px=camera_geometry.slot_center_y_px,
        slot_direction_x=camera_geometry.slot_direction_x,
        slot_direction_y=camera_geometry.slot_direction_y,
        vehicle_x_px=camera_geometry.vehicle_x_px,
        vehicle_y_px=camera_geometry.vehicle_y_px,
        back_center_x_px=back_center_x,
        back_center_y_px=back_center_y,
        stop_target_x_px=stop_target_x,
        stop_target_y_px=stop_target_y,
        depth_to_back_px=depth_to_back,
        depth_remaining_px=depth_remaining,
        observed_line_count=camera_geometry.observed_line_count,
        confidence=max(merged.confidence, camera_geometry.confidence),
        selection_mode=camera_geometry.selection_mode,
    )


def axial_angle_difference(first_deg: float, second_deg: float) -> float:
    difference = abs((first_deg - second_deg) % 180.0)
    return min(difference, 180.0 - difference)


def axial_signed_difference(first_deg: float, second_deg: float) -> float:
    return abs(((first_deg - second_deg + 90.0) % 180.0) - 90.0)


def direction_angle(direction: Tuple[float, float]) -> float:
    return degrees(atan2(direction[1], direction[0])) % 180.0


def average_direction(first: ParkingLine, second: ParkingLine) -> Tuple[float, float]:
    first_vector = (first.direction_x, first.direction_y)
    second_vector = (second.direction_x, second.direction_y)
    if first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1] < 0.0:
        second_vector = (-second_vector[0], -second_vector[1])
    dx = first_vector[0] + second_vector[0]
    dy = first_vector[1] + second_vector[1]
    length = hypot(dx, dy)
    if length < 1e-9:
        return first_vector
    return (dx / length, dy / length)


def parallel_line_distance(
    first: ParkingLine,
    second: ParkingLine,
    direction: Tuple[float, float],
) -> float:
    normal = (-direction[1], direction[0])
    delta = (second.center_x - first.center_x, second.center_y - first.center_y)
    return abs(delta[0] * normal[0] + delta[1] * normal[1])


def point_on_average_direction(
    line: ParkingLine,
    direction: Tuple[float, float],
    target_y: float,
) -> Optional[Tuple[float, float]]:
    if abs(direction[1]) < 1e-6:
        return None
    scale = (target_y - line.center_y) / direction[1]
    return (
        line.center_x + scale * direction[0],
        target_y,
    )


def line_intersection(
    first: ParkingLine,
    second: ParkingLine,
) -> Optional[Tuple[float, float]]:
    p = (first.center_x, first.center_y)
    r = (first.direction_x, first.direction_y)
    q = (second.center_x, second.center_y)
    s = (second.direction_x, second.direction_y)
    denominator = cross(r, s)
    if abs(denominator) < 1e-8:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    scale = cross(q_minus_p, s) / denominator
    return (p[0] + scale * r[0], p[1] + scale * r[1])


def cross(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
