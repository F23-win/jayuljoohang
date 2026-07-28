from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, ceil, cos, degrees, hypot, sin
from typing import Optional, Tuple

from ..estimation.parking_geometry import ParkingGeometry


Point = Tuple[float, float]


class ReversePathStatus(str, Enum):
    """Why a reverse path is or is not safe to arm."""

    READY = "ready"
    GEOMETRY_UNAVAILABLE = "geometry_unavailable"
    NEEDS_ALIGNMENT = "needs_alignment"
    PHYSICALLY_IMPOSSIBLE = "physically_impossible"
    COLLISION_RISK = "collision_risk"


@dataclass(frozen=True)
class ReversePathConfig:
    """Image-plane path parameters for rear-BEV reverse parking.

    ``entry_steering_ratio_candidates`` represents experimentally selectable
    steering magnitudes relative to the calibrated full-steering curvature.
    Keeping the ratios in configuration lets the real vehicle test a different
    steering set without changing the path code.
    """

    samples: int = 21
    start_tangent_px: float = 120.0
    end_tangent_px: float = 120.0
    lookahead_px: float = 90.0
    minimum_target_distance_px: float = 8.0
    maximum_curvature_per_px: float = 0.0054
    full_steering_curvature_per_px: float = 0.0054
    footprint_clearance_px: float = 2.0
    entry_steering_ratio_candidates: Tuple[float, ...] = (
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
    )
    collision_sample_step_px: float = 8.0
    minimum_final_inside_depth_px: float = 20.0
    minimum_side_clearance_px: float = 8.0


@dataclass(frozen=True)
class ReversePath:
    found: bool = False
    points: Tuple[Point, ...] = ()
    lookahead_point: Optional[Point] = None
    curvature_per_px: float = 0.0
    maximum_curvature_per_px: float = 0.0
    reason: str = "no_geometry"
    # Keep this after the legacy fields so positional construction remains
    # backward compatible.
    status: ReversePathStatus = ReversePathStatus.GEOMETRY_UNAVAILABLE
    entry_steering_ratio: float = 0.0
    entry_heading_change_deg: float = 0.0
    final_lateral_offset_px: Optional[float] = None
    minimum_side_clearance_px: Optional[float] = None
    maximum_entry_depth_px: Optional[float] = None


class ReverseParkingPathGenerator:
    """Search a complete arc-then-straight reverse manoeuvre.

    A path is armable only when the swept vehicle body can pass the neighbouring
    cars, enter the fixed slot, and finish at the configured back clearance.
    This prevents a short path that never reaches the slot mouth from being
    incorrectly marked ``READY``.
    """

    def __init__(self, config: ReversePathConfig = ReversePathConfig()):
        self.config = config

    def generate(self, geometry: ParkingGeometry) -> ReversePath:
        if not geometry.found or not geometry.has_side_pair:
            return ReversePath(
                status=ReversePathStatus.GEOMETRY_UNAVAILABLE,
                reason="parking_side_lines_missing",
            )
        if (
            not geometry.has_back_line
            or geometry.stop_target_x_px is None
            or geometry.stop_target_y_px is None
        ):
            return ReversePath(
                status=ReversePathStatus.GEOMETRY_UNAVAILABLE,
                reason="parking_back_line_missing",
            )

        start = (geometry.vehicle_x_px, geometry.vehicle_y_px)
        stop_target = (geometry.stop_target_x_px, geometry.stop_target_y_px)
        direction = normalize((geometry.slot_direction_x, geometry.slot_direction_y))
        if direction is None:
            return ReversePath(
                status=ReversePathStatus.PHYSICALLY_IMPOSSIBLE,
                reason="invalid_slot_direction",
            )
        physical_failure = footprint_fit_failure(
            geometry,
            max(0.0, self.config.footprint_clearance_px),
        )
        if physical_failure is not None:
            return ReversePath(
                status=ReversePathStatus.PHYSICALLY_IMPOSSIBLE,
                reason=physical_failure,
            )
        progress = dot(
            (stop_target[0] - start[0], stop_target[1] - start[1]),
            direction,
        )
        if progress < self.config.minimum_target_distance_px:
            return ReversePath(
                status=ReversePathStatus.NEEDS_ALIGNMENT,
                reason="target_not_behind_vehicle",
            )

        # Camera-only legacy geometries and older unit-test fixtures do not
        # carry metric body/slot dimensions. Keep their local path behaviour,
        # but require the complete swept-footprint search for LiDAR geometry.
        if (
            footprint_dimensions(geometry) is None
            or slot_collision_frame(geometry) is None
        ):
            return self._generate_local_path(
                geometry,
                start,
                stop_target,
                direction,
                progress,
            )
        return self._generate_full_path(
            geometry,
            start,
            stop_target,
            direction,
        )

    def _generate_local_path(
        self,
        geometry: ParkingGeometry,
        start: Point,
        stop_target: Point,
        direction: Point,
        progress: float,
    ) -> ReversePath:
        """Compatibility path for geometry without a measurable footprint."""

        start_from_line = (
            start[0] - stop_target[0],
            start[1] - stop_target[1],
        )
        along = dot(start_from_line, direction)
        projection = (
            stop_target[0] + direction[0] * along,
            stop_target[1] + direction[1] * along,
        )
        lookahead_distance = min(max(1.0, self.config.lookahead_px), progress)
        target = (
            projection[0] + direction[0] * lookahead_distance,
            projection[1] + direction[1] * lookahead_distance,
        )

        dx = target[0] - start[0]
        dy_reverse = start[1] - target[1]
        distance_squared = dx * dx + dy_reverse * dy_reverse
        if distance_squared < 1.0:
            return ReversePath(
                status=ReversePathStatus.NEEDS_ALIGNMENT,
                reason="lookahead_too_close",
            )
        curvature = 2.0 * dx / distance_squared
        curvature_limit = max(1e-9, abs(self.config.maximum_curvature_per_px))
        limited = abs(curvature) > curvature_limit
        curvature = max(-curvature_limit, min(curvature_limit, curvature))
        points = short_arc_points(
            start,
            curvature,
            target,
            max(5, int(self.config.samples)),
        )
        collision = footprint_collision_reason(
            geometry,
            points,
            max(0.0, self.config.footprint_clearance_px),
        )
        if collision is not None:
            return ReversePath(
                found=False,
                status=ReversePathStatus.COLLISION_RISK,
                points=points,
                lookahead_point=target,
                curvature_per_px=curvature,
                maximum_curvature_per_px=abs(curvature),
                reason=collision,
            )
        return ReversePath(
            found=True,
            status=ReversePathStatus.READY,
            points=points,
            lookahead_point=target,
            curvature_per_px=curvature,
            maximum_curvature_per_px=abs(curvature),
            reason="local_target_curvature_limited" if limited else "local_target_ready",
        )

    def _generate_full_path(
        self,
        geometry: ParkingGeometry,
        start: Point,
        stop_target: Point,
        direction: Point,
    ) -> ReversePath:
        slot_frame = slot_collision_frame(geometry)
        dimensions = footprint_dimensions(geometry)
        if slot_frame is None or dimensions is None:
            return ReversePath(
                status=ReversePathStatus.GEOMETRY_UNAVAILABLE,
                reason="full_path_dimensions_unavailable",
            )

        initial_reverse_direction = (0.0, -1.0)
        heading_change = signed_angle(
            initial_reverse_direction,
            direction,
        )
        curvature_limit = max(
            1e-9,
            abs(self.config.maximum_curvature_per_px),
        )
        ratios = steering_ratio_candidates(
            self.config.entry_steering_ratio_candidates,
        )
        if abs(heading_change) <= 1e-5:
            ratios = (0.0,)

        ready_paths = []
        rejected_paths = []
        alignment_failures = []
        for ratio in ratios:
            curvature = (
                0.0
                if ratio == 0.0
                else (
                    curvature_limit
                    * ratio
                    * (1.0 if heading_change > 0.0 else -1.0)
                )
            )
            candidate = arc_straight_candidate(
                start,
                stop_target,
                direction,
                heading_change,
                curvature,
                samples=max(5, int(self.config.samples)),
                sample_step_px=max(
                    0.0,
                    float(self.config.collision_sample_step_px),
                ),
            )
            if candidate is None:
                alignment_failures.append(
                    ReversePath(
                        status=ReversePathStatus.NEEDS_ALIGNMENT,
                        curvature_per_px=curvature,
                        maximum_curvature_per_px=abs(curvature),
                        entry_steering_ratio=ratio,
                        entry_heading_change_deg=degrees(heading_change),
                        reason="entry_arc_overshoots_stop_depth",
                    )
                )
                continue

            points, final_lateral_offset = candidate
            assessment = footprint_sweep_assessment(
                geometry,
                points,
                max(0.0, self.config.footprint_clearance_px),
            )
            final_minimum_depth = final_body_minimum_depth(
                geometry,
                points[-1],
                direction,
            )
            common = dict(
                points=points,
                lookahead_point=point_at_distance(
                    points,
                    max(1.0, self.config.lookahead_px),
                ),
                curvature_per_px=curvature,
                maximum_curvature_per_px=abs(curvature),
                entry_steering_ratio=ratio,
                entry_heading_change_deg=degrees(heading_change),
                final_lateral_offset_px=final_lateral_offset,
                minimum_side_clearance_px=(
                    assessment.minimum_side_clearance_px
                ),
                maximum_entry_depth_px=assessment.maximum_entry_depth_px,
            )
            if assessment.collision_reason is not None:
                rejected_paths.append(
                    ReversePath(
                        status=ReversePathStatus.COLLISION_RISK,
                        reason=assessment.collision_reason,
                        **common,
                    )
                )
                continue
            if (
                assessment.minimum_side_clearance_px is None
                or assessment.minimum_side_clearance_px
                < max(0.0, self.config.minimum_side_clearance_px)
            ):
                rejected_paths.append(
                    ReversePath(
                        status=ReversePathStatus.COLLISION_RISK,
                        reason="insufficient_side_clearance_for_reverse_path",
                        **common,
                    )
                )
                continue
            if (
                final_minimum_depth is None
                or final_minimum_depth
                < max(
                    0.0,
                    self.config.minimum_final_inside_depth_px,
                )
            ):
                alignment_failures.append(
                    ReversePath(
                        status=ReversePathStatus.NEEDS_ALIGNMENT,
                        reason="final_vehicle_body_not_inside_slot",
                        **common,
                    )
                )
                continue
            ready_paths.append(
                ReversePath(
                    found=True,
                    status=ReversePathStatus.READY,
                    reason="full_arc_straight_ready",
                    **common,
                )
            )

        if ready_paths:
            return min(
                ready_paths,
                key=lambda path: (
                    # Curved reverse is executed with the calibrated full-lock
                    # command. Prefer the largest safe candidate so the path
                    # selected here matches the command that will actually be
                    # sent to the vehicle. A smaller ratio is returned only
                    # when every larger ratio was rejected.
                    -abs(path.entry_steering_ratio),
                    abs(path.final_lateral_offset_px or 0.0),
                    -(
                        path.minimum_side_clearance_px
                        if path.minimum_side_clearance_px is not None
                        else float("-inf")
                    ),
                    abs(path.curvature_per_px),
                ),
            )
        if rejected_paths:
            return max(
                rejected_paths,
                key=lambda path: (
                    path.minimum_side_clearance_px
                    if path.minimum_side_clearance_px is not None
                    else float("-inf")
                ),
            )
        if alignment_failures:
            return alignment_failures[0]
        return ReversePath(
            status=ReversePathStatus.NEEDS_ALIGNMENT,
            reason="no_entry_steering_candidates",
        )


@dataclass(frozen=True)
class FootprintSweepAssessment:
    collision_reason: Optional[str] = None
    minimum_side_clearance_px: Optional[float] = None
    maximum_entry_depth_px: Optional[float] = None


def steering_ratio_candidates(values: Tuple[float, ...]) -> Tuple[float, ...]:
    candidates = []
    for value in values:
        ratio = min(1.0, max(0.0, abs(float(value))))
        if ratio <= 1e-6 or any(
            abs(ratio - existing) <= 1e-6 for existing in candidates
        ):
            continue
        candidates.append(ratio)
    return tuple(candidates)


def arc_straight_candidate(
    start: Point,
    stop_target: Point,
    direction: Point,
    heading_change: float,
    curvature: float,
    *,
    samples: int,
    sample_step_px: float,
) -> Optional[Tuple[Tuple[Point, ...], float]]:
    """Create a kinematically continuous constant-curve then straight path."""

    if abs(heading_change) <= 1e-5:
        arc_length = 0.0
        arc_end = start
    else:
        if (
            abs(curvature) <= 1e-9
            or heading_change * curvature <= 0.0
        ):
            return None
        arc_length = abs(heading_change / curvature)
        arc_end = reverse_arc_point(start, curvature, arc_length)

    straight_length = dot(
        (stop_target[0] - arc_end[0], stop_target[1] - arc_end[1]),
        direction,
    )
    if straight_length < 0.0:
        return None
    final_point = (
        arc_end[0] + direction[0] * straight_length,
        arc_end[1] + direction[1] * straight_length,
    )
    total_length = arc_length + straight_length
    count = max(5, int(samples))
    if sample_step_px > 1e-6:
        count = max(count, int(ceil(total_length / sample_step_px)) + 1)
    distances = [
        total_length * index / float(max(1, count - 1))
        for index in range(count)
    ]
    if 1e-6 < arc_length < total_length - 1e-6:
        distances.append(arc_length)
        distances.sort()

    points = []
    for distance in distances:
        if distance <= arc_length or straight_length <= 1e-9:
            point = reverse_arc_point(start, curvature, distance)
        else:
            along = distance - arc_length
            point = (
                arc_end[0] + direction[0] * along,
                arc_end[1] + direction[1] * along,
            )
        if not points or hypot(
            point[0] - points[-1][0],
            point[1] - points[-1][1],
        ) > 1e-9:
            points.append(point)
    if not points:
        return None
    points[-1] = final_point
    left_normal = (direction[1], -direction[0])
    lateral_offset = dot(
        (
            final_point[0] - stop_target[0],
            final_point[1] - stop_target[1],
        ),
        left_normal,
    )
    return tuple(points), lateral_offset


def reverse_arc_point(
    start: Point,
    curvature: float,
    distance: float,
) -> Point:
    if abs(curvature) <= 1e-9:
        return start[0], start[1] - distance
    angle = curvature * distance
    return (
        start[0] + (1.0 - cos(angle)) / curvature,
        start[1] - sin(angle) / curvature,
    )


def footprint_fit_failure(
    geometry: ParkingGeometry,
    clearance_px: float,
) -> Optional[str]:
    dimensions = footprint_dimensions(geometry)
    if dimensions is None:
        return None
    vehicle_width, vehicle_length, _rear_overhang = dimensions
    if geometry.slot_width_px is not None:
        usable_width = geometry.slot_width_px - 2.0 * clearance_px
        if vehicle_width > usable_width:
            return "slot_too_narrow_for_vehicle_footprint"
    if geometry.slot_depth_px is not None:
        usable_depth = geometry.slot_depth_px - 2.0 * clearance_px
        if vehicle_length > usable_depth:
            return "slot_too_shallow_for_vehicle_footprint"
    return None


def footprint_collision_reason(
    geometry: ParkingGeometry,
    path_points: Tuple[Point, ...],
    clearance_px: float,
) -> Optional[str]:
    """Check the swept body rectangle against both side cars and the back edge.

    The inferred slot side boundaries are treated as the inside faces of the
    two neighboring cars. The opening remains free, so a footprint may stay in
    front of the entrance while approaching the bay.
    """

    return footprint_sweep_assessment(
        geometry,
        path_points,
        clearance_px,
    ).collision_reason


def footprint_sweep_assessment(
    geometry: ParkingGeometry,
    path_points: Tuple[Point, ...],
    clearance_px: float,
) -> FootprintSweepAssessment:
    dimensions = footprint_dimensions(geometry)
    slot_frame = slot_collision_frame(geometry)
    if dimensions is None or slot_frame is None or len(path_points) < 2:
        return FootprintSweepAssessment()
    vehicle_width, vehicle_length, rear_overhang = dimensions
    entrance, direction, left_normal, half_slot_width, slot_depth = slot_frame
    front_overhang = vehicle_length - rear_overhang
    allowed_half_width = half_slot_width - clearance_px
    allowed_back_depth = slot_depth - clearance_px
    minimum_side_clearance = None
    maximum_entry_depth = None

    for index, rear_axle in enumerate(path_points):
        reverse_direction = path_tangent(path_points, index)
        if reverse_direction is None:
            continue
        body_points = vehicle_footprint_points(
            rear_axle,
            reverse_direction,
            vehicle_width,
            rear_overhang,
            front_overhang,
        )
        lateral_values = tuple(
            dot((point[0] - entrance[0], point[1] - entrance[1]), left_normal)
            for point in body_points
        )
        depth_values = tuple(
            dot((point[0] - entrance[0], point[1] - entrance[1]), direction)
            for point in body_points
        )
        minimum_depth = min(depth_values)
        maximum_depth = max(depth_values)
        maximum_entry_depth = (
            maximum_depth
            if maximum_entry_depth is None
            else max(maximum_entry_depth, maximum_depth)
        )
        overlaps_slot_depth = maximum_depth >= 0.0 and minimum_depth <= slot_depth
        if overlaps_slot_depth:
            side_clearance = allowed_half_width - max(
                abs(min(lateral_values)),
                abs(max(lateral_values)),
            )
            minimum_side_clearance = (
                side_clearance
                if minimum_side_clearance is None
                else min(minimum_side_clearance, side_clearance)
            )
            if side_clearance < 0.0:
                return FootprintSweepAssessment(
                    collision_reason=(
                        "vehicle_footprint_crosses_side_boundary"
                    ),
                    minimum_side_clearance_px=minimum_side_clearance,
                    maximum_entry_depth_px=maximum_entry_depth,
                )
        if maximum_depth > allowed_back_depth:
            return FootprintSweepAssessment(
                collision_reason="vehicle_footprint_crosses_back_boundary",
                minimum_side_clearance_px=minimum_side_clearance,
                maximum_entry_depth_px=maximum_entry_depth,
            )
    return FootprintSweepAssessment(
        minimum_side_clearance_px=minimum_side_clearance,
        maximum_entry_depth_px=maximum_entry_depth,
    )


def final_body_minimum_depth(
    geometry: ParkingGeometry,
    rear_axle: Point,
    reverse_direction: Point,
) -> Optional[float]:
    dimensions = footprint_dimensions(geometry)
    slot_frame = slot_collision_frame(geometry)
    if dimensions is None or slot_frame is None:
        return None
    vehicle_width, vehicle_length, rear_overhang = dimensions
    entrance, _direction, _left_normal, _half_width, _depth = slot_frame
    body_points = vehicle_footprint_points(
        rear_axle,
        reverse_direction,
        vehicle_width,
        rear_overhang,
        vehicle_length - rear_overhang,
    )
    return min(
        dot(
            (point[0] - entrance[0], point[1] - entrance[1]),
            reverse_direction,
        )
        for point in body_points
    )


def footprint_dimensions(
    geometry: ParkingGeometry,
) -> Optional[Tuple[float, float, float]]:
    values = (
        geometry.vehicle_width_px,
        geometry.vehicle_length_px,
        geometry.rear_axle_to_rear_bumper_px,
    )
    if any(value is None for value in values):
        return None
    width, length, rear_overhang = (float(value) for value in values)
    if width <= 0.0 or length <= 0.0:
        return None
    if rear_overhang < 0.0 or rear_overhang > length:
        return None
    return width, length, rear_overhang


def slot_collision_frame(
    geometry: ParkingGeometry,
) -> Optional[Tuple[Point, Point, Point, float, float]]:
    values = (
        geometry.slot_center_x_px,
        geometry.slot_center_y_px,
        geometry.slot_width_px,
        geometry.slot_depth_px,
    )
    if any(value is None for value in values):
        return None
    direction = normalize((geometry.slot_direction_x, geometry.slot_direction_y))
    if direction is None:
        return None
    center_x, center_y, slot_width, slot_depth = (float(value) for value in values)
    if slot_width <= 0.0 or slot_depth <= 0.0:
        return None
    entrance = (
        center_x - direction[0] * slot_depth / 2.0,
        center_y - direction[1] * slot_depth / 2.0,
    )
    left_normal = (direction[1], -direction[0])
    return entrance, direction, left_normal, slot_width / 2.0, slot_depth


def path_tangent(points: Tuple[Point, ...], index: int) -> Optional[Point]:
    if index <= 0:
        vector = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    elif index >= len(points) - 1:
        vector = (
            points[-1][0] - points[-2][0],
            points[-1][1] - points[-2][1],
        )
    else:
        vector = (
            points[index + 1][0] - points[index - 1][0],
            points[index + 1][1] - points[index - 1][1],
        )
    return normalize(vector)


def vehicle_footprint_points(
    rear_axle: Point,
    reverse_direction: Point,
    vehicle_width_px: float,
    rear_overhang_px: float,
    front_overhang_px: float,
) -> Tuple[Point, ...]:
    half_width = vehicle_width_px / 2.0
    lateral = (-reverse_direction[1], reverse_direction[0])
    rear_center = (
        rear_axle[0] + reverse_direction[0] * rear_overhang_px,
        rear_axle[1] + reverse_direction[1] * rear_overhang_px,
    )
    front_center = (
        rear_axle[0] - reverse_direction[0] * front_overhang_px,
        rear_axle[1] - reverse_direction[1] * front_overhang_px,
    )
    return (
        (
            rear_center[0] - lateral[0] * half_width,
            rear_center[1] - lateral[1] * half_width,
        ),
        (
            rear_center[0] + lateral[0] * half_width,
            rear_center[1] + lateral[1] * half_width,
        ),
        (
            front_center[0] + lateral[0] * half_width,
            front_center[1] + lateral[1] * half_width,
        ),
        (
            front_center[0] - lateral[0] * half_width,
            front_center[1] - lateral[1] * half_width,
        ),
    )


def short_arc_points(
    start: Point,
    curvature: float,
    target: Point,
    samples: int,
) -> Tuple[Point, ...]:
    """Sample the short rear-axle arc used only until the next LiDAR update."""

    dx = target[0] - start[0]
    dy_reverse = start[1] - target[1]
    if abs(curvature) <= 1e-9:
        return tuple(
            (
                start[0] + dx * index / float(samples - 1),
                start[1] - dy_reverse * index / float(samples - 1),
            )
            for index in range(samples)
        )
    bearing = atan2(dx, dy_reverse)
    arc_length = abs((2.0 * bearing) / curvature)
    if arc_length <= 1e-6:
        arc_length = hypot(dx, dy_reverse)
    points = []
    for index in range(samples):
        distance = arc_length * index / float(samples - 1)
        angle = curvature * distance
        local_x = (1.0 - cos(angle)) / curvature
        local_y = sin(angle) / curvature
        points.append((start[0] + local_x, start[1] - local_y))
    return tuple(points)


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    one_minus = 1.0 - t
    a = one_minus ** 3
    b = 3.0 * one_minus ** 2 * t
    c = 3.0 * one_minus * t ** 2
    d = t ** 3
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def point_at_distance(points: Tuple[Point, ...], distance: float) -> Point:
    if not points:
        raise ValueError("path must contain at least one point")
    remaining = max(0.0, distance)
    for first, second in zip(points, points[1:]):
        segment = hypot(second[0] - first[0], second[1] - first[1])
        if segment >= remaining and segment > 1e-9:
            ratio = remaining / segment
            return (
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            )
        remaining -= segment
    return points[-1]


def sampled_maximum_curvature(points: Tuple[Point, ...]) -> float:
    maximum = 0.0
    for previous, current, following in zip(points, points[1:], points[2:]):
        a = hypot(current[0] - previous[0], current[1] - previous[1])
        b = hypot(following[0] - current[0], following[1] - current[1])
        c = hypot(following[0] - previous[0], following[1] - previous[1])
        denominator = a * b * c
        if denominator <= 1e-9:
            continue
        twice_area = abs(
            (current[0] - previous[0]) * (following[1] - previous[1])
            - (current[1] - previous[1]) * (following[0] - previous[0])
        )
        maximum = max(maximum, 2.0 * twice_area / denominator)
    return maximum


def normalize(vector: Point) -> Optional[Point]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def signed_angle(first: Point, second: Point) -> float:
    return atan2(
        first[0] * second[1] - first[1] * second[0],
        dot(first, second),
    )
