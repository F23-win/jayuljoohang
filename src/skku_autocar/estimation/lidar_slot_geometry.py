from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Optional, Sequence, Tuple

from .parking_geometry import ParkingGeometry, ParkingGeometryConfig, ParkingLine
from .parking_lidar import (
    LidarParkingConfig,
    LidarParkingObservation,
    infer_dynamic_slot_polygon,
)


Point = Tuple[float, float]


@dataclass(frozen=True)
class SlotFootprintMeasurement:
    corners_local_mm: Tuple[Point, Point, Point, Point]
    min_lateral_mm: float
    max_lateral_mm: float
    min_depth_mm: float
    max_depth_mm: float
    inside_ratio: float
    fully_inside: bool
    completion_candidate: bool
    reason: str


class LidarSlotGeometryProjector:
    """Project the tracked LiDAR parking box into the rear-BEV path frame.

    LiDAR coordinates use x to vehicle-right and y toward the rear.  Rear-BEV
    pixels use x to vehicle-right and y toward the vehicle, so positive LiDAR
    rear distance maps to decreasing image y.  The projected rectangle becomes
    the virtual left, right, and back lines consumed by the reverse path planner.
    """

    def __init__(
        self,
        lidar_config: LidarParkingConfig,
        geometry_config: ParkingGeometryConfig,
        canvas_width: int,
        canvas_height: int,
        vehicle_width_mm: float = 600.0,
        vehicle_length_mm: float = 1000.0,
        rear_axle_to_rear_bumper_mm: float = 200.0,
        sensor_behind_vehicle_rear_mm: Optional[float] = None,
        park_completion_clearance_mm: float = 20.0,
    ) -> None:
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("LiDAR slot geometry canvas must be positive")
        if lidar_config.parking_space_width_mm <= 0.0:
            raise ValueError("parking_space_width_mm must be positive")
        self.lidar_config = lidar_config
        self.geometry_config = geometry_config
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.vehicle_width_mm = max(1.0, float(vehicle_width_mm))
        self.vehicle_length_mm = max(1.0, float(vehicle_length_mm))
        self.rear_axle_to_rear_bumper_mm = clip(
            float(rear_axle_to_rear_bumper_mm),
            0.0,
            self.vehicle_length_mm,
        )
        if sensor_behind_vehicle_rear_mm is None:
            rear_bumper_y_back_mm = (
                lidar_config.sensor_to_rear_axle_y_back_mm
                + self.rear_axle_to_rear_bumper_mm
            )
            sensor_behind_vehicle_rear_mm = -rear_bumper_y_back_mm
        self.sensor_behind_vehicle_rear_mm = max(
            0.0,
            float(sensor_behind_vehicle_rear_mm),
        )
        self.park_completion_clearance_mm = max(
            0.0,
            float(park_completion_clearance_mm),
        )
        self.pixels_per_mm = (
            geometry_config.expected_slot_width_px
            / lidar_config.parking_space_width_mm
        )

    def project(self, observation: LidarParkingObservation) -> ParkingGeometry:
        polygon = infer_dynamic_slot_polygon(
            observation,
            self.lidar_config.parking_space_depth_mm,
            self.lidar_config.parking_space_width_mm,
        )
        if polygon is None:
            return ParkingGeometry(reason="lidar_slot_box_unavailable")

        if observation.coasted:
            reason = "lidar_slot_box_hold"
        elif observation.gap_confirmed:
            reason = "lidar_slot_box"
        else:
            reason = "lidar_slot_box_confirming"
        return self.project_polygon(
            polygon,
            confirmed=observation.gap_confirmed,
            coasted=observation.coasted,
            reason=reason,
        )

    def project_polygon(
        self,
        polygon: Sequence[Point],
        *,
        confirmed: bool = True,
        coasted: bool = False,
        reason: str = "lidar_slot_box_locked",
    ) -> ParkingGeometry:
        if len(polygon) != 4:
            return ParkingGeometry(reason="lidar_slot_box_invalid_polygon")

        points = tuple(self._world_to_bev(point) for point in polygon)
        entrance_first, entrance_second, far_second, far_first = points
        entrance_center = midpoint(entrance_first, entrance_second)
        back_center = midpoint(far_first, far_second)
        direction = normalize(subtract(back_center, entrance_center))
        if direction is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_depth")

        first_side = virtual_line(entrance_first, far_first, direction)
        second_side = virtual_line(entrance_second, far_second, direction)
        if first_side is None or second_side is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_side")

        # For a bay pointing upward in rear BEV, the left normal is (-1, 0).
        # This role assignment also works while the box rotates with the car.
        left_normal = (direction[1], -direction[0])
        first_side_mid = midpoint(entrance_first, far_first)
        second_side_mid = midpoint(entrance_second, far_second)
        if dot(first_side_mid, left_normal) >= dot(second_side_mid, left_normal):
            left, left_far = first_side, far_first
            right, right_far = second_side, far_second
        else:
            left, left_far = second_side, far_second
            right, right_far = first_side, far_first

        back = virtual_line(left_far, right_far)
        if back is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_back")

        vehicle = self._rear_axle_pixel()
        clearance = max(0.0, self.geometry_config.desired_back_clearance_px)
        stop_target = (
            back_center[0] - direction[0] * clearance,
            back_center[1] - direction[1] * clearance,
        )
        depth_to_back = dot(subtract(back_center, vehicle), direction)
        depth_remaining = dot(subtract(stop_target, vehicle), direction)
        slot_width = distance(entrance_first, entrance_second)
        slot_depth = distance(entrance_center, back_center)
        slot_center = midpoint(entrance_center, back_center)
        vehicle_width_px = self.vehicle_width_mm * self.pixels_per_mm
        vehicle_length_px = self.vehicle_length_mm * self.pixels_per_mm
        rear_axle_to_rear_bumper_px = (
            self.rear_axle_to_rear_bumper_mm * self.pixels_per_mm
        )

        # Positive lateral error means the bay center is to the vehicle-right
        # when the bay direction is straight up in the rear-BEV frame.
        lateral_error = dot(subtract(vehicle, entrance_center), left_normal)
        lateral_norm = lateral_error / max(1.0, slot_width / 2.0)
        heading_error = degrees(atan2(direction[0], -direction[1]))

        confidence = 0.65 if coasted else 0.95
        found = (
            confirmed
            and confidence >= self.geometry_config.min_geometry_confidence
        )
        footprint = self._vehicle_footprint_in_slot(polygon)

        return ParkingGeometry(
            found=found,
            has_side_pair=True,
            has_back_line=True,
            left=left,
            right=right,
            back=back,
            lateral_error_px=lateral_error,
            lateral_error_norm=clip(lateral_norm, -2.0, 2.0),
            heading_error_deg=heading_error,
            depth_to_back_px=depth_to_back,
            depth_remaining_px=depth_remaining,
            slot_width_px=slot_width,
            slot_depth_px=slot_depth,
            vehicle_x_px=vehicle[0],
            vehicle_y_px=vehicle[1],
            vehicle_width_px=vehicle_width_px,
            vehicle_length_px=vehicle_length_px,
            rear_axle_to_rear_bumper_px=rear_axle_to_rear_bumper_px,
            slot_center_x_px=slot_center[0],
            slot_center_y_px=slot_center[1],
            slot_direction_x=direction[0],
            slot_direction_y=direction[1],
            back_center_x_px=back_center[0],
            back_center_y_px=back_center[1],
            stop_target_x_px=stop_target[0],
            stop_target_y_px=stop_target[1],
            vehicle_inside_ratio=footprint.inside_ratio,
            vehicle_fully_inside=footprint.fully_inside,
            vehicle_footprint_slot_local_mm=(
                footprint.corners_local_mm
            ),
            vehicle_footprint_min_lateral_mm=(
                footprint.min_lateral_mm
            ),
            vehicle_footprint_max_lateral_mm=(
                footprint.max_lateral_mm
            ),
            vehicle_footprint_min_depth_mm=footprint.min_depth_mm,
            vehicle_footprint_max_depth_mm=footprint.max_depth_mm,
            park_completion_candidate=(
                footprint.completion_candidate
            ),
            park_completion_reason=footprint.reason,
            confidence=confidence,
            observed_line_count=0,
            coasted=coasted,
            reason=reason,
        )

    def _vehicle_footprint_in_slot(
        self,
        slot_polygon: Sequence[Point],
    ) -> SlotFootprintMeasurement:
        entrance_center = midpoint(slot_polygon[0], slot_polygon[1])
        width_axis = normalize(
            subtract(slot_polygon[1], slot_polygon[0])
        )
        depth_axis = normalize(
            subtract(slot_polygon[3], slot_polygon[0])
        )
        slot_width_mm = distance(slot_polygon[0], slot_polygon[1])
        slot_depth_mm = distance(slot_polygon[0], slot_polygon[3])
        if (
            width_axis is None
            or depth_axis is None
            or slot_width_mm <= 0.0
            or slot_depth_mm <= 0.0
        ):
            return SlotFootprintMeasurement(
                corners_local_mm=((0.0, 0.0),) * 4,
                min_lateral_mm=0.0,
                max_lateral_mm=0.0,
                min_depth_mm=0.0,
                max_depth_mm=0.0,
                inside_ratio=0.0,
                fully_inside=False,
                completion_candidate=False,
                reason="footprint_slot_frame_invalid",
            )

        half_vehicle_width = self.vehicle_width_mm / 2.0
        rear_bumper_y_back = -self.sensor_behind_vehicle_rear_mm
        front_bumper_y_back = (
            rear_bumper_y_back - self.vehicle_length_mm
        )
        vehicle_corners = (
            (-half_vehicle_width, front_bumper_y_back),
            (half_vehicle_width, front_bumper_y_back),
            (half_vehicle_width, rear_bumper_y_back),
            (-half_vehicle_width, rear_bumper_y_back),
        )

        def to_slot_local(point: Point) -> Point:
            relative = subtract(point, entrance_center)
            return dot(relative, width_axis), dot(relative, depth_axis)

        corners_local = tuple(
            to_slot_local(point) for point in vehicle_corners
        )
        lateral_values = tuple(point[0] for point in corners_local)
        depth_values = tuple(point[1] for point in corners_local)
        minimum_lateral = min(lateral_values)
        maximum_lateral = max(lateral_values)
        minimum_depth = min(depth_values)
        maximum_depth = max(depth_values)
        half_slot_width = slot_width_mm / 2.0
        epsilon = 1e-6
        fully_inside = (
            minimum_lateral >= -half_slot_width - epsilon
            and maximum_lateral <= half_slot_width + epsilon
            and minimum_depth >= -epsilon
            and maximum_depth <= slot_depth_mm + epsilon
        )

        inside = 0
        total = 0
        for row in range(9):
            y_back = (
                front_bumper_y_back
                + (rear_bumper_y_back - front_bumper_y_back)
                * row
                / 8.0
            )
            for column in range(5):
                x_right = (
                    -half_vehicle_width
                    + 2.0 * half_vehicle_width * column / 4.0
                )
                lateral, depth = to_slot_local((x_right, y_back))
                total += 1
                inside += int(
                    -half_slot_width - epsilon
                    <= lateral
                    <= half_slot_width + epsilon
                    and -epsilon <= depth <= slot_depth_mm + epsilon
                )

        clearance = self.park_completion_clearance_mm
        completion_candidate = (
            minimum_lateral >= -half_slot_width + clearance
            and maximum_lateral <= half_slot_width - clearance
            and minimum_depth >= clearance
            and maximum_depth <= slot_depth_mm - clearance
        )
        if completion_candidate:
            reason = "footprint_inside_fixed_slot"
        elif minimum_depth < clearance:
            reason = "footprint_crosses_slot_entrance"
        elif maximum_depth > slot_depth_mm - clearance:
            reason = "footprint_crosses_slot_back"
        elif (
            minimum_lateral < -half_slot_width + clearance
            or maximum_lateral > half_slot_width - clearance
        ):
            reason = "footprint_crosses_slot_side"
        else:
            reason = "footprint_not_complete"

        return SlotFootprintMeasurement(
            corners_local_mm=corners_local,  # type: ignore[arg-type]
            min_lateral_mm=minimum_lateral,
            max_lateral_mm=maximum_lateral,
            min_depth_mm=minimum_depth,
            max_depth_mm=maximum_depth,
            inside_ratio=inside / float(max(1, total)),
            fully_inside=fully_inside,
            completion_candidate=completion_candidate,
            reason=reason,
        )

    def _rear_axle_pixel(self) -> Point:
        return (
            self.canvas_width * self.geometry_config.vehicle_center_x_ratio,
            self.canvas_height * self.geometry_config.vehicle_reference_y_ratio,
        )

    def _world_to_bev(self, point: Point) -> Point:
        rear_axle = self._rear_axle_pixel()
        relative_x = point[0]
        relative_y_back = (
            point[1] - self.lidar_config.sensor_to_rear_axle_y_back_mm
        )
        return (
            rear_axle[0] + relative_x * self.pixels_per_mm,
            rear_axle[1] - relative_y_back * self.pixels_per_mm,
        )


def virtual_line(
    first: Point,
    second: Point,
    preferred_direction: Optional[Point] = None,
) -> Optional[ParkingLine]:
    vector = subtract(second, first)
    direction = normalize(vector)
    if direction is None:
        return None
    if preferred_direction is not None and dot(direction, preferred_direction) < 0.0:
        direction = (-direction[0], -direction[1])
    return ParkingLine(
        center_x=(first[0] + second[0]) / 2.0,
        center_y=(first[1] + second[1]) / 2.0,
        direction_x=direction[0],
        direction_y=direction[1],
        length_px=distance(first, second),
        residual_px=0.0,
        quality=1.0,
        point_count=2,
        mask_index=-1,
    )


def midpoint(first: Point, second: Point) -> Point:
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def subtract(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def normalize(vector: Point) -> Optional[Point]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_in_convex_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    signs = []
    for first, second in zip(polygon, tuple(polygon[1:]) + (polygon[0],)):
        cross = (
            (second[0] - first[0]) * (point[1] - first[1])
            - (second[1] - first[1]) * (point[0] - first[0])
        )
        if abs(cross) > 1e-6:
            signs.append(cross > 0.0)
    return not signs or all(sign == signs[0] for sign in signs)
