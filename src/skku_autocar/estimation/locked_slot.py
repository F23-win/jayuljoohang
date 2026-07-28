from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import atan2, cos, degrees, hypot, sin
from typing import Optional, Sequence, Tuple

import numpy as np

from .lidar_slot_geometry import LidarSlotGeometryProjector
from .parking_geometry import ParkingGeometry
from .parking_lidar import (
    LidarParkingObservation,
    RawTwoCarGapMeasurement,
    infer_dynamic_slot_polygon,
)


Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


class SlotPoseSource(str, Enum):
    NONE = "NONE"
    DIRECT_PAIR = "DIRECT_PAIR"
    DIRECT_PAIR_CORRECTION = "DIRECT_PAIR_CORRECTION"
    DIRECT_PAIR_RECOVERY = "DIRECT_PAIR_RECOVERY"
    SINGLE_CAR_CORRECTION = "SINGLE_CAR_CORRECTION"
    SINGLE_CAR_RECOVERY = "SINGLE_CAR_RECOVERY"
    MAP_LOCALIZATION = "MAP_LOCALIZATION"
    TRACKER_FALLBACK = "TRACKER_FALLBACK"


class MapLocalizationStatus(str, Enum):
    LOCALIZED = "LOCALIZED"
    DEPTH_UNOBSERVABLE = "DEPTH_UNOBSERVABLE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LockedSlotTrackerConfig:
    """Short-range LiDAR odometry used after the parking bay is locked.

    The locked rectangle is expressed in the current LiDAR/vehicle frame. Fresh
    observations of both bordering cars re-anchor its pose, while ICP estimates
    the rigid motion between consecutive scans during temporary occlusion.
    """

    min_points: int = 12
    max_points: int = 180
    min_range_mm: float = 200.0
    max_range_mm: float = 3500.0
    max_correspondence_mm: float = 320.0
    trim_ratio: float = 0.65
    iterations: int = 6
    max_translation_per_scan_mm: float = 300.0
    max_rotation_per_scan_deg: float = 15.0
    max_hold_scans: int = 3
    map_min_points: int = 12
    map_min_points_per_landmark: int = 2
    map_max_points: int = 240
    map_capture_radius_mm: float = 1200.0
    map_max_correspondence_mm: float = 260.0
    map_min_depth_span_mm: float = 120.0
    map_icp_fallback_max_scans: int = 2
    direct_pair_correction_max_translation_mm: float = 80.0
    direct_pair_correction_max_rotation_deg: float = 5.0
    direct_pair_correction_alpha: float = 0.25
    direct_pair_correction_rotation_alpha: float = 0.55
    direct_pair_slew_translation_per_scan_mm: float = 50.0
    direct_pair_slew_rotation_per_scan_deg: float = 3.0


@dataclass(frozen=True)
class LockedSlotPose:
    polygon: Optional[Polygon] = None
    locked: bool = False
    tracked: bool = False
    stale: bool = False
    held: bool = False
    lost: bool = False
    source: SlotPoseSource = SlotPoseSource.NONE
    scan_timestamp: Optional[float] = None
    translation_mm: float = 0.0
    rotation_deg: float = 0.0
    support_points: int = 0
    bilateral_support: bool = False
    depth_span_mm: float = 0.0
    reason: str = "slot_not_locked"


@dataclass(frozen=True)
class MapLocalizationResult:
    status: MapLocalizationStatus
    pose: LockedSlotPose


@dataclass(frozen=True)
class SlotLandmark:
    """One bordering-car surface frozen in slot-local coordinates."""

    landmark_id: str
    local_x_mm: float
    local_y_mm: float


@dataclass(frozen=True)
class FrozenSlotLandmarkMap:
    """Immutable two-car landmark map captured from one DIRECT_PAIR pose.

    The map itself never follows later raw detections. Localization projects
    the current scan into this fixed frame; the planner receives only the
    resulting fixed-size slot geometry, not the map object.
    """

    source_scan_timestamp: float
    slot_width_mm: float
    slot_depth_mm: float
    landmarks: Tuple[SlotLandmark, SlotLandmark]
    surface_points_local: Tuple[Point, ...] = ()
    frozen: bool = True
    reason: str = "direct_pair_landmarks_frozen"

    @classmethod
    def from_direct_pair(
        cls,
        pose: LockedSlotPose,
        measurement: RawTwoCarGapMeasurement,
        *,
        slot_width_mm: float,
        slot_depth_mm: float,
        vehicle_points: Sequence[Point] = (),
        max_surface_points: int = 240,
        capture_radius_mm: float = 1200.0,
        min_surface_points: int = 0,
        min_points_per_landmark: int = 0,
    ) -> Optional["FrozenSlotLandmarkMap"]:
        if (
            pose.source != SlotPoseSource.DIRECT_PAIR
            or pose.polygon is None
            or not pose.locked
            or pose.scan_timestamp is None
            or pose.scan_timestamp != measurement.scan_timestamp
        ):
            return None
        frame = slot_frame_from_polygon(pose.polygon)
        if frame is None:
            return None
        origin, width_axis, depth_axis = frame
        first_local = point_to_slot_local(
            (
                measurement.first_edge_x_right_mm,
                measurement.first_edge_y_back_mm,
            ),
            origin,
            width_axis,
            depth_axis,
        )
        second_local = point_to_slot_local(
            (
                measurement.second_edge_x_right_mm,
                measurement.second_edge_y_back_mm,
            ),
            origin,
            width_axis,
            depth_axis,
        )
        ordered_landmarks = sorted(
            (first_local, second_local),
            key=lambda point: (point[0], point[1]),
        )
        landmarks = (
            SlotLandmark(
                "border_car_first",
                ordered_landmarks[0][0],
                ordered_landmarks[0][1],
            ),
            SlotLandmark(
                "border_car_second",
                ordered_landmarks[1][0],
                ordered_landmarks[1][1],
            ),
        )
        local_surface_points = []
        capture_radius_squared = max(1.0, float(capture_radius_mm)) ** 2
        for point in vehicle_points:
            local = point_to_slot_local(
                (float(point[0]), float(point[1])),
                origin,
                width_axis,
                depth_axis,
            )
            if min(
                (local[0] - landmark.local_x_mm) ** 2
                + (local[1] - landmark.local_y_mm) ** 2
                for landmark in landmarks
            ) <= capture_radius_squared:
                local_surface_points.append(local)
        maximum = max(3, int(max_surface_points))
        if len(local_surface_points) > maximum:
            indexes = np.linspace(
                0,
                len(local_surface_points) - 1,
                maximum,
                dtype=int,
            )
            local_surface_points = [
                local_surface_points[index] for index in indexes
            ]
        if len(local_surface_points) < max(0, int(min_surface_points)):
            return None
        required_per_landmark = max(0, int(min_points_per_landmark))
        if required_per_landmark > 0:
            support = landmark_support_counts(
                np.asarray(local_surface_points, dtype=np.float64),
                landmarks,
            )
            if any(
                count < required_per_landmark
                for count in support
            ):
                return None
        return cls(
            source_scan_timestamp=float(measurement.scan_timestamp),
            slot_width_mm=max(1.0, float(slot_width_mm)),
            slot_depth_mm=max(1.0, float(slot_depth_mm)),
            landmarks=landmarks,
            surface_points_local=tuple(local_surface_points),
        )

    def points_in_vehicle_frame(
        self,
        pose: LockedSlotPose,
    ) -> Tuple[Tuple[str, Point], ...]:
        if pose.polygon is None or not pose.locked:
            return ()
        frame = slot_frame_from_polygon(pose.polygon)
        if frame is None:
            return ()
        origin, width_axis, depth_axis = frame
        return tuple(
            (
                landmark.landmark_id,
                slot_local_to_point(
                    (landmark.local_x_mm, landmark.local_y_mm),
                    origin,
                    width_axis,
                    depth_axis,
                ),
            )
            for landmark in self.landmarks
        )

    def localize(
        self,
        points: Sequence[Point],
        previous_pose: LockedSlotPose,
        config: LockedSlotTrackerConfig,
        *,
        scan_timestamp: Optional[float],
    ) -> MapLocalizationResult:
        """Estimate the frozen slot pose in the current vehicle frame.

        The frozen surface cloud stays in slot-local coordinates. The previous
        pose is only the initial transform for association; the returned pose is
        corrected directly against the immutable map on every scan.
        """

        if previous_pose.polygon is None or not previous_pose.locked:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_previous_pose_unavailable",
            )
        minimum = max(3, int(config.map_min_points))
        minimum_per_landmark = max(
            1,
            int(config.map_min_points_per_landmark),
        )
        if len(self.surface_points_local) < minimum:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.DEPTH_UNOBSERVABLE,
                "slot_map_depth_unobservable",
            )
        map_points = np.asarray(self.surface_points_local, dtype=np.float64)
        map_landmark_indexes = nearest_landmark_indexes(
            map_points,
            self.landmarks,
        )
        (
            map_observable,
            _map_bilateral,
            _map_depth_span,
            _map_required,
        ) = localization_observability(
            map_points,
            map_landmark_indexes,
            minimum_points=minimum,
            minimum_points_per_landmark=minimum_per_landmark,
            minimum_depth_span_mm=config.map_min_depth_span_mm,
        )
        if not map_observable:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.DEPTH_UNOBSERVABLE,
                "slot_map_depth_unobservable",
            )

        frame = slot_frame_from_polygon(previous_pose.polygon)
        if frame is None:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_previous_pose_invalid",
            )
        origin, width_axis, depth_axis = frame
        current_local = [
            point_to_slot_local(
                (float(x), float(y)),
                origin,
                width_axis,
                depth_axis,
            )
            for x, y in points
            if config.min_range_mm
            <= hypot(float(x), float(y))
            <= config.max_range_mm
        ]
        if not current_local:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_scan_points_unavailable",
            )

        current_array = np.asarray(current_local, dtype=np.float64)
        differences = current_array[:, None, :] - map_points[None, :, :]
        distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
        nearest_distances = np.min(distances_squared, axis=1)
        nearest_map_indexes = np.argmin(distances_squared, axis=1)
        observable_limit = max(
            1.0,
            float(config.map_capture_radius_mm),
        ) ** 2
        observable_mask = nearest_distances <= observable_limit
        observable = current_array[observable_mask]
        observable_landmark_indexes = map_landmark_indexes[
            nearest_map_indexes[observable_mask]
        ]
        (
            observable_ok,
            observable_bilateral,
            observable_depth_span,
            _observable_required,
        ) = localization_observability(
            observable,
            observable_landmark_indexes,
            minimum_points=minimum,
            minimum_points_per_landmark=minimum_per_landmark,
            minimum_depth_span_mm=config.map_min_depth_span_mm,
        )
        if not observable_ok:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.DEPTH_UNOBSERVABLE,
                "slot_map_depth_unobservable",
                support_points=len(observable),
                bilateral_support=observable_bilateral,
                depth_span_mm=observable_depth_span,
            )
        association_limit = max(
            1.0,
            float(config.map_max_correspondence_mm),
        ) ** 2
        associated = nearest_distances <= association_limit
        current_array = current_array[associated]
        associated_landmark_indexes = map_landmark_indexes[
            nearest_map_indexes[associated]
        ]
        (
            associated_ok,
            associated_bilateral,
            associated_depth_span,
            required_correspondences,
        ) = localization_observability(
            current_array,
            associated_landmark_indexes,
            minimum_points=minimum,
            minimum_points_per_landmark=minimum_per_landmark,
            minimum_depth_span_mm=config.map_min_depth_span_mm,
        )
        if len(current_array) < required_correspondences:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_correspondences_unavailable",
                support_points=len(current_array),
                bilateral_support=associated_bilateral,
                depth_span_mm=associated_depth_span,
            )
        if not associated_ok:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.DEPTH_UNOBSERVABLE,
                "slot_map_depth_unobservable",
                support_points=len(current_array),
                bilateral_support=associated_bilateral,
                depth_span_mm=associated_depth_span,
            )
        maximum = max(minimum, int(config.map_max_points))
        if len(current_array) > maximum:
            indexes = np.linspace(
                0,
                len(current_array) - 1,
                maximum,
                dtype=int,
            )
            current_array = current_array[indexes]

        transform = estimate_current_to_previous_transform(
            current_array,
            map_points,
            max_correspondence_mm=max(
                1.0,
                float(config.map_max_correspondence_mm),
            ),
            trim_ratio=config.trim_ratio,
            iterations=max(1, int(config.iterations)),
            min_correspondences=max(3, required_correspondences // 2),
        )
        if transform is None:
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_alignment_failed",
            )
        rotation, translation = transform
        rotation_deg = degrees(atan2(rotation[1, 0], rotation[0, 0]))
        translation_mm = hypot(float(translation[0]), float(translation[1]))
        if (
            translation_mm
            > max(0.0, float(config.max_translation_per_scan_mm))
            or abs(rotation_deg)
            > max(0.0, float(config.max_rotation_per_scan_deg))
        ):
            return self._localization_failure(
                previous_pose,
                scan_timestamp,
                MapLocalizationStatus.FAILED,
                "slot_map_motion_gate_rejected",
            )

        inverse_rotation = rotation.T
        half_width = self.slot_width_mm / 2.0
        local_polygon = (
            (-half_width, 0.0),
            (half_width, 0.0),
            (half_width, self.slot_depth_mm),
            (-half_width, self.slot_depth_mm),
        )
        polygon = []
        for local in local_polygon:
            corrected_local = inverse_rotation @ (
                np.asarray(local, dtype=np.float64) - translation
            )
            polygon.append(
                slot_local_to_point(
                    (
                        float(corrected_local[0]),
                        float(corrected_local[1]),
                    ),
                    origin,
                    width_axis,
                    depth_axis,
                )
            )
        return MapLocalizationResult(
            status=MapLocalizationStatus.LOCALIZED,
            pose=LockedSlotPose(
                polygon=tuple(polygon),  # type: ignore[arg-type]
                locked=True,
                tracked=True,
                source=SlotPoseSource.MAP_LOCALIZATION,
                scan_timestamp=scan_timestamp,
                translation_mm=translation_mm,
                rotation_deg=rotation_deg,
                support_points=len(current_array),
                bilateral_support=associated_bilateral,
                depth_span_mm=associated_depth_span,
                reason="slot_map_localized",
            ),
        )

    @staticmethod
    def _localization_failure(
        previous_pose: LockedSlotPose,
        scan_timestamp: Optional[float],
        status: MapLocalizationStatus,
        reason: str,
        support_points: int = 0,
        bilateral_support: bool = False,
        depth_span_mm: float = 0.0,
    ) -> MapLocalizationResult:
        return MapLocalizationResult(
            status=status,
            pose=LockedSlotPose(
                polygon=previous_pose.polygon,
                locked=previous_pose.locked,
                lost=status == MapLocalizationStatus.DEPTH_UNOBSERVABLE,
                source=SlotPoseSource.MAP_LOCALIZATION,
                scan_timestamp=scan_timestamp,
                support_points=support_points,
                bilateral_support=bilateral_support,
                depth_span_mm=depth_span_mm,
                reason=reason,
            ),
        )


@dataclass(frozen=True)
class FixedSlotFrame:
    """Official-size rectangle expressed in a fixed slot-local frame.

    The local origin is the center of the entrance, local x spans the slot
    width, and local y points into the slot. A DIRECT_PAIR pose is rebuilt from
    one raw two-car measurement; it never depends on the previous polygon.
    """

    width_mm: float
    depth_mm: float

    @property
    def local_polygon(self) -> Polygon:
        half_width = max(1.0, float(self.width_mm)) / 2.0
        depth = max(1.0, float(self.depth_mm))
        return (
            (-half_width, 0.0),
            (half_width, 0.0),
            (half_width, depth),
            (-half_width, depth),
        )

    def direct_pair_pose(
        self,
        measurement: RawTwoCarGapMeasurement,
    ) -> Optional[LockedSlotPose]:
        axis_x = (
            measurement.second_edge_x_right_mm
            - measurement.first_edge_x_right_mm
        )
        axis_y = (
            measurement.second_edge_y_back_mm
            - measurement.first_edge_y_back_mm
        )
        axis_length = hypot(axis_x, axis_y)
        if axis_length <= 1e-6 or measurement.observed_width_mm <= 0.0:
            return None
        axis_x /= axis_length
        axis_y /= axis_length

        # Keep the fixed frame orthogonal. Raw depth is used only to select the
        # correct normal sign, not averaged into a previous slot orientation.
        depth_x = -axis_y
        depth_y = axis_x
        raw_depth_length = hypot(
            measurement.slot_depth_x_right,
            measurement.slot_depth_y_back,
        )
        if raw_depth_length <= 1e-6:
            return None
        if (
            depth_x * measurement.slot_depth_x_right
            + depth_y * measurement.slot_depth_y_back
            < 0.0
        ):
            depth_x = -depth_x
            depth_y = -depth_y
        # A gap is geometrically unchanged when the detector swaps its two
        # bordering-car endpoints. The directed depth normal is stable, so
        # rebuild the unoriented width axis from it instead of preserving the
        # arbitrary raw endpoint order. This is a per-scan canonicalization,
        # not temporal averaging or reuse of an older slot box.
        axis_x = -depth_y
        axis_y = depth_x

        origin_x = (
            measurement.first_edge_x_right_mm
            + measurement.second_edge_x_right_mm
        ) / 2.0
        origin_y = (
            measurement.first_edge_y_back_mm
            + measurement.second_edge_y_back_mm
        ) / 2.0
        polygon = tuple(
            (
                origin_x + local_x * axis_x + local_y * depth_x,
                origin_y + local_x * axis_y + local_y * depth_y,
            )
            for local_x, local_y in self.local_polygon
        )
        return LockedSlotPose(
            polygon=polygon,  # type: ignore[arg-type]
            locked=True,
            tracked=True,
            source=SlotPoseSource.DIRECT_PAIR,
            scan_timestamp=measurement.scan_timestamp,
            support_points=2,
            bilateral_support=True,
            reason="direct_pair",
        )


class LockedSlotTracker:
    """Keep one physical parking rectangle stable while the vehicle reverses."""

    def __init__(self, config: LockedSlotTrackerConfig = LockedSlotTrackerConfig()):
        self.config = config
        self._polygon: Optional[Polygon] = None
        self._previous_points: Optional[np.ndarray] = None
        self._hold_scans = 0

    @property
    def locked(self) -> bool:
        return self._polygon is not None

    def reset(self) -> None:
        self._polygon = None
        self._previous_points = None
        self._hold_scans = 0

    def lock(self, polygon: Sequence[Point], points: Sequence[Point]) -> LockedSlotPose:
        return self._set_polygon(polygon, points, reason="slot_locked")

    def reanchor(
        self,
        polygon: Sequence[Point],
        points: Sequence[Point],
    ) -> LockedSlotPose:
        """Replace accumulated ICP pose drift with a fresh two-car gap pose."""

        return self._set_polygon(
            polygon,
            points,
            reason="locked_slot_reanchored",
        )

    def _set_polygon(
        self,
        polygon: Sequence[Point],
        points: Sequence[Point],
        *,
        reason: str,
    ) -> LockedSlotPose:
        if len(polygon) != 4:
            return LockedSlotPose(reason="slot_lock_requires_four_corners")
        sampled = self._prepare_points(points)
        if sampled is None:
            return LockedSlotPose(reason="slot_lock_requires_lidar_points")
        self._polygon = tuple(
            (float(point[0]), float(point[1])) for point in polygon
        )  # type: ignore[assignment]
        self._previous_points = sampled
        self._hold_scans = 0
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            tracked=True,
            source=SlotPoseSource.TRACKER_FALLBACK,
            reason=reason,
        )

    def update(self, points: Sequence[Point]) -> LockedSlotPose:
        if self._polygon is None or self._previous_points is None:
            return LockedSlotPose(reason="slot_not_locked")
        current = self._prepare_points(points)
        if current is None:
            return self._hold("locked_slot_insufficient_points")

        transform = estimate_current_to_previous_transform(
            current,
            self._previous_points,
            max_correspondence_mm=max(1.0, self.config.max_correspondence_mm),
            trim_ratio=self.config.trim_ratio,
            iterations=max(1, self.config.iterations),
            min_correspondences=max(3, self.config.min_points // 2),
        )
        if transform is None:
            return self._hold("locked_slot_scan_match_failed")
        rotation, translation = transform
        rotation_deg = degrees(atan2(rotation[1, 0], rotation[0, 0]))
        translation_mm = hypot(float(translation[0]), float(translation[1]))
        if (
            translation_mm > max(0.0, self.config.max_translation_per_scan_mm)
            or abs(rotation_deg) > max(0.0, self.config.max_rotation_per_scan_deg)
        ):
            return self._hold("locked_slot_motion_gate_rejected")

        # ICP returns previous ~= R * current + t.  The slot is stored in the
        # previous frame, so apply the inverse to express it in the current frame.
        inverse_rotation = rotation.T
        transformed = []
        for point in self._polygon:
            previous = np.asarray(point, dtype=np.float64)
            current_point = inverse_rotation @ (previous - translation)
            transformed.append((float(current_point[0]), float(current_point[1])))
        self._polygon = tuple(transformed)  # type: ignore[assignment]
        self._previous_points = current
        self._hold_scans = 0
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            tracked=True,
            source=SlotPoseSource.TRACKER_FALLBACK,
            translation_mm=translation_mm,
            rotation_deg=rotation_deg,
            reason="locked_slot_tracked",
        )

    def _prepare_points(self, points: Sequence[Point]) -> Optional[np.ndarray]:
        filtered = [
            (float(x), float(y))
            for x, y in points
            if self.config.min_range_mm
            <= hypot(float(x), float(y))
            <= self.config.max_range_mm
        ]
        if len(filtered) < max(3, self.config.min_points):
            return None
        maximum = max(3, int(self.config.max_points))
        if len(filtered) > maximum:
            indexes = np.linspace(0, len(filtered) - 1, maximum, dtype=int)
            filtered = [filtered[index] for index in indexes]
        return np.asarray(filtered, dtype=np.float64)

    def _hold(self, reason: str) -> LockedSlotPose:
        self._hold_scans += 1
        lost = self._hold_scans > max(0, self.config.max_hold_scans)
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            held=not lost,
            lost=lost,
            source=SlotPoseSource.TRACKER_FALLBACK,
            reason="locked_slot_lost" if lost else reason,
        )


class LockedSlotGeometryEstimator:
    """DIRECT_PAIR, frozen-map localization, then short legacy ICP fallback."""

    def __init__(
        self,
        projector: LidarSlotGeometryProjector,
        tracker: LockedSlotTracker,
        *,
        width_mm: float,
        depth_mm: float,
        enabled: bool = True,
    ) -> None:
        self.projector = projector
        self.tracker = tracker
        self.width_mm = max(1.0, float(width_mm))
        self.depth_mm = max(1.0, float(depth_mm))
        self.slot_frame = FixedSlotFrame(self.width_mm, self.depth_mm)
        self.enabled = bool(enabled)
        self.pose = LockedSlotPose()
        self.landmark_map: Optional[FrozenSlotLandmarkMap] = None
        self._icp_fallback_scans = 0

    @property
    def icp_fallback_scans(self) -> int:
        return self._icp_fallback_scans

    def reset(self) -> None:
        self.tracker.reset()
        self.pose = LockedSlotPose()
        self.landmark_map = None
        self._icp_fallback_scans = 0

    def update(
        self,
        observation: LidarParkingObservation,
        points: Sequence[Point],
        *,
        lock_requested: bool,
    ) -> ParkingGeometry:
        if not self.enabled:
            return self.projector.project(observation)
        if not lock_requested:
            # Leaving all slot-pose tracking states starts a new acquisition.
            # Forward prealignment itself requests tracking so an ordered pair
            # can be captured before it disappears behind the turning vehicle.
            if self.pose.locked or self.landmark_map is not None:
                self.reset()
            return self.projector.project(observation)

        direct_pose = (
            self.slot_frame.direct_pair_pose(observation.raw_two_car_gap)
            if (
                lock_requested
                and observation.gap_confirmed
                and observation.raw_two_car_gap is not None
            )
            else None
        )
        single_car_pose = self._single_car_pose(observation)
        first_direct_lock = (
            direct_pose is not None
            and direct_pose.polygon is not None
            and (
                self.landmark_map is None
                or not self.pose.locked
                or self.pose.polygon is None
            )
        )
        if first_direct_lock:
            self._accept_tracked_pose(direct_pose, points)
            self.landmark_map = FrozenSlotLandmarkMap.from_direct_pair(
                direct_pose,
                observation.raw_two_car_gap,
                slot_width_mm=self.width_mm,
                slot_depth_mm=self.depth_mm,
                vehicle_points=points,
                max_surface_points=self.tracker.config.map_max_points,
                capture_radius_mm=(
                    self.tracker.config.map_capture_radius_mm
                ),
                min_surface_points=(
                    self.tracker.config.map_min_points
                ),
                min_points_per_landmark=(
                    self.tracker.config.map_min_points_per_landmark
                ),
            )
        elif (
            lock_requested
            and observation.is_new_scan
            and self.landmark_map is not None
            and self.pose.locked
            and self.pose.polygon is not None
        ):
            map_result = self.landmark_map.localize(
                points,
                self.pose,
                self.tracker.config,
                scan_timestamp=observation.timestamp,
            )
            if map_result.status == MapLocalizationStatus.LOCALIZED:
                selected_pose = map_result.pose
                if direct_pose is not None:
                    corrected = self._bounded_direct_pair_correction(
                        selected_pose,
                        direct_pose,
                    )
                    if corrected is not None:
                        selected_pose = corrected
                    else:
                        # A fresh ordered two-car pair is the best observable
                        # slot reference. If its delta is too large for the
                        # normal filter, follow it with independent per-scan
                        # translation/yaw limits instead of discarding it and
                        # allowing a sparse frozen map to remain authoritative.
                        corrected = self._slew_limited_direct_pair_recovery(
                            selected_pose,
                            direct_pose,
                            reason="direct_pair_slew_correction",
                        )
                        if corrected is not None:
                            selected_pose = corrected
                        else:
                            selected_pose = replace(
                                selected_pose,
                                reason=(
                                    "direct_pair_follow_unavailable:"
                                    "slot_map_localized"
                                ),
                            )
                elif single_car_pose is not None:
                    corrected = self._single_car_correction(
                        selected_pose,
                        single_car_pose,
                    )
                    if corrected is not None:
                        selected_pose = corrected
                    else:
                        selected_pose = replace(
                            selected_pose,
                            reason=(
                                "single_car_follow_unavailable:"
                                "slot_map_localized"
                            ),
                        )
                self._accept_tracked_pose(selected_pose, points)
            elif direct_pose is not None:
                # Normal observations may only make a small filtered
                # correction. When map localization itself is unavailable,
                # however, a large but ordered two-car observation is followed
                # with a per-scan slew limit instead of stopping or replacing
                # the locked rectangle in one jump.
                corrected = self._bounded_direct_pair_correction(
                    self.pose,
                    direct_pose,
                )
                if corrected is not None:
                    self._accept_tracked_pose(corrected, points)
                else:
                    recovered = self._slew_limited_direct_pair_recovery(
                        self.pose,
                        direct_pose,
                    )
                    if recovered is not None:
                        self._accept_tracked_pose(recovered, points)
                    else:
                        self.pose = self._short_icp_fallback_or_lost(
                            points,
                            scan_timestamp=observation.timestamp,
                            map_reason=(
                                "direct_pair_recovery_unavailable:"
                                f"{map_result.pose.reason}"
                            ),
                        )
            elif single_car_pose is not None:
                # A one-car observation is a useful positional anchor, but its
                # axis is inherited from the gap tracker and can be stale while
                # the vehicle turns. Recover the current yaw from consecutive
                # LiDAR scans first, then let the one-car pose correct only the
                # entrance-center translation.
                scan_reference = self._single_car_scan_rotation_reference(
                    points,
                    scan_timestamp=observation.timestamp,
                )
                corrected = self._single_car_correction(
                    scan_reference,
                    single_car_pose,
                )
                if corrected is not None:
                    self._accept_tracked_pose(corrected, points)
                else:
                    self.pose = self._short_icp_fallback_or_lost(
                        points,
                        scan_timestamp=observation.timestamp,
                        map_reason=(
                            "single_car_recovery_unavailable:"
                            f"{map_result.pose.reason}"
                        ),
                    )
            elif (
                map_result.status
                == MapLocalizationStatus.DEPTH_UNOBSERVABLE
            ):
                self.pose = self._short_icp_fallback_or_lost(
                    points,
                    scan_timestamp=observation.timestamp,
                    map_reason=map_result.pose.reason,
                )
            else:
                self.pose = self._short_icp_fallback_or_lost(
                    points,
                    scan_timestamp=observation.timestamp,
                    map_reason=map_result.pose.reason,
                )
        elif self.tracker.locked:
            # Legacy compatibility before a frozen map exists. Once 7B has a
            # map, all scan matching is routed through the bounded fallback.
            if observation.is_new_scan:
                fallback = self.tracker.update(points)
                if fallback.tracked:
                    self.pose = LockedSlotPose(
                        polygon=fallback.polygon,
                        locked=True,
                        tracked=True,
                        source=SlotPoseSource.TRACKER_FALLBACK,
                        scan_timestamp=observation.timestamp,
                        translation_mm=fallback.translation_mm,
                        rotation_deg=fallback.rotation_deg,
                        reason="short_icp_fallback",
                    )
                else:
                    self.pose = self._lost_pose(
                        observation.timestamp,
                        "all_slot_localization_failed:"
                        f"{fallback.reason}",
                    )
        elif lock_requested and observation.is_new_scan:
            self.pose = LockedSlotPose(reason="locked_slot_fallback_unavailable")

        if self.pose.locked:
            if self.pose.lost or self.pose.polygon is None:
                return ParkingGeometry(reason=self.pose.reason)
            return self.projector.project_polygon(
                self.pose.polygon,
                confirmed=True,
                coasted=self.pose.held,
                reason=self.pose.reason,
            )
        if lock_requested:
            return ParkingGeometry(reason=self.pose.reason)
        return self.projector.project(observation)

    def _single_car_pose(
        self,
        observation: LidarParkingObservation,
    ) -> Optional[LockedSlotPose]:
        """Rebuild the fixed-size box from one freshly tracked border car."""

        if (
            not observation.is_new_scan
            or not observation.gap_confirmed
            or not observation.gap_single_cluster_observed
            or observation.coasted
        ):
            return None
        polygon = infer_dynamic_slot_polygon(
            observation,
            self.depth_mm,
            self.width_mm,
        )
        if polygon is None or len(polygon) != 4:
            return None
        return LockedSlotPose(
            polygon=tuple(polygon),  # type: ignore[arg-type]
            locked=True,
            tracked=True,
            source=SlotPoseSource.SINGLE_CAR_CORRECTION,
            scan_timestamp=observation.timestamp,
            support_points=1,
            bilateral_support=False,
            reason="single_car_slot_pose",
        )

    def _single_car_correction(
        self,
        reference: LockedSlotPose,
        observed: LockedSlotPose,
    ) -> Optional[LockedSlotPose]:
        """Use one border car for translation without accepting its stale yaw."""

        if reference.polygon is None or observed.polygon is None:
            return None
        reference_frame = slot_frame_from_polygon(reference.polygon)
        observed_frame = slot_frame_from_polygon(observed.polygon)
        if reference_frame is None or observed_frame is None:
            return None
        reference_origin, reference_width, reference_depth = reference_frame
        observed_origin, _observed_width, _observed_depth = observed_frame
        delta_x = observed_origin[0] - reference_origin[0]
        delta_y = observed_origin[1] - reference_origin[1]
        translation = hypot(delta_x, delta_y)
        config = self.tracker.config
        gated_limit = max(
            0.0,
            float(config.direct_pair_correction_max_translation_mm),
        )
        if translation <= gated_limit:
            translation_scale = min(
                1.0,
                max(0.0, float(config.direct_pair_correction_alpha)),
            )
            source = SlotPoseSource.SINGLE_CAR_CORRECTION
            reason = "single_car_translation_correction"
        else:
            translation_limit = max(
                0.0,
                float(config.direct_pair_slew_translation_per_scan_mm),
            )
            translation_scale = (
                1.0
                if translation <= translation_limit or translation <= 1e-9
                else translation_limit / translation
            )
            source = SlotPoseSource.SINGLE_CAR_RECOVERY
            reason = "single_car_translation_slew_recovery"

        corrected_origin = (
            reference_origin[0] + delta_x * translation_scale,
            reference_origin[1] + delta_y * translation_scale,
        )
        polygon = tuple(
            slot_local_to_point(
                local,
                corrected_origin,
                reference_width,
                reference_depth,
            )
            for local in self.slot_frame.local_polygon
        )
        return LockedSlotPose(
            polygon=polygon,  # type: ignore[arg-type]
            locked=True,
            tracked=True,
            source=source,
            scan_timestamp=observed.scan_timestamp,
            translation_mm=translation * translation_scale,
            # Preserve the scan/map rotation estimate for diagnostics. The
            # observed one-car axis is intentionally never fused here.
            rotation_deg=reference.rotation_deg,
            support_points=1,
            bilateral_support=False,
            reason=reason,
        )

    def _single_car_scan_rotation_reference(
        self,
        points: Sequence[Point],
        *,
        scan_timestamp: Optional[float],
    ) -> LockedSlotPose:
        """Advance yaw with scan matching before one-car translation anchoring."""

        if not self.tracker.locked:
            return self.pose
        fallback = self.tracker.update(points)
        if not fallback.tracked or fallback.polygon is None:
            return self.pose
        return LockedSlotPose(
            polygon=fallback.polygon,
            locked=True,
            tracked=True,
            source=SlotPoseSource.TRACKER_FALLBACK,
            scan_timestamp=scan_timestamp,
            translation_mm=fallback.translation_mm,
            rotation_deg=fallback.rotation_deg,
            reason="single_car_scan_rotation",
        )

    def _accept_tracked_pose(
        self,
        pose: LockedSlotPose,
        points: Sequence[Point],
    ) -> None:
        self.pose = pose
        self._icp_fallback_scans = 0
        if pose.polygon is None:
            self.tracker.reset()
            return
        if self.tracker.locked:
            fallback_seed = self.tracker.reanchor(
                pose.polygon,
                points,
            )
        else:
            fallback_seed = self.tracker.lock(
                pose.polygon,
                points,
            )
        if not fallback_seed.tracked:
            self.tracker.reset()

    def _bounded_direct_pair_correction(
        self,
        reference: LockedSlotPose,
        direct: LockedSlotPose,
    ) -> Optional[LockedSlotPose]:
        if reference.polygon is None or direct.polygon is None:
            return None
        reference_frame = slot_frame_from_polygon(reference.polygon)
        direct_frame = slot_frame_from_polygon(direct.polygon)
        if reference_frame is None or direct_frame is None:
            return None
        reference_origin, reference_width, reference_depth = reference_frame
        direct_origin, _direct_width, direct_depth = direct_frame
        delta_x = direct_origin[0] - reference_origin[0]
        delta_y = direct_origin[1] - reference_origin[1]
        translation = hypot(delta_x, delta_y)
        rotation_rad = atan2(
            reference_depth[0] * direct_depth[1]
            - reference_depth[1] * direct_depth[0],
            reference_depth[0] * direct_depth[0]
            + reference_depth[1] * direct_depth[1],
        )
        rotation_deg = degrees(rotation_rad)
        config = self.tracker.config
        if (
            translation
            > max(
                0.0,
                float(
                    config.direct_pair_correction_max_translation_mm
                ),
            )
            or abs(rotation_deg)
            > max(
                0.0,
                float(config.direct_pair_correction_max_rotation_deg),
            )
        ):
            return None

        translation_alpha = min(
            1.0,
            max(0.0, float(config.direct_pair_correction_alpha)),
        )
        rotation_alpha = min(
            1.0,
            max(
                0.0,
                float(config.direct_pair_correction_rotation_alpha),
            ),
        )
        applied_rotation = rotation_rad * rotation_alpha
        cosine = cos(applied_rotation)
        sine = sin(applied_rotation)

        def rotate(axis: Point) -> Point:
            return (
                cosine * axis[0] - sine * axis[1],
                sine * axis[0] + cosine * axis[1],
            )

        corrected_origin = (
            reference_origin[0] + delta_x * translation_alpha,
            reference_origin[1] + delta_y * translation_alpha,
        )
        corrected_width = rotate(reference_width)
        corrected_depth = rotate(reference_depth)
        polygon = tuple(
            slot_local_to_point(
                local,
                corrected_origin,
                corrected_width,
                corrected_depth,
            )
            for local in self.slot_frame.local_polygon
        )
        return LockedSlotPose(
            polygon=polygon,  # type: ignore[arg-type]
            locked=True,
            tracked=True,
            source=SlotPoseSource.DIRECT_PAIR_CORRECTION,
            scan_timestamp=direct.scan_timestamp,
            translation_mm=translation * translation_alpha,
            rotation_deg=rotation_deg * rotation_alpha,
            support_points=direct.support_points,
            bilateral_support=True,
            reason="direct_pair_gated_correction",
        )

    def _slew_limited_direct_pair_recovery(
        self,
        reference: LockedSlotPose,
        direct: LockedSlotPose,
        *,
        reason: str = "direct_pair_slew_recovery",
    ) -> Optional[LockedSlotPose]:
        """Follow a large direct-pair delta without accepting it in one jump."""

        if reference.polygon is None or direct.polygon is None:
            return None
        reference_frame = slot_frame_from_polygon(reference.polygon)
        direct_frame = slot_frame_from_polygon(direct.polygon)
        if reference_frame is None or direct_frame is None:
            return None
        reference_origin, reference_width, reference_depth = reference_frame
        direct_origin, _direct_width, direct_depth = direct_frame
        delta_x = direct_origin[0] - reference_origin[0]
        delta_y = direct_origin[1] - reference_origin[1]
        translation = hypot(delta_x, delta_y)
        rotation_rad = atan2(
            reference_depth[0] * direct_depth[1]
            - reference_depth[1] * direct_depth[0],
            reference_depth[0] * direct_depth[0]
            + reference_depth[1] * direct_depth[1],
        )
        rotation_deg = degrees(rotation_rad)
        config = self.tracker.config
        translation_limit = max(
            0.0,
            float(config.direct_pair_slew_translation_per_scan_mm),
        )
        rotation_limit_deg = max(
            0.0,
            float(config.direct_pair_slew_rotation_per_scan_deg),
        )
        translation_scale = (
            1.0
            if translation <= translation_limit or translation <= 1e-9
            else translation_limit / translation
        )
        rotation_scale = (
            1.0
            if abs(rotation_deg) <= rotation_limit_deg
            or abs(rotation_deg) <= 1e-9
            else rotation_limit_deg / abs(rotation_deg)
        )
        applied_rotation = rotation_rad * rotation_scale
        cosine = cos(applied_rotation)
        sine = sin(applied_rotation)

        def rotate(axis: Point) -> Point:
            return (
                cosine * axis[0] - sine * axis[1],
                sine * axis[0] + cosine * axis[1],
            )

        recovered_origin = (
            reference_origin[0] + delta_x * translation_scale,
            reference_origin[1] + delta_y * translation_scale,
        )
        recovered_width = rotate(reference_width)
        recovered_depth = rotate(reference_depth)
        polygon = tuple(
            slot_local_to_point(
                local,
                recovered_origin,
                recovered_width,
                recovered_depth,
            )
            for local in self.slot_frame.local_polygon
        )
        return LockedSlotPose(
            polygon=polygon,  # type: ignore[arg-type]
            locked=True,
            tracked=True,
            source=SlotPoseSource.DIRECT_PAIR_RECOVERY,
            scan_timestamp=direct.scan_timestamp,
            translation_mm=translation * translation_scale,
            rotation_deg=rotation_deg * rotation_scale,
            support_points=direct.support_points,
            bilateral_support=True,
            reason=reason,
        )

    def _short_icp_fallback_or_lost(
        self,
        points: Sequence[Point],
        *,
        scan_timestamp: Optional[float],
        map_reason: str,
    ) -> LockedSlotPose:
        maximum = max(
            0,
            int(self.tracker.config.map_icp_fallback_max_scans),
        )
        if self._icp_fallback_scans >= maximum:
            self.tracker.reset()
            return self._lost_pose(
                scan_timestamp,
                "short_icp_fallback_exhausted",
            )
        if not self.tracker.locked:
            return self._lost_pose(
                scan_timestamp,
                f"all_slot_localization_failed:{map_reason}:"
                "tracker_unavailable",
            )
        fallback = self.tracker.update(points)
        if not fallback.tracked or fallback.polygon is None:
            self.tracker.reset()
            return self._lost_pose(
                scan_timestamp,
                f"all_slot_localization_failed:{map_reason}:"
                f"{fallback.reason}",
            )
        self._icp_fallback_scans += 1
        return LockedSlotPose(
            polygon=fallback.polygon,
            locked=True,
            tracked=True,
            source=SlotPoseSource.TRACKER_FALLBACK,
            scan_timestamp=scan_timestamp,
            translation_mm=fallback.translation_mm,
            rotation_deg=fallback.rotation_deg,
            reason="short_icp_fallback",
        )

    def _lost_pose(
        self,
        scan_timestamp: Optional[float],
        reason: str,
    ) -> LockedSlotPose:
        return LockedSlotPose(
            polygon=self.pose.polygon,
            locked=self.pose.locked,
            lost=True,
            source=SlotPoseSource.MAP_LOCALIZATION,
            scan_timestamp=scan_timestamp,
            reason=reason,
        )


def slot_frame_from_polygon(
    polygon: Polygon,
) -> Optional[Tuple[Point, Point, Point]]:
    entrance_center = (
        (polygon[0][0] + polygon[1][0]) / 2.0,
        (polygon[0][1] + polygon[1][1]) / 2.0,
    )
    width_axis = normalize_point(
        (
            polygon[1][0] - polygon[0][0],
            polygon[1][1] - polygon[0][1],
        )
    )
    depth_axis = normalize_point(
        (
            polygon[3][0] - polygon[0][0],
            polygon[3][1] - polygon[0][1],
        )
    )
    if width_axis is None or depth_axis is None:
        return None
    return entrance_center, width_axis, depth_axis


def point_to_slot_local(
    point: Point,
    origin: Point,
    width_axis: Point,
    depth_axis: Point,
) -> Point:
    relative = (point[0] - origin[0], point[1] - origin[1])
    return (
        relative[0] * width_axis[0] + relative[1] * width_axis[1],
        relative[0] * depth_axis[0] + relative[1] * depth_axis[1],
    )


def nearest_landmark_indexes(
    points: np.ndarray,
    landmarks: Sequence[SlotLandmark],
) -> np.ndarray:
    if len(points) == 0 or not landmarks:
        return np.empty((0,), dtype=int)
    landmark_points = np.asarray(
        [
            (landmark.local_x_mm, landmark.local_y_mm)
            for landmark in landmarks
        ],
        dtype=np.float64,
    )
    differences = points[:, None, :] - landmark_points[None, :, :]
    distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
    return np.argmin(distances_squared, axis=1)


def landmark_support_counts(
    points: np.ndarray,
    landmarks: Sequence[SlotLandmark],
) -> Tuple[int, ...]:
    indexes = nearest_landmark_indexes(points, landmarks)
    return tuple(
        int(np.count_nonzero(indexes == index))
        for index in range(len(landmarks))
    )


def localization_observability(
    points: np.ndarray,
    landmark_indexes: np.ndarray,
    *,
    minimum_points: int,
    minimum_points_per_landmark: int,
    minimum_depth_span_mm: float,
) -> Tuple[bool, bool, float, int]:
    """Check whether current returns constrain translation along slot depth.

    One visible car needs a sufficiently long depth surface. When both frozen
    bordering-car landmarks have support, their separation anchors the rigid
    transform even if each individual panel is short in the depth direction.
    """

    depth_span = (
        float(np.ptp(points[:, 1]))
        if len(points) > 1
        else 0.0
    )
    per_landmark = max(1, int(minimum_points_per_landmark))
    support_counts = tuple(
        int(np.count_nonzero(landmark_indexes == index))
        for index in range(2)
    )
    bilateral = (
        len(support_counts) == 2
        and all(count >= per_landmark for count in support_counts)
    )
    required = (
        2 * per_landmark
        if bilateral
        else max(3, int(minimum_points))
    )
    observable = (
        len(points) >= required
        and (
            bilateral
            or depth_span
            >= max(0.0, float(minimum_depth_span_mm))
        )
    )
    return observable, bilateral, depth_span, required


def slot_local_to_point(
    local: Point,
    origin: Point,
    width_axis: Point,
    depth_axis: Point,
) -> Point:
    return (
        origin[0] + local[0] * width_axis[0] + local[1] * depth_axis[0],
        origin[1] + local[0] * width_axis[1] + local[1] * depth_axis[1],
    )


def normalize_point(point: Point) -> Optional[Point]:
    length = hypot(point[0], point[1])
    if length <= 1e-9:
        return None
    return point[0] / length, point[1] / length


def estimate_current_to_previous_transform(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    max_correspondence_mm: float,
    trim_ratio: float,
    iterations: int,
    min_correspondences: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Trimmed point-to-point ICP returning ``previous ~= R*current + t``."""

    if current.ndim != 2 or previous.ndim != 2 or current.shape[1:] != (2,) or previous.shape[1:] != (2,):
        return None
    if len(current) < min_correspondences or len(previous) < min_correspondences:
        return None

    rotation = np.eye(2, dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    keep_ratio = min(1.0, max(0.25, float(trim_ratio)))
    max_distance_squared = float(max_correspondence_mm) ** 2

    for _ in range(max(1, int(iterations))):
        transformed = current @ rotation.T + translation
        differences = transformed[:, None, :] - previous[None, :, :]
        distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
        nearest_indexes = np.argmin(distances_squared, axis=1)
        nearest_distances = distances_squared[np.arange(len(transformed)), nearest_indexes]
        valid_indexes = np.flatnonzero(nearest_distances <= max_distance_squared)
        if len(valid_indexes) < min_correspondences:
            return None
        keep_count = max(min_correspondences, int(round(len(valid_indexes) * keep_ratio)))
        ordered = valid_indexes[np.argsort(nearest_distances[valid_indexes])[:keep_count]]
        source = transformed[ordered]
        target = previous[nearest_indexes[ordered]]
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        u, _, vt = np.linalg.svd(covariance)
        delta_rotation = vt.T @ u.T
        if np.linalg.det(delta_rotation) < 0.0:
            vt[-1, :] *= -1.0
            delta_rotation = vt.T @ u.T
        delta_translation = target_center - delta_rotation @ source_center
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation

    return rotation, translation
