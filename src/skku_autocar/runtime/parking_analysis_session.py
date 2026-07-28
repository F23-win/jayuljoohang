from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..control.motion_safety import MotionSafetySnapshot
from ..estimation.locked_slot import FrozenSlotLandmarkMap, LockedSlotPose
from ..estimation.parking_geometry import ParkingGeometry
from ..estimation.parking_lidar import LidarParkingObservation
from ..planning.t_parking_planner import ParkingPlan
from ..sensors.lidar import LidarScan
from ..types import ControlCommand


ANALYSIS_SCHEMA_VERSION = 1


def analysis_recording_enabled(mode: str, is_replay: bool) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return not is_replay


def timestamped_session_directory(
    directory: str,
    *,
    now: Optional[datetime] = None,
) -> Path:
    root = Path(directory).expanduser().resolve()
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / ("%s_%02d" % (timestamp, suffix))
        suffix += 1
    return candidate


class ParkingAnalysisSession:
    """Persist one live parking run in replayable and analysis-friendly forms."""

    def __init__(
        self,
        directory: Path,
        *,
        config: Any,
        cli_args: Mapping[str, Any],
        started_at: Optional[datetime] = None,
    ) -> None:
        self.directory = directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.session_id = self.directory.name
        self.started_at = started_at or datetime.now().astimezone()
        if self.started_at.tzinfo is None:
            self.started_at = self.started_at.astimezone()

        self.dashboard_path = self.directory / (
            self.session_id + "_dashboard.mp4"
        )
        self.lidar_path = self.directory / (self.session_id + "_lidar.csv")
        self.telemetry_path = self.directory / (
            self.session_id + "_telemetry.csv"
        )
        self.metadata_path = self.directory / (
            self.session_id + "_metadata.json"
        )
        self.bundle_path = self.directory / (self.session_id + "_replay.zip")

        self._lidar_handle = self.lidar_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
            buffering=1,
        )
        self._lidar_writer = csv.DictWriter(
            self._lidar_handle,
            fieldnames=(
                "timestamp",
                "elapsed_s",
                "scan_index",
                "quality",
                "angle_deg",
                "distance_mm",
            ),
        )
        self._lidar_writer.writeheader()
        self._lidar_handle.flush()

        self._telemetry_handle = self.telemetry_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
            buffering=1,
        )
        self._telemetry_writer: Optional[csv.DictWriter] = None
        self._last_lidar_timestamp: Optional[float] = None
        self.lidar_scans = 0
        self.lidar_points = 0
        self.telemetry_rows = 0
        self._closed = False
        self._metadata_base = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ultrasonic_used": False,
            "config": json_safe(config),
            "cli_args": json_safe(dict(cli_args)),
            "files": {
                "dashboard_video": self.dashboard_path.name,
                "lidar_csv": self.lidar_path.name,
                "telemetry_csv": self.telemetry_path.name,
                "metadata_json": self.metadata_path.name,
                "replay_zip": self.bundle_path.name,
            },
        }
        self._write_metadata(
            {
                "status": "recording",
                "completed_cleanly": False,
            }
        )

    def record_lidar(
        self,
        scan: Optional[LidarScan],
        *,
        elapsed_s: float,
    ) -> bool:
        if self._closed or scan is None:
            return False
        timestamp = float(scan.timestamp)
        if (
            self._last_lidar_timestamp is not None
            and timestamp == self._last_lidar_timestamp
        ):
            return False

        scan_index = self.lidar_scans
        for point in scan.points:
            self._lidar_writer.writerow(
                {
                    "timestamp": "%.9f" % timestamp,
                    "elapsed_s": "%.6f" % max(0.0, float(elapsed_s)),
                    "scan_index": scan_index,
                    "quality": int(point.quality),
                    "angle_deg": "%.6f" % float(point.angle_deg),
                    "distance_mm": "%.3f" % float(point.distance_mm),
                }
            )
        self._lidar_handle.flush()
        self._last_lidar_timestamp = timestamp
        self.lidar_scans += 1
        self.lidar_points += len(scan.points)
        return True

    def record_telemetry(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            return
        normalized = {
            str(key): csv_value(value) for key, value in row.items()
        }
        if self._telemetry_writer is None:
            self._telemetry_writer = csv.DictWriter(
                self._telemetry_handle,
                fieldnames=tuple(normalized.keys()),
            )
            self._telemetry_writer.writeheader()
        self._telemetry_writer.writerow(normalized)
        self._telemetry_handle.flush()
        self.telemetry_rows += 1

    def close(
        self,
        *,
        final_state: str,
        final_reason: str,
        completed_cleanly: bool,
        dashboard_path: Optional[Path] = None,
    ) -> Optional[Path]:
        if self._closed:
            return self.bundle_path if self.bundle_path.exists() else None
        self._closed = True
        self._lidar_handle.close()
        self._telemetry_handle.close()

        video_path = dashboard_path or self.dashboard_path
        video_available = video_path.exists() and video_path.stat().st_size > 0
        bundle_available = video_available and self.lidar_scans > 0
        finished_at = datetime.now().astimezone()
        metadata = {
            "status": "complete" if completed_cleanly else "interrupted",
            "completed_cleanly": bool(completed_cleanly),
            "finished_at": finished_at.isoformat(),
            "duration_wall_s": round(
                (finished_at - self.started_at).total_seconds(),
                6,
            ),
            "final_state": final_state,
            "final_reason": final_reason,
            "lidar_scans": self.lidar_scans,
            "lidar_points": self.lidar_points,
            "telemetry_rows": self.telemetry_rows,
            "dashboard_video_available": video_available,
            "replay_bundle_available": bundle_available,
        }
        self._write_metadata(metadata)
        if not bundle_available:
            return None

        with zipfile.ZipFile(
            str(self.bundle_path),
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in (
                video_path,
                self.lidar_path,
                self.telemetry_path,
                self.metadata_path,
            ):
                archive.write(str(path), arcname=path.name)
        return self.bundle_path

    def _write_metadata(self, values: Mapping[str, Any]) -> None:
        document = dict(self._metadata_base)
        document.update(values)
        document["lidar_scans"] = self.lidar_scans
        document["lidar_points"] = self.lidar_points
        document["telemetry_rows"] = self.telemetry_rows
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)


def parking_telemetry_row(
    *,
    elapsed_s: float,
    lidar: LidarParkingObservation,
    geometry: ParkingGeometry,
    pose: LockedSlotPose,
    landmark_map: Optional[FrozenSlotLandmarkMap],
    plan: ParkingPlan,
    planned_command: ControlCommand,
    safety: MotionSafetySnapshot,
    safety_overrode_command: bool,
    pose_required: bool,
    icp_fallback_scans: int,
) -> Dict[str, Any]:
    path = plan.path
    raw_gap = lidar.raw_two_car_gap
    row: Dict[str, Any] = {
        "elapsed_s": rounded(elapsed_s, 6),
        "planner_state": plan.state.value,
        "plan_reason": plan.reason,
        "planned_speed": planned_command.speed,
        "planned_steering": planned_command.steering,
        "command_speed": plan.command.speed,
        "command_steering": plan.command.steering,
        "command_reason": plan.command.reason,
        "safety_overrode_command": int(safety_overrode_command),
        "lidar_scan_timestamp": rounded(lidar.timestamp, 9),
        "lidar_is_new_scan": int(lidar.is_new_scan),
        "lidar_valid": int(lidar.valid),
        "lidar_unsafe": int(lidar.unsafe),
        "lidar_observed_points": lidar.observed_points,
        "lidar_reason": lidar.reason,
        "car_count": lidar.car_count,
        "first_car_seen": int(lidar.first_car_seen),
        "first_car_confirmed": int(lidar.first_car_confirmed),
        "first_car_turn_reached": int(lidar.first_car_turn_reached),
        "first_car_edge_y_back_mm": rounded(
            lidar.first_car_slot_edge_y_back_mm
        ),
        "first_car_turn_error_mm": rounded(lidar.first_car_turn_error_mm),
        "gap_pair_observed": int(lidar.gap_pair_observed),
        "gap_single_cluster_observed": int(
            lidar.gap_single_cluster_observed
        ),
        "tracking_roi_expanded": int(lidar.tracking_roi_expanded),
        "first_car_gate_passed": int(lidar.first_car_gate_passed),
        "second_car_gate_armed": int(lidar.second_car_gate_armed),
        "ordered_second_car_pairing": int(
            lidar.ordered_second_car_pairing
        ),
        "gap_found": int(lidar.gap_found),
        "gap_confirmed": int(lidar.gap_confirmed),
        "gap_coasted": int(lidar.coasted),
        "gap_width_mm": rounded(lidar.gap_width_mm),
        "gap_center_x_right_mm": rounded(lidar.gap_center_x_right_mm),
        "gap_center_y_back_mm": rounded(lidar.gap_center_y_back_mm),
        "entry_target_y_back_mm": rounded(lidar.entry_target_y_back_mm),
        "entry_error_mm": rounded(lidar.entry_error_mm),
        "entry_reached": int(lidar.entry_reached),
        "slot_depth_x_right": rounded(lidar.slot_depth_x_right),
        "slot_depth_y_back": rounded(lidar.slot_depth_y_back),
        "raw_gap_scan_timestamp": rounded(
            raw_gap.scan_timestamp if raw_gap else None,
            9,
        ),
        "raw_gap_observed_width_mm": rounded(
            raw_gap.observed_width_mm if raw_gap else None
        ),
        "raw_gap_first_x_right_mm": rounded(
            raw_gap.first_edge_x_right_mm if raw_gap else None
        ),
        "raw_gap_first_y_back_mm": rounded(
            raw_gap.first_edge_y_back_mm if raw_gap else None
        ),
        "raw_gap_second_x_right_mm": rounded(
            raw_gap.second_edge_x_right_mm if raw_gap else None
        ),
        "raw_gap_second_y_back_mm": rounded(
            raw_gap.second_edge_y_back_mm if raw_gap else None
        ),
        "geometry_found": int(geometry.found),
        "geometry_has_side_pair": int(geometry.has_side_pair),
        "geometry_has_back_line": int(geometry.has_back_line),
        "geometry_confidence": rounded(geometry.confidence, 6),
        "geometry_coasted": int(geometry.coasted),
        "geometry_reason": geometry.reason,
        "geometry_lateral_error_px": rounded(geometry.lateral_error_px),
        "geometry_lateral_error_norm": rounded(
            geometry.lateral_error_norm,
            6,
        ),
        "geometry_heading_error_deg": rounded(
            geometry.heading_error_deg
        ),
        "geometry_depth_remaining_px": rounded(
            geometry.depth_remaining_px
        ),
        "geometry_slot_width_px": rounded(geometry.slot_width_px),
        "geometry_slot_depth_px": rounded(geometry.slot_depth_px),
        "completion_candidate": int(geometry.park_completion_candidate),
        "completion_reason": geometry.park_completion_reason,
        "footprint_min_lateral_mm": rounded(
            geometry.vehicle_footprint_min_lateral_mm
        ),
        "footprint_max_lateral_mm": rounded(
            geometry.vehicle_footprint_max_lateral_mm
        ),
        "footprint_min_depth_mm": rounded(
            geometry.vehicle_footprint_min_depth_mm
        ),
        "footprint_max_depth_mm": rounded(
            geometry.vehicle_footprint_max_depth_mm
        ),
        "pose_required": int(pose_required),
        "pose_source": pose.source.value,
        "pose_locked": int(pose.locked),
        "pose_tracked": int(pose.tracked),
        "pose_stale": int(pose.stale),
        "pose_held": int(pose.held),
        "pose_lost": int(pose.lost),
        "pose_scan_timestamp": rounded(pose.scan_timestamp, 9),
        "pose_translation_mm": rounded(pose.translation_mm),
        "pose_rotation_deg": rounded(pose.rotation_deg),
        "pose_support_points": pose.support_points,
        "pose_bilateral_support": int(pose.bilateral_support),
        "pose_depth_span_mm": rounded(pose.depth_span_mm),
        "pose_reason": pose.reason,
        "pose_polygon_mm": points_text(pose.polygon or ()),
        "map_frozen": int(bool(landmark_map and landmark_map.frozen)),
        "map_source_scan_timestamp": rounded(
            landmark_map.source_scan_timestamp if landmark_map else None,
            9,
        ),
        "map_landmark_count": (
            len(landmark_map.landmarks) if landmark_map else 0
        ),
        "map_surface_point_count": (
            len(landmark_map.surface_points_local) if landmark_map else 0
        ),
        "map_reason": landmark_map.reason if landmark_map else "",
        "map_icp_fallback_scans": icp_fallback_scans,
        "path_status": path.status.value if path else "",
        "path_found": int(bool(path and path.found)),
        "path_reason": path.reason if path else "",
        "path_curvature_per_px": rounded(
            path.curvature_per_px if path else None,
            9,
        ),
        "path_maximum_curvature_per_px": rounded(
            path.maximum_curvature_per_px if path else None,
            9,
        ),
        "path_entry_steering_ratio": rounded(
            path.entry_steering_ratio if path else None,
            6,
        ),
        "path_entry_heading_change_deg": rounded(
            path.entry_heading_change_deg if path else None,
        ),
        "path_final_lateral_offset_px": rounded(
            path.final_lateral_offset_px if path else None,
        ),
        "path_minimum_side_clearance_px": rounded(
            path.minimum_side_clearance_px if path else None,
        ),
        "path_maximum_entry_depth_px": rounded(
            path.maximum_entry_depth_px if path else None,
        ),
        "path_lookahead_x_px": rounded(
            path.lookahead_point[0]
            if path and path.lookahead_point
            else None
        ),
        "path_lookahead_y_px": rounded(
            path.lookahead_point[1]
            if path and path.lookahead_point
            else None
        ),
        "path_points_px": points_text(path.points if path else ()),
        "lease_valid": int(safety.lease_valid),
        "lease_expires_at_s": rounded(safety.lease_expires_at, 6),
        "lease_remaining_s": rounded(safety.lease_remaining_s, 6),
        "lease_last_fresh_pose_at_s": rounded(
            safety.last_fresh_pose_at,
            6,
        ),
        "lease_pose_age_s": rounded(safety.pose_age_s, 6),
    }
    return row


def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def points_text(points: Sequence[Tuple[float, float]]) -> str:
    return "|".join("%.3f:%.3f" % (point[0], point[1]) for point in points)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    return value


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
