import csv
import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from skku_autocar.control.motion_safety import MotionSafetySnapshot
from skku_autocar.estimation.locked_slot import LockedSlotPose
from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.t_parking_planner import (
    ParkingPlan,
    ParkingState,
)
from skku_autocar.runtime.parking_analysis_session import (
    ParkingAnalysisSession,
    analysis_recording_enabled,
    parking_telemetry_row,
    timestamped_session_directory,
)
from skku_autocar.runtime.parking_app import extract_recording_zip
from skku_autocar.sensors.lidar import LidarPoint, LidarScan, load_lidar_csv
from skku_autocar.types import ControlCommand


class ParkingAnalysisSessionTest(unittest.TestCase):
    def test_auto_records_live_but_not_replay(self):
        self.assertTrue(analysis_recording_enabled("auto", is_replay=False))
        self.assertFalse(analysis_recording_enabled("auto", is_replay=True))
        self.assertTrue(analysis_recording_enabled("on", is_replay=True))
        self.assertFalse(analysis_recording_enabled("off", is_replay=False))

    def test_timestamped_directory_avoids_existing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 7, 28, 9, 30, 15)
            first = timestamped_session_directory(directory, now=now)
            first.mkdir()
            second = timestamped_session_directory(directory, now=now)

            self.assertEqual(first.name, "20260728_093015")
            self.assertEqual(second.name, "20260728_093015_01")

    def test_session_saves_replay_bundle_and_skips_duplicate_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "20260728_093015"
            session = ParkingAnalysisSession(
                session_dir,
                config={"runtime": {"camera_enabled": False}},
                cli_args={"serial": True, "lidar_port": "COM5"},
                started_at=datetime(2026, 7, 28, 9, 30, 15),
            )
            scan = LidarScan(
                1000.25,
                (
                    LidarPoint(15, 12.5, 850.0),
                    LidarPoint(20, 15.0, 900.0),
                ),
            )
            self.assertTrue(session.record_lidar(scan, elapsed_s=0.25))
            self.assertFalse(session.record_lidar(scan, elapsed_s=0.30))

            planned = ControlCommand(20, -150, reason="planned")
            plan = ParkingPlan(
                ParkingState.PROVISIONAL_PREALIGN,
                ControlCommand.stop("motion_safety:pose_lost"),
                "motion_safety:pose_lost",
            )
            row = parking_telemetry_row(
                elapsed_s=0.25,
                lidar=LidarParkingObservation(
                    timestamp=1000.25,
                    is_new_scan=True,
                    valid=True,
                    observed_points=2,
                    first_car_gate_passed=True,
                    second_car_gate_armed=True,
                    ordered_second_car_pairing=True,
                    reason="cars_not_found",
                ),
                geometry=ParkingGeometry(reason="slot_not_found"),
                pose=LockedSlotPose(reason="slot_not_locked"),
                landmark_map=None,
                plan=plan,
                planned_command=planned,
                safety=MotionSafetySnapshot(
                    pose_required=True,
                    lease_valid=False,
                    lease_expires_at=None,
                    lease_remaining_s=None,
                    last_fresh_pose_at=None,
                    pose_age_s=None,
                ),
                safety_overrode_command=True,
                pose_required=True,
                icp_fallback_scans=0,
            )
            session.record_telemetry(row)
            session.dashboard_path.write_bytes(b"test-video")
            bundle = session.close(
                final_state=ParkingState.REACQUIRE_SLOT.value,
                final_reason="operator_quit",
                completed_cleanly=True,
            )

            self.assertEqual(session.lidar_scans, 1)
            self.assertEqual(session.lidar_points, 2)
            self.assertEqual(session.telemetry_rows, 1)
            self.assertIsNotNone(bundle)
            self.assertTrue(bundle.exists())
            loaded = load_lidar_csv(str(session.lidar_path))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(loaded[0].points), 2)

            with session.telemetry_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                telemetry = list(csv.DictReader(handle))
            self.assertEqual(len(telemetry), 1)
            self.assertEqual(
                telemetry[0]["planner_state"],
                ParkingState.PROVISIONAL_PREALIGN.value,
            )
            self.assertEqual(telemetry[0]["safety_overrode_command"], "1")
            self.assertEqual(telemetry[0]["command_speed"], "0")
            self.assertEqual(telemetry[0]["first_car_gate_passed"], "1")
            self.assertEqual(telemetry[0]["second_car_gate_armed"], "1")
            self.assertEqual(
                telemetry[0]["ordered_second_car_pairing"],
                "1",
            )

            metadata = json.loads(
                session.metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["lidar_scans"], 1)
            self.assertEqual(metadata["telemetry_rows"], 1)
            self.assertFalse(metadata["ultrasonic_used"])
            self.assertTrue(metadata["replay_bundle_available"])

            with zipfile.ZipFile(str(bundle), "r") as archive:
                names = set(archive.namelist())
            self.assertEqual(
                names,
                {
                    session.dashboard_path.name,
                    session.lidar_path.name,
                    session.telemetry_path.name,
                    session.metadata_path.name,
                },
            )
            extracted_video, extracted_lidar = extract_recording_zip(
                bundle,
                Path(directory) / "extracted",
            )
            self.assertEqual(
                extracted_video.name,
                session.dashboard_path.name,
            )
            self.assertEqual(extracted_lidar.name, session.lidar_path.name)


if __name__ == "__main__":
    unittest.main()
