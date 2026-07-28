import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from skku_autocar.estimation.locked_slot import (
    FrozenSlotLandmarkMap,
    LockedSlotPose,
    SlotLandmark,
    SlotPoseSource,
)
from skku_autocar.control.motion_safety import MotionSafetySnapshot
from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.planning.reverse_parking_path import (
    ReversePath,
    ReversePathStatus,
)
from skku_autocar.runtime.parking_app import (
    DashboardVideoRecorder,
    compose_parking_dashboard,
    dashboard_recording_enabled,
    format_locked_slot_pose_status,
    format_motion_safety_status,
    format_reverse_path_status,
    format_slot_footprint_status,
    format_slot_landmark_map_status,
    parse_args,
    timestamped_dashboard_path,
)


class ParkingDashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest("OpenCV/NumPy unavailable") from exc
        cls.cv2 = cv2
        cls.np = np

    def test_live_camera_records_by_default(self):
        args = parse_args([])
        self.assertEqual(args.record_dashboard, "auto")
        self.assertEqual(args.parking_record_dir, "data/parking")
        self.assertEqual(args.dashboard_record_fps, 10.0)
        self.assertEqual(args.record_session, "auto")
        self.assertEqual(args.session_record_dir, "data/parking_sessions")
        self.assertFalse(args.record_camera)
        self.assertTrue(dashboard_recording_enabled("auto", is_video=False))
        self.assertFalse(dashboard_recording_enabled("auto", is_video=True))
        self.assertTrue(dashboard_recording_enabled("on", is_video=True))
        self.assertFalse(dashboard_recording_enabled("off", is_video=False))

    def test_record_camera_is_explicit_and_does_not_enable_yolo(self):
        args = parse_args(["--no-camera", "--record-camera"])

        self.assertFalse(args.camera_enabled)
        self.assertTrue(args.record_camera)

    def test_timestamp_path_avoids_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 7, 21, 14, 30, 45)
            first = timestamped_dashboard_path(directory, now)
            self.assertEqual(first.name, "20260721_143045.mp4")
            first.touch()
            second = timestamped_dashboard_path(directory, now)
            self.assertEqual(second.name, "20260721_143045_01.mp4")

    def test_shared_dashboard_is_1280_by_720(self):
        rear = self.np.zeros((480, 640, 3), dtype=self.np.uint8)
        bev = self.np.zeros((640, 640, 3), dtype=self.np.uint8)
        lidar = self.np.zeros((600, 600, 3), dtype=self.np.uint8)
        dashboard = compose_parking_dashboard(
            self.cv2,
            self.np,
            rear,
            bev,
            lidar,
            "LIVE idle",
            (255, 255, 255),
            ("REC=test.mp4",),
        )
        self.assertEqual(dashboard.shape, (720, 1280, 3))

    def test_dashboard_reports_frozen_landmark_map_and_surface_count(self):
        landmark_map = FrozenSlotLandmarkMap(
            source_scan_timestamp=12.5,
            slot_width_mm=950.0,
            slot_depth_mm=1500.0,
            landmarks=(
                SlotLandmark("border_car_first", -650.0, 0.0),
                SlotLandmark("border_car_second", 650.0, 0.0),
            ),
        )

        self.assertEqual(
            format_slot_landmark_map_status(None),
            "UNFROZEN",
        )
        self.assertEqual(
            format_slot_landmark_map_status(landmark_map),
            (
                "FROZEN scan=12.500 size=950x1500mm landmarks=2 "
                "surfaces=0 (direct_pair_landmarks_frozen)"
            ),
        )

    def test_dashboard_reports_pose_path_and_motion_lease(self):
        pose_status = format_locked_slot_pose_status(
            LockedSlotPose(
                polygon=(
                    (-475.0, 0.0),
                    (475.0, 0.0),
                    (475.0, 1500.0),
                    (-475.0, 1500.0),
                ),
                locked=True,
                tracked=True,
                source=SlotPoseSource.MAP_LOCALIZATION,
                scan_timestamp=4.2,
                translation_mm=12.5,
                rotation_deg=-1.25,
                reason="slot_map_localized",
            )
        )
        path_status = format_reverse_path_status(
            ReversePath(
                found=True,
                points=((10.0, 20.0), (11.0, 21.0)),
                lookahead_point=(11.0, 21.0),
                curvature_per_px=0.001,
                maximum_curvature_per_px=0.0054,
                reason="reverse_path_ready",
                status=ReversePathStatus.READY,
            )
        )
        lease_status = format_motion_safety_status(
            MotionSafetySnapshot(
                pose_required=True,
                lease_valid=True,
                lease_expires_at=4.5,
                lease_remaining_s=0.3,
                last_fresh_pose_at=4.2,
                pose_age_s=0.0,
            )
        )

        self.assertIn("MAP_LOCALIZATION", pose_status)
        self.assertIn("scan=4.200", pose_status)
        self.assertIn("ready found=Y pts=2", path_status)
        self.assertIn("look=(11.0,21.0)", path_status)
        self.assertIn("required=Y valid=Y", lease_status)
        self.assertIn("remain=0.300s", lease_status)

    def test_dashboard_reports_fixed_slot_footprint_candidate(self):
        status = format_slot_footprint_status(
            ParkingGeometry(
                vehicle_footprint_min_lateral_mm=-300.0,
                vehicle_footprint_max_lateral_mm=300.0,
                vehicle_footprint_min_depth_mm=100.0,
                vehicle_footprint_max_depth_mm=1100.0,
                park_completion_candidate=True,
                park_completion_reason="footprint_inside_fixed_slot",
            )
        )

        self.assertIn("candidate=Y", status)
        self.assertIn("lat=[-30.0,+30.0]cm", status)
        self.assertIn("depth=[+10.0,+110.0]cm", status)

    def test_recorder_writes_readable_dashboard_mp4(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dashboard.mp4"
            recorder = DashboardVideoRecorder(self.cv2, output, fps=10.0)
            frame = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
            recorder.write(frame, 0.0)
            recorder.write(frame, 0.25)
            recorder.close()

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            capture = self.cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH)), 1280)
                self.assertEqual(int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT)), 720)
                self.assertGreaterEqual(
                    int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT)),
                    3,
                )
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
