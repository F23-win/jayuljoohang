import tempfile
import unittest
from pathlib import Path

from skku_autocar.runtime.obstacle_mode import ObstacleDriveMode
from skku_autocar.runtime.yolo_drive_app import parse_args
from skku_autocar.sensors.ultrasonic_replay import UltrasonicFrontTrace


class UltrasonicFrontTraceTests(unittest.TestCase):
    def test_loads_frame_aligned_front_values_and_rejects_ocr_outliers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "frame,front_mm\n"
                "0,\n"
                "1,2205\n"
                "2,14488\n"
                "3,0\n",
                encoding="utf-8",
            )

            trace = UltrasonicFrontTrace.from_csv(str(path))

        self.assertEqual(trace.values, (None, 2205, None, None))
        self.assertIsNone(trace.front_mm(-1))
        self.assertIsNone(trace.front_mm(10))

    def test_replay_front_preserves_three_sensor_quorum(self):
        args = parse_args([])
        mode = ObstacleDriveMode(args, object(), object())

        mode.accept_replay_front(1537)
        snapshot = mode._ultrasonic_replay

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.front_min_mm, 1537)
        self.assertEqual(snapshot.front_fresh_count, 3)

        mode.accept_replay_front(None)
        snapshot = mode._ultrasonic_replay
        self.assertIsNone(snapshot.front_min_mm)
        self.assertEqual(snapshot.front_fresh_count, 3)


if __name__ == "__main__":
    unittest.main()
