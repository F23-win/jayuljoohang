import tempfile
import unittest
from types import SimpleNamespace

from skku_autocar.sensors.lidar import find_lidar_port


def port(device, description="", hwid="", manufacturer=""):
    return SimpleNamespace(
        device=device,
        description=description,
        hwid=hwid,
        manufacturer=manufacturer,
    )


class LidarPortDetectionTest(unittest.TestCase):
    def test_prefers_usbserial_lidar_over_usbmodem_arduino(self):
        detected = find_lidar_port(
            ports=(
                port("/dev/cu.usbmodem11101", "IOUSBHostDevice", "USB VID:PID=2341:0042"),
                port("/dev/cu.usbserial-1410", "USB Serial", "USB VID:PID=1A86:7523"),
            ),
            exists=lambda _: False,
        )

        self.assertEqual(detected, "/dev/cu.usbserial-1410")

    def test_stale_explicit_port_falls_back_to_detected_lidar(self):
        detected = find_lidar_port(
            "/dev/cu.usbserial-1120",
            ports=(
                port("/dev/cu.usbserial-1410", "USB Serial", "USB VID:PID=1A86:7523"),
            ),
            exists=lambda _: False,
        )

        self.assertEqual(detected, "/dev/cu.usbserial-1410")

    def test_existing_explicit_port_is_kept(self):
        with tempfile.NamedTemporaryFile() as handle:
            detected = find_lidar_port(
                handle.name,
                ports=(
                    port("/dev/cu.usbserial-1410", "USB Serial", "USB VID:PID=1A86:7523"),
                ),
            )

        self.assertEqual(detected, handle.name)

    def test_windows_com_port_is_kept_without_filesystem_entry(self):
        detected = find_lidar_port(
            "COM7",
            ports=(port("COM9", "USB Serial"),),
            exists=lambda _: False,
        )

        self.assertEqual(detected, "COM7")


if __name__ == "__main__":
    unittest.main()
