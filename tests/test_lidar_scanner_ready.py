import unittest

from skku_autocar.sensors.lidar import LidarScan, wait_for_scanner_ready


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now


class WaitForScannerReadyTest(unittest.TestCase):
    def test_returns_as_soon_as_a_scan_is_available(self):
        wait_for_scanner_ready(
            latest=lambda: LidarScan(0.0, ()),
            error=lambda: None,
            timeout_s=5.0,
            sleep=lambda s: self.fail("should not sleep when already ready"),
            now=lambda: 0.0,
        )

    def test_raises_immediately_on_a_connection_error(self):
        with self.assertRaisesRegex(RuntimeError, "connection failed"):
            wait_for_scanner_ready(
                latest=lambda: None,
                error=lambda: RuntimeError("could not open port 'COM10'"),
                timeout_s=5.0,
                sleep=lambda s: self.fail("should not sleep after an error"),
                now=lambda: 0.0,
            )

    def test_times_out_when_neither_a_scan_nor_an_error_ever_arrives(self):
        clock = FakeClock()
        with self.assertRaisesRegex(RuntimeError, "did not produce a scan"):
            wait_for_scanner_ready(
                latest=lambda: None,
                error=lambda: None,
                timeout_s=1.0,
                sleep=clock.sleep,
                now=clock.monotonic,
            )


if __name__ == "__main__":
    unittest.main()
