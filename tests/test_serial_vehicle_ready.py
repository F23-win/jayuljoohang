import unittest

from skku_autocar.control.serial_vehicle import (
    SerialVehicleClient,
    SerialVehicleConfig,
)


class PingResponsiveSerial:
    def __init__(self):
        self.writes = []
        self.responses = []

    def write(self, value):
        self.writes.append(value)
        if value == b"PING\n":
            self.responses.append(b"OK PONG\n")

    def readline(self):
        return self.responses.pop(0) if self.responses else b""


class SerialVehicleReadyTest(unittest.TestCase):
    def test_wait_ready_pings_already_running_firmware(self):
        client = SerialVehicleClient(
            SerialVehicleConfig(ready_timeout_s=0.1)
        )
        serial = PingResponsiveSerial()
        client._serial = serial

        client._wait_ready()

        self.assertIn(b"PING\n", serial.writes)


if __name__ == "__main__":
    unittest.main()
