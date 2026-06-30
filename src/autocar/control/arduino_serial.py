from autocar.common.types import DriveCommand
from autocar.control.protocol import encode_command
import time


class ArduinoSerial:
    """Serial transport for Arduino. This layer only sends commands."""

    def __init__(self, config):
        self.config = config
        self.serial = None

    def open(self):
        port = self.config.get("serial_port")
        if not port:
            raise ValueError("control.serial_port is not configured")

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required. Run: pip install -r requirements.txt") from exc

        self.serial = serial.Serial(
            port,
            int(self.config.get("baudrate", 115200)),
            timeout=0.05,
        )
        # Opening the serial port resets many Arduino boards. Give setup() time to finish.
        time.sleep(float(self.config.get("startup_wait_s", 2.0)))
        return self

    def send(self, command: DriveCommand):
        if self.serial is None:
            raise RuntimeError("serial is not open")
        self.serial.write(encode_command(command).encode("ascii"))

    def close(self):
        if self.serial is not None:
            self.send(DriveCommand.stop("shutdown"))
            self.serial.close()
            self.serial = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
