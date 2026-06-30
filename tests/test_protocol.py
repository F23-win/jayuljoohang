import unittest

from autocar.common.types import DriveCommand
from autocar.control.protocol import encode_command


class ProtocolTest(unittest.TestCase):
    def test_stop(self):
        self.assertEqual(encode_command(DriveCommand.stop("test")), "STOP\n")

    def test_drive(self):
        self.assertEqual(encode_command(DriveCommand(speed=80, steering=-12)), "DRIVE 80 -12\n")


if __name__ == "__main__":
    unittest.main()
