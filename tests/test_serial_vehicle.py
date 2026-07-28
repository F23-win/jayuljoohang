import unittest

from skku_autocar.control.serial_vehicle import parse_ultrasonic_line


class SerialVehicleUltrasonicTest(unittest.TestCase):
    def test_parses_full_ultrasonic_stream_line_and_rejects_zero_echo(self):
        sample = parse_ultrasonic_line("US FR=410 FL=0 SR=725 SL=680")

        self.assertIsNotNone(sample)
        self.assertEqual(sample.front_right_mm, 410.0)
        self.assertIsNone(sample.front_left_mm)
        self.assertEqual(sample.side_right_mm, 725.0)
        self.assertEqual(sample.side_left_mm, 680.0)

    def test_non_ultrasonic_line_is_ignored(self):
        self.assertIsNone(parse_ultrasonic_line("OK DRIVE"))

if __name__ == "__main__":
    unittest.main()
