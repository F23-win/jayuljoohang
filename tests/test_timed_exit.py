from dataclasses import replace
import unittest

from skku_autocar.config import PaperControllerConfig, RearLidarConfig
from skku_autocar.perception.rear_lidar import (
    RearLidarObservation,
    RearLidarPerception,
)
from skku_autocar.planning.paper_controller import PaperParkingController
from skku_autocar.sensors.lidar import LidarPoint, LidarScan
from skku_autocar.types import ParkingState


class TimedExitTest(unittest.TestCase):
    def test_perception_preserves_closest_cd_y_coordinates(self):
        perception = RearLidarPerception(RearLidarConfig())
        observation = perception.observe(
            LidarScan(
                timestamp=1.0,
                points=(
                    LidarPoint(15, 350.0, 1000.0),
                    LidarPoint(15, 190.0, 1000.0),
                ),
            )
        )

        self.assertAlmostEqual(observation.c_y_back_mm, 173.65, places=2)
        self.assertAlmostEqual(observation.d_y_back_mm, 173.65, places=2)

    def test_parking_waits_for_reverse_straight_and_fresh_confirmation(self):
        config = replace(
            PaperControllerConfig(),
            park_y_threshold_mm=0.0,
            park_y_cross_confirm_scans=2,
            park_heading_tolerance_deg=4.0,
            park_heading_stability_deg=3.0,
            park_confirm_scans=3,
            center_observation_scans=2,
            cd_center_confirm_scans=1,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.CENTER_CHECK

        for timestamp in (1.0, 2.0):
            command = controller.update(
                RearLidarObservation(
                    timestamp=timestamp,
                    valid=True,
                    dist_c_mm=700.0,
                    dist_d_mm=850.0,
                    c_y_back_mm=-10.0,
                    d_y_back_mm=-5.0,
                    slot_heading_deg=2.0,
                ),
                timestamp,
            )

        self.assertEqual(controller.state, ParkingState.REVERSE_STRAIGHT)
        self.assertEqual(command.speed, config.inside_reverse_speed)

        for timestamp in (3.0, 3.0, 4.0):
            command = controller.update(
                RearLidarObservation(
                    timestamp=timestamp,
                    valid=True,
                    dist_c_mm=700.0,
                    dist_d_mm=850.0,
                    c_y_back_mm=None,
                    d_y_back_mm=None,
                    slot_heading_deg=2.0,
                ),
                timestamp,
            )
            self.assertEqual(
                controller.state,
                ParkingState.REVERSE_STRAIGHT,
            )

        command = controller.update(
            RearLidarObservation(
                timestamp=5.0,
                valid=True,
                dist_c_mm=700.0,
                dist_d_mm=850.0,
                c_y_back_mm=None,
                d_y_back_mm=None,
                slot_heading_deg=2.0,
            ),
            5.0,
        )

        self.assertEqual(controller.state, ParkingState.PARKED)
        self.assertTrue(command.brake)
        self.assertEqual(
            command.reason,
            "paper_park_confirmed C_crossed=1 D_crossed=1 "
            "heading=+2.0 scans=3",
        )

    def test_cd_crossings_are_latched_independently(self):
        config = replace(
            PaperControllerConfig(),
            center_observation_scans=2,
            park_y_threshold_mm=0.0,
            park_y_cross_confirm_scans=2,
            park_confirm_scans=2,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.REVERSE_STRAIGHT

        samples = (
            (1.0, 700.0, 850.0, -5.0, 20.0),
            (2.0, 700.0, 850.0, -6.0, 15.0),
            (3.0, None, 850.0, None, -1.0),
            (4.0, None, 850.0, None, -2.0),
            (5.0, None, None, None, None),
        )
        for timestamp, dist_c, dist_d, c_y, d_y in samples:
            command = controller.update(
                RearLidarObservation(
                    timestamp=timestamp,
                    valid=True,
                    dist_c_mm=dist_c,
                    dist_d_mm=dist_d,
                    c_y_back_mm=c_y,
                    d_y_back_mm=d_y,
                    slot_heading_deg=0.0,
                ),
                timestamp,
            )

        self.assertEqual(controller.state, ParkingState.PARKED)
        self.assertTrue(command.brake)

    def test_reverse_straight_safety_does_not_restart_forward_recovery(self):
        config = replace(
            PaperControllerConfig(),
            center_observation_scans=2,
            dist_bias_cd_threshold_mm=100.0,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.REVERSE_STRAIGHT

        for timestamp in (1.0, 2.0):
            command = controller.update(
                RearLidarObservation(
                    timestamp=timestamp,
                    valid=True,
                    dist_c_mm=500.0,
                    dist_d_mm=800.0,
                    c_y_back_mm=100.0,
                    d_y_back_mm=100.0,
                    slot_heading_deg=0.0,
                ),
                timestamp,
            )

        self.assertEqual(controller.state, ParkingState.REVERSE_STRAIGHT)
        self.assertEqual(command.speed, 0)
        self.assertTrue(command.brake)
        self.assertEqual(
            command.reason,
            "reverse_straight_cd_bias_safety_stop",
        )

    def test_center_alignment_steers_from_heading_not_cd_balance(self):
        config = replace(
            PaperControllerConfig(),
            actuator_steering_offset=0,
            dist_bias_cd_threshold_mm=220.0,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.CENTER_CHECK

        for timestamp in (1.0, 2.0, 3.0, 4.0):
            command = controller.update(
                RearLidarObservation(
                    timestamp=timestamp,
                    valid=True,
                    dist_c_mm=700.0,
                    dist_d_mm=880.0,
                    c_y_back_mm=0.0,
                    d_y_back_mm=0.0,
                    slot_heading_deg=0.0,
                ),
                timestamp,
            )

        self.assertEqual(controller.state, ParkingState.CENTER_CHECK)
        self.assertEqual((command.speed, command.steering), (-50, 0))
        self.assertIn("heading_reverse_align", command.reason)

    def test_parked_runs_forward_right_then_straight_without_lidar(self):
        config = replace(
            PaperControllerConfig(),
            park_hold_s=4.0,
            exit_speed=50,
            exit_forward_s=4.0,
            exit_turn_steering=150,
            exit_turn_right_s=5.0,
            actuator_steering_offset=-28,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.PARKED
        controller._parked_started_at = 0.0
        no_lidar = RearLidarObservation(timestamp=0.0, valid=False)

        command = controller.update(no_lidar, 3.99)
        self.assertTrue(command.brake)

        command = controller.update(no_lidar, 4.0)
        self.assertEqual(controller.state, ParkingState.EXIT_FORWARD)
        self.assertEqual((command.speed, command.steering), (50, -28))

        command = controller.update(no_lidar, 8.0)
        self.assertEqual(controller.state, ParkingState.EXIT_TURN_RIGHT)
        self.assertEqual((command.speed, command.steering), (50, 150))

        command = controller.update(no_lidar, 13.0)
        self.assertEqual(controller.state, ParkingState.EXIT_STRAIGHT)
        self.assertEqual((command.speed, command.steering), (50, -28))

        command = controller.update(no_lidar, 20.0)
        self.assertEqual((command.speed, command.steering), (50, -28))


if __name__ == "__main__":
    unittest.main()
