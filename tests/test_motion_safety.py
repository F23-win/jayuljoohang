import unittest
from dataclasses import replace

from skku_autocar.control.motion_safety import (
    MotionSafetyConfig,
    MotionSafetyGate,
)
from skku_autocar.estimation.locked_slot import (
    LockedSlotPose,
    SlotPoseSource,
)
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.types import ControlCommand


SLOT_POLYGON = (
    (-475.0, 0.0),
    (475.0, 0.0),
    (475.0, 1500.0),
    (-475.0, 1500.0),
)


def valid_lidar(*, is_new_scan=True):
    return LidarParkingObservation(
        timestamp=1.0,
        is_new_scan=is_new_scan,
        valid=True,
        reason="gap_confirmed",
    )


def valid_pose():
    return LockedSlotPose(
        polygon=SLOT_POLYGON,
        locked=True,
        tracked=True,
        source=SlotPoseSource.DIRECT_PAIR,
        scan_timestamp=1.0,
        reason="direct_pair",
    )


class MotionSafetyGateTest(unittest.TestCase):
    def setUp(self):
        self.moving = ControlCommand(
            speed=-30,
            steering=120,
            reason="following_entry_curve",
        )

    def test_fresh_pose_grants_motion_lease(self):
        gate = MotionSafetyGate(
            MotionSafetyConfig(
                lease_duration_s=0.30,
                pose_stale_after_s=0.35,
            )
        )

        command = gate.apply(
            self.moving,
            now=1.0,
            lidar=valid_lidar(),
            pose=valid_pose(),
            pose_required=True,
        )

        self.assertEqual(command, self.moving)
        self.assertTrue(gate.lease.valid(1.30))
        snapshot = gate.snapshot(1.10, pose_required=True)
        self.assertTrue(snapshot.pose_required)
        self.assertTrue(snapshot.lease_valid)
        self.assertAlmostEqual(snapshot.lease_expires_at, 1.30)
        self.assertAlmostEqual(snapshot.lease_remaining_s, 0.20)
        self.assertAlmostEqual(snapshot.last_fresh_pose_at, 1.0)
        self.assertAlmostEqual(snapshot.pose_age_s, 0.10)

    def test_stale_held_or_lost_pose_stops_in_same_call(self):
        for field, reason in (
            ("stale", "motion_safety:pose_stale"),
            ("held", "motion_safety:pose_held"),
            ("lost", "motion_safety:pose_lost"),
        ):
            with self.subTest(field=field):
                gate = MotionSafetyGate()
                pose = replace(valid_pose(), **{field: True})

                command = gate.apply(
                    self.moving,
                    now=1.0,
                    lidar=valid_lidar(),
                    pose=pose,
                    pose_required=True,
                )

                self.assertEqual(command.speed, 0)
                self.assertTrue(command.brake)
                self.assertEqual(command.reason, reason)

    def test_expired_lease_stops_duplicate_pose_in_same_call(self):
        gate = MotionSafetyGate(
            MotionSafetyConfig(
                lease_duration_s=0.30,
                pose_stale_after_s=1.0,
            )
        )
        gate.apply(
            self.moving,
            now=1.0,
            lidar=valid_lidar(),
            pose=valid_pose(),
            pose_required=True,
        )

        command = gate.apply(
            self.moving,
            now=1.31,
            lidar=valid_lidar(is_new_scan=False),
            pose=valid_pose(),
            pose_required=True,
        )

        self.assertEqual(command.speed, 0)
        self.assertTrue(command.brake)
        self.assertEqual(command.reason, "motion_safety:lease_expired")

    def test_pose_age_stops_motion_before_longer_lease_expires(self):
        gate = MotionSafetyGate(
            MotionSafetyConfig(
                lease_duration_s=1.0,
                pose_stale_after_s=0.20,
            )
        )
        gate.apply(
            self.moving,
            now=1.0,
            lidar=valid_lidar(),
            pose=valid_pose(),
            pose_required=True,
        )

        command = gate.apply(
            self.moving,
            now=1.21,
            lidar=valid_lidar(is_new_scan=False),
            pose=valid_pose(),
            pose_required=True,
        )

        self.assertEqual(command.speed, 0)
        self.assertEqual(command.reason, "motion_safety:pose_stale")

    def test_lidar_disconnect_stops_and_revokes_existing_lease(self):
        gate = MotionSafetyGate(
            MotionSafetyConfig(
                lease_duration_s=1.0,
                pose_stale_after_s=1.0,
            )
        )
        gate.apply(
            self.moving,
            now=1.0,
            lidar=valid_lidar(),
            pose=valid_pose(),
            pose_required=True,
        )

        disconnected = gate.apply(
            self.moving,
            now=1.1,
            lidar=LidarParkingObservation(
                reason="lidar_error:disconnected",
            ),
            pose=valid_pose(),
            pose_required=True,
        )

        self.assertEqual(disconnected.speed, 0)
        self.assertTrue(disconnected.brake)
        self.assertEqual(
            disconnected.reason,
            "lidar_unavailable:lidar_error:disconnected",
        )
        self.assertFalse(gate.lease.valid(1.1))

    def test_search_motion_does_not_require_slot_pose(self):
        gate = MotionSafetyGate()

        command = gate.apply(
            ControlCommand(speed=30, reason="searching"),
            now=1.0,
            lidar=valid_lidar(),
            pose=LockedSlotPose(),
            pose_required=False,
        )

        self.assertEqual(command.speed, 30)


if __name__ == "__main__":
    unittest.main()
