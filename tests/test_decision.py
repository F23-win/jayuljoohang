import unittest

from autocar.common.types import LaneInfo
from autocar.decision.lane_follower import LaneFollower


class LaneFollowerTest(unittest.TestCase):
    def test_paused_returns_stop(self):
        follower = LaneFollower({})
        command = follower.decide(LaneInfo(confidence=1.0), enabled=False)
        self.assertTrue(command.brake)

    def test_lane_offset_becomes_steering(self):
        follower = LaneFollower({"steering_kp": 100, "base_speed": 70, "max_speed": 140, "max_steering": 120})
        command = follower.decide(LaneInfo(center_offset_norm=0.2, confidence=1.0), enabled=True)
        self.assertEqual(command.speed, 70)
        self.assertEqual(command.steering, 20)


if __name__ == "__main__":
    unittest.main()
