from autocar.common.types import DriveCommand, LaneInfo


class LaneFollower:
    """Converts recognized lane information into speed and steering."""

    def __init__(self, config):
        self.config = config

    def decide(self, lane_info: LaneInfo, enabled: bool) -> DriveCommand:
        if not enabled:
            return DriveCommand.stop("paused")

        min_confidence = float(self.config.get("min_lane_confidence", 0.2))
        if lane_info.confidence < min_confidence:
            return DriveCommand.stop("lane_confidence_low")

        steering_kp = float(self.config.get("steering_kp", 95.0))
        steering_sign = int(self.config.get("steering_sign", 1))
        steering = int(round(steering_sign * steering_kp * lane_info.center_offset_norm))

        command = DriveCommand(
            speed=int(self.config.get("base_speed", 75)),
            steering=steering,
            brake=False,
            reason="lane_follow",
        )
        return command.clipped(
            max_speed=int(self.config.get("max_speed", 140)),
            max_steering=int(self.config.get("max_steering", 120)),
        )
