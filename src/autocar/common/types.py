from dataclasses import dataclass


@dataclass(frozen=True)
class LaneInfo:
    center_offset_norm: float = 0.0
    confidence: float = 0.0
    lane_center_x: int = 0
    image_center_x: int = 0
    reason: str = ""


@dataclass(frozen=True)
class DriveCommand:
    speed: int = 0
    steering: int = 0
    brake: bool = False
    reason: str = ""

    @classmethod
    def stop(cls, reason: str) -> "DriveCommand":
        return cls(speed=0, steering=0, brake=True, reason=reason)

    def clipped(self, max_speed: int, max_steering: int) -> "DriveCommand":
        return DriveCommand(
            speed=_clip(self.speed, -max_speed, max_speed),
            steering=_clip(self.steering, -max_steering, max_steering),
            brake=self.brake,
            reason=self.reason,
        )


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
