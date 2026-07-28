from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..estimation.locked_slot import LockedSlotPose
from ..estimation.parking_lidar import LidarParkingObservation
from ..types import ControlCommand


@dataclass(frozen=True)
class MotionSafetyConfig:
    lease_duration_s: float = 0.30
    pose_stale_after_s: float = 0.35


@dataclass(frozen=True)
class MotionSafetySnapshot:
    """Read-only observability for the final motor-output safety gate."""

    pose_required: bool
    lease_valid: bool
    lease_expires_at: Optional[float]
    lease_remaining_s: Optional[float]
    last_fresh_pose_at: Optional[float]
    pose_age_s: Optional[float]


class MotionLease:
    """Short-lived permission to move using the latest valid position pose."""

    def __init__(self, duration_s: float) -> None:
        self.duration_s = max(0.0, float(duration_s))
        self._expires_at: Optional[float] = None

    @property
    def expires_at(self) -> Optional[float]:
        return self._expires_at

    def renew(self, now: float) -> None:
        self._expires_at = float(now) + self.duration_s

    def revoke(self) -> None:
        self._expires_at = None

    def valid(self, now: float) -> bool:
        return (
            self._expires_at is not None
            and float(now) <= self._expires_at
        )


class MotionSafetyGate:
    """Final common safety gate applied immediately before motor output."""

    def __init__(
        self,
        config: MotionSafetyConfig = MotionSafetyConfig(),
    ) -> None:
        self.config = config
        self.lease = MotionLease(config.lease_duration_s)
        self._last_fresh_pose_at: Optional[float] = None

    def reset(self) -> None:
        self.lease.revoke()
        self._last_fresh_pose_at = None

    def snapshot(
        self,
        now: float,
        *,
        pose_required: bool,
    ) -> MotionSafetySnapshot:
        expires_at = self.lease.expires_at
        last_fresh_pose_at = self._last_fresh_pose_at
        return MotionSafetySnapshot(
            pose_required=pose_required,
            lease_valid=self.lease.valid(now),
            lease_expires_at=expires_at,
            lease_remaining_s=(
                None
                if expires_at is None
                else max(0.0, expires_at - float(now))
            ),
            last_fresh_pose_at=last_fresh_pose_at,
            pose_age_s=(
                None
                if last_fresh_pose_at is None
                else max(0.0, float(now) - last_fresh_pose_at)
            ),
        )

    def apply(
        self,
        command: ControlCommand,
        *,
        now: float,
        lidar: LidarParkingObservation,
        pose: LockedSlotPose,
        pose_required: bool,
    ) -> ControlCommand:
        fresh_pose = (
            pose_required
            and lidar.is_new_scan
            and pose.locked
            and pose.tracked
            and pose.polygon is not None
            and not pose.stale
            and not pose.held
            and not pose.lost
        )
        if fresh_pose:
            self._last_fresh_pose_at = float(now)
            self.lease.renew(now)

        if not lidar.valid:
            self.reset()
            return ControlCommand.stop(
                "lidar_unavailable:%s" % lidar.reason
            )

        if not pose_required:
            return command

        if pose.lost:
            self.reset()
            return ControlCommand.stop("motion_safety:pose_lost")
        if pose.held:
            self.reset()
            return ControlCommand.stop("motion_safety:pose_held")
        if pose.stale:
            self.reset()
            return ControlCommand.stop("motion_safety:pose_stale")
        if not pose.locked or not pose.tracked or pose.polygon is None:
            self.reset()
            return ControlCommand.stop("motion_safety:pose_unavailable")

        if self._pose_age_expired(now):
            self.lease.revoke()
            return ControlCommand.stop("motion_safety:pose_stale")
        if not self.lease.valid(now):
            return ControlCommand.stop("motion_safety:lease_expired")
        return command

    def _pose_age_expired(self, now: float) -> bool:
        if self._last_fresh_pose_at is None:
            return True
        maximum_age = max(0.0, float(self.config.pose_stale_after_s))
        return float(now) - self._last_fresh_pose_at > maximum_age
