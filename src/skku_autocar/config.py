import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"


@dataclass(frozen=True)
class LaneConfig:
    gray_threshold: int = 175
    min_contour_area: int = 120
    roi_bottom_left_x: float = 0.18
    roi_bottom_right_x: float = 0.82
    roi_top_left_x: float = 0.05
    roi_top_right_x: float = 0.95
    roi_top_y: float = 0.42
    hsv_white_s_max: int = 110
    hsv_white_v_min: int = 115
    tophat_kernel_size: int = 25
    tophat_threshold: int = 30
    canny_low_threshold: int = 35
    canny_high_threshold: int = 110
    edge_brightness_min: int = 95
    min_elongation: float = 2.2
    min_contour_height: int = 20
    max_contour_width_ratio: float = 0.30
    min_contour_bottom_y: float = 0.50
    max_angle_from_vertical: float = 65.0
    bird_src_bottom_left_x: float = 0.05
    bird_src_bottom_right_x: float = 0.95
    bird_src_top_left_x: float = 0.34
    bird_src_top_right_x: float = 0.66
    bird_src_top_y: float = 0.42
    bird_dst_left_x: float = 0.25
    bird_dst_right_x: float = 0.75
    sliding_windows: int = 9
    sliding_margin: int = 55
    sliding_min_pixels: int = 12
    sliding_min_histogram: int = 800
    sliding_min_fit_pixels: int = 45
    sliding_lookahead_y: float = 0.65
    min_lane_width_ratio: float = 0.25
    max_lane_width_ratio: float = 0.85
    single_lane_center_offset_ratio: float = 0.24


@dataclass(frozen=True)
class LidarConfig:
    port: Optional[str] = None
    rpm: int = 660
    front_angle_min: float = 330.0
    front_angle_max: float = 30.0
    stop_distance_mm: float = 300.0


@dataclass(frozen=True)
class SerialConfig:
    arduino_port: Optional[str] = None
    baudrate: int = 115200
    timeout_s: float = 0.1


@dataclass(frozen=True)
class ControlConfig:
    base_speed: int = 90
    max_speed: int = 160
    max_steering: int = 120
    lane_kp: float = 85.0
    lane_lost_brake: bool = True


@dataclass(frozen=True)
class MissionConfig:
    mode: str = "time_trial"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    lane: LaneConfig
    lidar: LidarConfig
    serial: SerialConfig
    control: ControlConfig
    mission: MissionConfig


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("config section '%s' must be an object" % name)
    return value


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        camera=CameraConfig(**_section(data, "camera")),
        lane=LaneConfig(**_section(data, "lane")),
        lidar=LidarConfig(**_section(data, "lidar")),
        serial=SerialConfig(**_section(data, "serial")),
        control=ControlConfig(**_section(data, "control")),
        mission=MissionConfig(**_section(data, "mission")),
    )


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return config_from_dict(data)
