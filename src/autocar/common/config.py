import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def merge_camera_index(config: Dict[str, Any], camera_index: int):
    merged = dict(config)
    camera = dict(merged.get("camera", {}))
    camera["index"] = camera_index
    merged["camera"] = camera
    return merged


def merge_serial_port(config: Dict[str, Any], serial_port: str):
    merged = dict(config)
    control = dict(merged.get("control", {}))
    control["serial_port"] = serial_port
    merged["control"] = control
    return merged
