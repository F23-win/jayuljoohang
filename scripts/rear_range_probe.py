#!/usr/bin/env python3
"""Console/CSV probe for the rear-center LiDAR sector.

Measures deadzone distance and bumper offset (M3) before the sensor's
angle_offset_deg is trusted, so this stays in raw LiDAR angle_deg space --
it does not use estimation.parking_lidar's vehicle-frame transform.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.sensors.lidar import (
    LidarCsvRecorder,
    LidarPoint,
    LidarScan,
    RplidarScanner,
    angle_in_window,
    find_lidar_port,
)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print rear-center LiDAR sector range at 10Hz and log the full scan to CSV"
    )
    parser.add_argument("--lidar-port", default=None, help="LiDAR serial port; auto-detected when omitted")
    parser.add_argument(
        "--center-deg", type=float, default=180.0,
        help="raw LiDAR angle_deg for the sector center (not vehicle-frame bearing); match to your mounting",
    )
    parser.add_argument("--half-width-deg", type=float, default=10.0, help="sector half-width in degrees")
    parser.add_argument("--quality-min", type=int, default=1, help="minimum quality for a return to count as valid")
    parser.add_argument("--max-range-mm", type=float, default=12000.0, help="sanity cap on distance_mm; 0 disables")
    parser.add_argument("--print-rate-hz", type=float, default=10.0)
    parser.add_argument("--summary-bin-deg", type=float, default=10.0, help="full-circle bin width for the exit-time summary")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means until Ctrl+C")
    parser.add_argument("--csv-out", default=None, help="defaults to data/raw/lidar_probe/<timestamp>.csv")
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    port = find_lidar_port(args.lidar_port)
    if port is None:
        raise RuntimeError(
            "LiDAR serial port was not found. Run scripts/list_serial_ports.py and pass --lidar-port explicitly."
        )
    print("opening lidar:", port)

    csv_path = args.csv_out or default_csv_path()
    recorder = LidarCsvRecorder(csv_path)
    print("logging full scans to:", csv_path)

    scanner = RplidarScanner(port)
    scanner.start()
    try:
        scanner.wait_ready(args.ready_timeout)
    except Exception:
        scanner.close()
        recorder.close()
        raise

    bins: Dict[int, list] = {}
    last_recorded_timestamp: Optional[float] = None
    next_print_at = time.monotonic()
    print_interval = 1.0 / max(1e-6, args.print_rate_hz)
    deadline = None if args.duration <= 0 else time.monotonic() + args.duration

    try:
        while deadline is None or time.monotonic() < deadline:
            scan = scanner.latest()
            if scan is not None and scan.timestamp != last_recorded_timestamp:
                recorder.write_scan(scan)
                accumulate_bins(bins, scan, args.summary_bin_deg, args.quality_min, args.max_range_mm)
                last_recorded_timestamp = scan.timestamp

            now = time.monotonic()
            if scan is not None and now >= next_print_at:
                min_distance, valid_count = sector_stats(
                    scan, args.center_deg, args.half_width_deg, args.quality_min, args.max_range_mm
                )
                print(
                    "t=%.3f sector=[%.1f..%.1f]deg min_dist_mm=%s valid_returns=%d"
                    % (
                        scan.timestamp,
                        args.center_deg - args.half_width_deg,
                        args.center_deg + args.half_width_deg,
                        "%.1f" % min_distance if min_distance is not None else "none",
                        valid_count,
                    )
                )
                next_print_at = now + print_interval
            time.sleep(0.01)
    finally:
        scanner.close()
        recorder.close()
        print("scans recorded:", recorder.scans_written)
        print_bin_summary(bins, args.summary_bin_deg)
    return 0


def sector_stats(
    scan: LidarScan,
    center_deg: float,
    half_width_deg: float,
    quality_min: int,
    max_range_mm: float,
) -> Tuple[Optional[float], int]:
    """Min distance and valid-return count within [center-half, center+half] raw degrees."""
    start = center_deg - half_width_deg
    end = center_deg + half_width_deg
    distances = [
        point.distance_mm
        for point in scan.points
        if angle_in_window(point.angle_deg, start, end) and is_valid_return(point, quality_min, max_range_mm)
    ]
    if not distances:
        return None, 0
    return min(distances), len(distances)


def is_valid_return(point: LidarPoint, quality_min: int, max_range_mm: float) -> bool:
    if point.quality < quality_min or point.distance_mm <= 0:
        return False
    if max_range_mm > 0 and point.distance_mm > max_range_mm:
        return False
    return True


def accumulate_bins(
    bins: Dict[int, list],
    scan: LidarScan,
    bin_width_deg: float,
    quality_min: int,
    max_range_mm: float,
) -> None:
    for point in scan.points:
        index = bin_index(point.angle_deg, bin_width_deg)
        counts = bins.setdefault(index, [0, 0])
        counts[1] += 1
        if is_valid_return(point, quality_min, max_range_mm):
            counts[0] += 1


def bin_index(angle_deg: float, bin_width_deg: float) -> int:
    return int((angle_deg % 360.0) // max(1e-6, bin_width_deg))


def print_bin_summary(bins: Dict[int, list], bin_width_deg: float) -> None:
    print("--- per-sector valid-return ratio (raw angle_deg bins) ---")
    for index in sorted(bins):
        valid, total = bins[index]
        start = index * bin_width_deg
        ratio = valid / total if total else 0.0
        print("[%6.1f..%6.1f)deg  valid=%5d/%5d  ratio=%.2f" % (start, start + bin_width_deg, valid, total, ratio))


def default_csv_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "raw" / "lidar_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / ("%s.csv" % timestamp))


if __name__ == "__main__":
    raise SystemExit(main())
