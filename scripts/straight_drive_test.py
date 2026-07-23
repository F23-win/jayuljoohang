#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.control.serial_vehicle import SerialVehicleClient, SerialVehicleConfig
from skku_autocar.types import ControlCommand


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive straight with steering fixed at zero, then stop safely."
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help="Arduino serial port such as COM9; auto-detected when omitted.",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--steering-trim", type=int, default=-15)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--command-hz", type=float, default=20.0)
    parser.add_argument("--startup-delay", type=float, default=5.0)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--max-speed", type=int, default=255)
    parser.add_argument("--max-steering", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if args.duration <= 0.0:
        raise SystemExit("--duration must be greater than zero")
    if args.command_hz <= 0.0:
        raise SystemExit("--command-hz must be greater than zero")

    vehicle = SerialVehicleClient(
        SerialVehicleConfig(
            port=args.serial_port,
            baudrate=args.baudrate,
            timeout_s=0.1,
            startup_delay_s=max(0.0, args.startup_delay),
            ready_timeout_s=max(0.1, args.ready_timeout),
        ),
        max_speed=max(1, abs(args.max_speed)),
        max_steering=max(1, abs(args.max_steering)),
    )
    connected = False
    try:
        print("connecting to Arduino...")
        vehicle.connect()
        connected = True
        print("connected:", vehicle.port)
        print(
            "driving straight: speed=%d steering=%d duration=%.1fs"
            % (args.speed, args.steering_trim, args.duration)
        )

        interval = 1.0 / args.command_hz
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            vehicle.send(
                ControlCommand(
                    speed=args.speed,
                    steering=args.steering_trim,
                    brake=False,
                    reason="straight_drive_test",
                )
            )
            time.sleep(interval)
        return 0
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1
    finally:
        if connected:
            try:
                vehicle.stop("straight_drive_test_finished")
                time.sleep(0.1)
                print("stopped")
            except Exception as exc:
                print("stop warning:", exc, file=sys.stderr)
        vehicle.close()


if __name__ == "__main__":
    raise SystemExit(main())
