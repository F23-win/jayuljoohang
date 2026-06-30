import argparse
import time
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autocar.common.types import DriveCommand
from autocar.control.arduino_serial import ArduinoSerial


def send_for(controller, command, duration_s, interval_s=0.1):
    end_at = time.time() + duration_s
    while time.time() < end_at:
        controller.send(command)
        time.sleep(interval_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--test",
        choices=["steering-left", "steering-right", "steering-sweep", "drive", "all"],
        default="steering-sweep",
        help="Which hardware behavior to test.",
    )
    parser.add_argument("--speed", type=int, default=0, help="Optional drive test speed. Keep wheels lifted.")
    parser.add_argument("--steering", type=int, default=35)
    parser.add_argument("--hold", type=float, default=1.5)
    args = parser.parse_args()

    controller = ArduinoSerial({"serial_port": args.port, "baudrate": args.baudrate, "startup_wait_s": 2.0})
    with controller:
        controller.send(DriveCommand.stop("test"))
        print("sent STOP")
        time.sleep(0.3)

        if args.test in ("steering-left", "steering-sweep", "all"):
            send_for(controller, DriveCommand(speed=0, steering=args.steering, reason="steering_positive"), args.hold)
            print(f"sent steering positive test: steering={args.steering}")
            time.sleep(0.3)
            controller.send(DriveCommand.stop("after_steering_positive"))

        if args.test in ("steering-right", "steering-sweep", "all"):
            send_for(controller, DriveCommand(speed=0, steering=-args.steering, reason="steering_negative"), args.hold)
            print(f"sent steering negative test: steering={-args.steering}")
            time.sleep(0.3)
            controller.send(DriveCommand.stop("after_steering_negative"))

        if args.test in ("drive", "all"):
            if args.speed == 0:
                print("drive test skipped because --speed is 0")
            else:
                send_for(controller, DriveCommand(speed=args.speed, steering=0, reason="drive_test"), args.hold)
                print(f"sent drive test: speed={args.speed}")

        if args.test not in ("drive", "all") and args.speed != 0:
            send_for(controller, DriveCommand(speed=args.speed, steering=0, reason="drive_test"), args.hold)
            print(f"sent drive test: speed={args.speed}")

        controller.send(DriveCommand.stop("done"))
        print("sent STOP")


if __name__ == "__main__":
    main()
