import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autocar.camera import CameraSource
from autocar.common.config import load_config, merge_camera_index, merge_serial_port
from autocar.common.types import DriveCommand
from autocar.control import ArduinoSerial, encode_command
from autocar.decision import LaneFollower
from autocar.recognition import OpenCVLaneRecognizer


def main():
    parser = argparse.ArgumentParser(description="Baseline camera -> recognition -> decision -> control loop")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--serial-port")
    parser.add_argument("--gray-threshold", type=int)
    parser.add_argument("--roi-top-y", type=float)
    parser.add_argument("--sample-y-ratio", type=float)
    parser.add_argument("--drive", action="store_true", help="Send commands to Arduino")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.camera_index is not None:
        config = merge_camera_index(config, args.camera_index)
    if args.serial_port:
        config = merge_serial_port(config, args.serial_port)
    recognition_config = dict(config.get("recognition", {}))
    if args.gray_threshold is not None:
        recognition_config["gray_threshold"] = args.gray_threshold
    if args.roi_top_y is not None:
        recognition_config["roi_top_y"] = args.roi_top_y
    if args.sample_y_ratio is not None:
        recognition_config["sample_y_ratio"] = args.sample_y_ratio
    config["recognition"] = recognition_config

    recognizer = OpenCVLaneRecognizer(recognition_config)
    decider = LaneFollower(config.get("decision", {}))
    controller = ArduinoSerial(config.get("control", {})) if args.drive else None
    send_interval_s = float(config.get("control", {}).get("send_interval_s", 0.05))

    import cv2

    enabled = False
    last_send_at = 0.0

    print("mode:", "DRIVE" if args.drive else "PREVIEW")
    print("space=start/pause, s=stop, q=quit")

    try:
        if controller is not None:
            controller.open()
            controller.send(DriveCommand.stop("startup"))

        with CameraSource(config.get("camera", {})) as camera:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("failed to read camera frame")
                    break

                result = recognizer.recognize(frame)
                command = decider.decide(result.lane_info, enabled)

                now = time.time()
                if controller is not None and now - last_send_at >= send_interval_s:
                    controller.send(command)
                    last_send_at = now

                cv2.putText(
                    result.debug_frame,
                    f"{'RUNNING' if enabled else 'PAUSED'} {encode_command(command).strip()} {command.reason}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if enabled else (0, 180, 255),
                    2,
                )
                cv2.imshow("baseline debug", result.debug_frame)
                cv2.imshow("baseline lane mask", result.lane_mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    enabled = not enabled
                if key == ord("s"):
                    enabled = False
                    if controller is not None:
                        controller.send(DriveCommand.stop("manual_stop"))
    finally:
        if controller is not None:
            controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
