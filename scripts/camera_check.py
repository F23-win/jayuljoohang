import argparse

from skku_autocar.config import load_config
from skku_autocar.perception.lane import LaneDetector
from skku_autocar.sensors.camera import Camera


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = LaneDetector(config.lane)

    import cv2

    with Camera(config.camera) as camera:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("failed to read frame")
                return 1
            result = detector.detect(frame)
            overlay = frame.copy()
            cv2.polylines(overlay, result.roi_points, True, (255, 0, 0), 2)
            original_lane_mask = cv2.warpPerspective(
                result.lane_mask,
                result.inverse_perspective,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            lane_pixels = original_lane_mask > 0
            overlay[lane_pixels] = (
                overlay[lane_pixels] * 0.45
                + (0, 140, 0)
            ).astype("uint8")
            cv2.putText(
                overlay,
                "offset=%.3f confidence=%.2f" % (
                    result.estimate.center_offset_norm,
                    result.estimate.confidence,
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.imshow("camera lane check", overlay)
            cv2.imshow("lane mask", result.lane_mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
