# 카메라 확인하는 코드

import argparse
import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=6)
    args = parser.parse_args()

    import cv2

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0
    opened_count = 0

    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"{index}: not opened")
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"{index}: OK {w}x{h}")
            opened_count += 1
        else:
            print(f"{index}: opened but no frame")
        cap.release()

    if opened_count == 0 and platform.system() == "Darwin":
        print()
        print("No camera opened on macOS.")
        print("Check System Settings > Privacy & Security > Camera.")
        print("Allow the terminal app you are using, then rerun this script.")


if __name__ == "__main__":
    main()
