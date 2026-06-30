from typing import Any, Tuple
import platform


class CameraSource:
    """OpenCV camera wrapper. This layer only reads frames."""

    def __init__(self, config):
        self.config = config
        self.cap = None

    def open(self):
        import cv2

        index = int(self.config.get("index", 0))
        width = int(self.config.get("width", 1280))
        height = int(self.config.get("height", 720))
        fourcc = str(self.config.get("fourcc", "MJPG"))

        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

        self.cap = cap
        return self

    def read(self) -> Tuple[bool, Any]:
        if self.cap is None:
            raise RuntimeError("camera is not open")
        return self.cap.read()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
