import platform
import time
from typing import Any, Optional, Tuple

from ..config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[Any] = None

    def open(self) -> None:
        cv2 = _load_cv2()
        for index in self._index_candidates():
            for backend_name, backend in _backend_candidates(cv2):
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                if self.config.fourcc and platform.system() == "Windows":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))

                for _ in range(10):
                    ok, frame = cap.read()
                    if ok and _is_usable_frame(frame):
                        print("camera opened: index=%s backend=%s" % (index, backend_name))
                        self._cap = cap
                        return
                    time.sleep(0.05)

                if ok and frame is not None:
                    print(
                        "skipping black camera frame: index=%s backend=%s %s"
                        % (index, backend_name, _describe_frame(frame))
                    )
                cap.release()

        raise RuntimeError("camera frame could not be read from indexes %s" % self._index_candidates())

    def read(self) -> Tuple[bool, Any]:
        cap = self._require_open()
        return cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _require_open(self):
        if self._cap is None:
            raise RuntimeError("camera is not open")
        return self._cap

    def _index_candidates(self):
        candidates = [self.config.index, 0, 1, 2, 3]
        unique = []
        for index in candidates:
            if index not in unique:
                unique.append(index)
        return unique

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for camera access") from exc
    return cv2


def _backend_candidates(cv2):
    system = platform.system()
    candidates = []
    if system == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        candidates.append(("AVFOUNDATION", cv2.CAP_AVFOUNDATION))
    elif system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        candidates.append(("DSHOW", cv2.CAP_DSHOW))
    candidates.append(("DEFAULT", cv2.CAP_ANY))
    return candidates


def _is_usable_frame(frame: Any) -> bool:
    if frame is None:
        return False
    cv2 = _load_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) >= 5.0 or float(gray.std()) >= 5.0


def _describe_frame(frame: Any) -> str:
    cv2 = _load_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return "mean=%.1f std=%.1f" % (float(gray.mean()), float(gray.std()))
