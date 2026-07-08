from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class BirdEyeViewConfig:
    enabled: bool = True
    src_top_y_ratio: float = 0.42
    src_bottom_y_ratio: float = 0.98
    src_top_width_ratio: float = 0.42
    src_bottom_width_ratio: float = 0.94
    dst_margin_x_ratio: float = 0.12


class BirdEyeViewTransformer:
    def __init__(self, config: BirdEyeViewConfig = BirdEyeViewConfig()):
        self.config = config
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._matrix = None
        self._inverse_matrix = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def warp_frame(self, frame: Any) -> Any:
        if not self.enabled:
            return frame

        import cv2

        height, width = frame.shape[:2]
        matrix, _ = self._matrices(height, width)
        return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    def warp_mask(self, mask: Optional[Any], frame_shape: Tuple[int, int, int]) -> Optional[Any]:
        if mask is None or not self.enabled:
            return mask

        import cv2

        height, width = frame_shape[:2]
        matrix, _ = self._matrices(height, width)
        return cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)

    def unwarp_point(self, x: float, y: float, frame_shape: Tuple[int, int, int]) -> Tuple[float, float]:
        if not self.enabled:
            return x, y

        import cv2
        import numpy as np

        height, width = frame_shape[:2]
        _, inverse = self._matrices(height, width)
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, inverse)[0][0]
        return float(mapped[0]), float(mapped[1])

    def source_points(self, frame_shape: Tuple[int, int, int]) -> Any:
        import numpy as np

        height, width = frame_shape[:2]
        return self._source_points(height, width).astype(np.int32)

    def _matrices(self, height: int, width: int) -> Tuple[Any, Any]:
        if self._cached_shape == (height, width) and self._matrix is not None:
            return self._matrix, self._inverse_matrix

        import cv2

        src = self._source_points(height, width)
        dst = self._destination_points(height, width)
        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self._inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        self._cached_shape = (height, width)
        return self._matrix, self._inverse_matrix

    def _source_points(self, height: int, width: int) -> Any:
        import numpy as np

        top_y = height * self.config.src_top_y_ratio
        bottom_y = height * self.config.src_bottom_y_ratio
        center_x = width / 2.0
        top_half = width * self.config.src_top_width_ratio / 2.0
        bottom_half = width * self.config.src_bottom_width_ratio / 2.0
        return np.float32(
            [
                [center_x - top_half, top_y],
                [center_x + top_half, top_y],
                [center_x + bottom_half, bottom_y],
                [center_x - bottom_half, bottom_y],
            ]
        )

    def _destination_points(self, height: int, width: int) -> Any:
        import numpy as np

        margin = width * self.config.dst_margin_x_ratio
        return np.float32(
            [
                [margin, 0.0],
                [width - margin, 0.0],
                [width - margin, float(height - 1)],
                [margin, float(height - 1)],
            ]
        )
