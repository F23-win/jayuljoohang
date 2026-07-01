from dataclasses import dataclass
from typing import Any, List, Tuple

from ..config import LaneConfig
from ..types import LaneEstimate


@dataclass(frozen=True)
class LaneDetectionResult:
    estimate: LaneEstimate
    gray: Any
    binary_mask: Any
    edge_mask: Any
    lane_mask: Any
    roi_points: Any
    inverse_perspective: Any
    contours: List[Any]


class LaneDetector:
    def __init__(self, config: LaneConfig):
        self.config = config

    def detect(self, frame: Any) -> LaneDetectionResult:
        cv2, np = _load_cv()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.config.hsv_white_v_min], dtype=np.uint8),
            np.array([179, self.config.hsv_white_s_max, 255], dtype=np.uint8),
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        th_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.tophat_kernel_size, self.config.tophat_kernel_size),
        )
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, th_kernel)
        _, tophat_mask = cv2.threshold(
            tophat, self.config.tophat_threshold, 255, cv2.THRESH_BINARY
        )

        edges = cv2.Canny(
            blurred,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold,
        )
        edge_kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)
        bright_mask = cv2.inRange(gray, self.config.edge_brightness_min, 255)
        bright_edge_mask = cv2.bitwise_and(edge_mask, bright_mask)

        binary_mask = cv2.bitwise_or(
            cv2.bitwise_and(white_mask, cv2.bitwise_or(tophat_mask, bright_mask)),
            cv2.bitwise_or(tophat_mask, bright_edge_mask),
        )

        roi_mask, roi_points = self._make_roi_mask(frame.shape, np, cv2)
        lane_mask = cv2.bitwise_and(binary_mask, roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        src, dst = self._make_birds_eye_points(frame.shape, np)
        perspective = cv2.getPerspectiveTransform(src, dst)
        inverse_perspective = cv2.getPerspectiveTransform(dst, src)
        warped_mask = cv2.warpPerspective(
            lane_mask,
            perspective,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours = self._select_contours(warped_mask, cv2, np)
        filtered_mask = self._mask_from_contours(warped_mask.shape, contours, cv2, np)
        estimate = self._estimate_lane(filtered_mask, cv2, np)
        return LaneDetectionResult(
            estimate=estimate,
            gray=gray,
            binary_mask=binary_mask,
            edge_mask=edge_mask,
            lane_mask=filtered_mask,
            roi_points=roi_points,
            inverse_perspective=inverse_perspective,
            contours=contours,
        )

    def _make_roi_mask(self, frame_shape: Tuple[int, int, int], np: Any, cv2: Any):
        height, width = frame_shape[:2]
        points = np.array(
            [
                [
                    (int(width * self.config.roi_bottom_left_x), height),
                    (int(width * self.config.roi_bottom_right_x), height),
                    (int(width * self.config.roi_top_right_x), int(height * self.config.roi_top_y)),
                    (int(width * self.config.roi_top_left_x), int(height * self.config.roi_top_y)),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, points, 255)
        return mask, points

    def _make_birds_eye_points(self, frame_shape: Tuple[int, int, int], np: Any):
        height, width = frame_shape[:2]
        src = np.float32(
            [
                [width * self.config.bird_src_bottom_left_x, height],
                [width * self.config.bird_src_bottom_right_x, height],
                [width * self.config.bird_src_top_right_x, height * self.config.bird_src_top_y],
                [width * self.config.bird_src_top_left_x, height * self.config.bird_src_top_y],
            ]
        )
        dst = np.float32(
            [
                [width * self.config.bird_dst_left_x, height],
                [width * self.config.bird_dst_right_x, height],
                [width * self.config.bird_dst_right_x, 0],
                [width * self.config.bird_dst_left_x, 0],
            ]
        )
        return src, dst

    def _select_contours(self, lane_mask: Any, cv2: Any, np: Any) -> List[Any]:
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        mask_height, mask_width = lane_mask.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if height < self.config.min_contour_height:
                continue
            if width > mask_width * self.config.max_contour_width_ratio:
                continue
            if y + height < mask_height * self.config.min_contour_bottom_y:
                continue

            (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
            long_side = max(rect_width, rect_height)
            short_side = min(rect_width, rect_height)
            if short_side < 1:
                continue
            if long_side / short_side < self.config.min_elongation:
                continue

            line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
            vx, vy = float(line[0]), float(line[1])
            raw_angle = abs(np.degrees(np.arctan2(vx, vy)))
            angle_from_vertical = min(raw_angle, abs(180.0 - raw_angle))
            if angle_from_vertical > self.config.max_angle_from_vertical:
                continue
            valid.append(contour)
        valid.sort(
            key=lambda contour: (
                cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3],
                cv2.contourArea(contour),
            ),
            reverse=True,
        )
        return valid[:4]

    def _mask_from_contours(self, mask_shape: Tuple[int, int], contours: List[Any], cv2: Any, np: Any):
        filtered = np.zeros(mask_shape, dtype=np.uint8)
        if contours:
            cv2.drawContours(filtered, contours, -1, 255, thickness=cv2.FILLED)
        return filtered

    def _histogram_base_x(self, histogram: Any, x_offset: int):
        if histogram.size == 0:
            return None
        peak_index = int(histogram.argmax())
        if histogram[peak_index] < self.config.sliding_min_histogram:
            return None
        return peak_index + x_offset

    def _collect_sliding_window_pixels(self, lane_mask: Any, base_x: int, np: Any):
        height = lane_mask.shape[0]
        window_height = height // self.config.sliding_windows
        nonzero_y, nonzero_x = lane_mask.nonzero()

        current_x = int(base_x)
        lane_indices = []
        for window in range(self.config.sliding_windows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_x_low = current_x - self.config.sliding_margin
            win_x_high = current_x + self.config.sliding_margin

            good_indices = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= win_x_low)
                & (nonzero_x < win_x_high)
            ).nonzero()[0]
            lane_indices.append(good_indices)

            if len(good_indices) >= self.config.sliding_min_pixels:
                current_x = int(np.mean(nonzero_x[good_indices]))

        if not lane_indices:
            return np.array([]), np.array([])
        lane_indices = np.concatenate(lane_indices)
        return nonzero_x[lane_indices], nonzero_y[lane_indices]

    def _fit_lane_x_at_y(self, xs: Any, ys: Any, y_eval: int, np: Any):
        if len(xs) < self.config.sliding_min_fit_pixels:
            return None
        fit = np.polyfit(ys, xs, 2)
        return float(fit[0] * y_eval * y_eval + fit[1] * y_eval + fit[2])

    def _infer_center_from_single_lane(self, lane_x: float, width: int, is_left_lane: bool) -> float:
        offset = width * self.config.single_lane_center_offset_ratio
        if is_left_lane:
            return lane_x + offset
        return lane_x - offset

    def _estimate_lane(self, lane_mask: Any, cv2: Any, np: Any) -> LaneEstimate:
        if cv2.countNonZero(lane_mask) == 0:
            return LaneEstimate(confidence=0.0)

        height, width = lane_mask.shape[:2]
        histogram_start_y = int(height * 0.55)
        histogram = np.sum(lane_mask[histogram_start_y:, :], axis=0)
        midpoint = width // 2

        left_base_x = self._histogram_base_x(histogram[:midpoint], 0)
        right_base_x = self._histogram_base_x(histogram[midpoint:], midpoint)
        if left_base_x is None and right_base_x is None:
            return LaneEstimate(confidence=0.0)

        y_eval = int(height * self.config.sliding_lookahead_y)
        left_x = right_x = None
        left_count = right_count = 0

        if left_base_x is not None:
            xs, ys = self._collect_sliding_window_pixels(lane_mask, left_base_x, np)
            left_count = len(xs)
            left_x = self._fit_lane_x_at_y(xs, ys, y_eval, np)

        if right_base_x is not None:
            xs, ys = self._collect_sliding_window_pixels(lane_mask, right_base_x, np)
            right_count = len(xs)
            right_x = self._fit_lane_x_at_y(xs, ys, y_eval, np)

        min_lane_width = width * self.config.min_lane_width_ratio
        max_lane_width = width * self.config.max_lane_width_ratio
        lane_x = None

        if left_x is not None and right_x is not None:
            lane_width = abs(right_x - left_x)
            if min_lane_width <= lane_width <= max_lane_width:
                lane_x = (left_x + right_x) / 2.0
            elif left_count >= right_count:
                lane_x = self._infer_center_from_single_lane(left_x, width, True)
            else:
                lane_x = self._infer_center_from_single_lane(right_x, width, False)
        elif left_x is not None:
            lane_x = self._infer_center_from_single_lane(left_x, width, True)
        elif right_x is not None:
            lane_x = self._infer_center_from_single_lane(right_x, width, False)

        if lane_x is None:
            return LaneEstimate(confidence=0.0)

        lane_x = max(0.0, min(width - 1.0, lane_x))
        offset_norm = (lane_x - (width / 2.0)) / (width / 2.0)
        confidence = min(1.0, cv2.countNonZero(lane_mask) / (width * height * 0.08))
        return LaneEstimate(center_offset_norm=offset_norm, confidence=confidence)


def _load_cv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for lane detection") from exc
    return cv2, np
