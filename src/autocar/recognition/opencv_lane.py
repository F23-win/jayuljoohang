from dataclasses import dataclass
from typing import Any

from autocar.common.types import LaneInfo


@dataclass(frozen=True)
class LaneRecognitionResult:
    lane_info: LaneInfo
    lane_mask: Any
    roi_points: Any
    debug_frame: Any


class OpenCVLaneRecognizer:
    """Baseline lane recognizer using grayscale threshold and ROI."""

    def __init__(self, config):
        self.config = config

    def recognize(self, frame) -> LaneRecognitionResult:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = int(self.config.get("gray_threshold", 175))
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        roi_mask, roi_points = self._roi_mask(binary.shape, np, cv2)
        lane_mask = cv2.bitwise_and(binary, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        lane_info = self._extract_lane_info(lane_mask, np)
        debug_frame = self._draw_debug(frame, lane_mask, roi_points, lane_info, cv2, np)
        return LaneRecognitionResult(
            lane_info=lane_info,
            lane_mask=lane_mask,
            roi_points=roi_points,
            debug_frame=debug_frame,
        )

    def _roi_mask(self, shape, np, cv2):
        height, width = shape[:2]
        points = np.array(
            [
                [
                    (int(width * float(self.config.get("roi_bottom_left_x", 0.18))), height),
                    (int(width * float(self.config.get("roi_bottom_right_x", 0.82))), height),
                    (
                        int(width * float(self.config.get("roi_top_right_x", 0.65))),
                        int(height * float(self.config.get("roi_top_y", 0.55))),
                    ),
                    (
                        int(width * float(self.config.get("roi_top_left_x", 0.35))),
                        int(height * float(self.config.get("roi_top_y", 0.55))),
                    ),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, points, 255)
        return mask, points

    def _extract_lane_info(self, lane_mask, np):
        height, width = lane_mask.shape[:2]
        sample_y_ratio = float(self.config.get("sample_y_ratio", 0.78))
        sample_y = int(height * sample_y_ratio)
        row_thickness = int(self.config.get("row_thickness", 14))
        lane_width_px = float(self.config.get("lane_width_px", 360))

        top = max(0, sample_y - row_thickness)
        bottom = min(height, sample_y + row_thickness + 1)
        band = lane_mask[top:bottom, :]
        cols = np.where(np.any(band > 0, axis=0))[0]

        image_center_x = width // 2
        if len(cols) == 0:
            return LaneInfo(
                confidence=0.0,
                lane_center_x=image_center_x,
                image_center_x=image_center_x,
                reason="no_lane_pixels",
            )

        left = cols[cols < image_center_x]
        right = cols[cols >= image_center_x]

        if len(left) > 0 and len(right) > 0:
            lane_center_x = int(round((float(np.max(left)) + float(np.min(right))) / 2.0))
            reason = "two_edges"
        elif len(left) > 0:
            lane_center_x = int(round(float(np.max(left)) + lane_width_px / 2.0))
            reason = "left_edge_only"
        else:
            lane_center_x = int(round(float(np.min(right)) - lane_width_px / 2.0))
            reason = "right_edge_only"

        offset = (lane_center_x - image_center_x) / max(1.0, image_center_x)
        confidence = min(1.0, len(cols) / max(1.0, width * 0.12))
        return LaneInfo(
            center_offset_norm=float(offset),
            confidence=float(confidence),
            lane_center_x=lane_center_x,
            image_center_x=image_center_x,
            reason=reason,
        )

    def _draw_debug(self, frame, lane_mask, roi_points, lane_info, cv2, np):
        debug = frame.copy()
        overlay = debug.copy()
        cv2.fillPoly(overlay, roi_points, (255, 0, 0))
        debug = cv2.addWeighted(overlay, 0.18, debug, 0.82, 0)
        cv2.polylines(debug, roi_points, True, (255, 0, 0), 2)

        green = np.zeros_like(debug)
        green[:, :, 1] = lane_mask
        debug = cv2.addWeighted(green, 0.55, debug, 1.0, 0)

        height, _ = debug.shape[:2]
        cv2.line(debug, (lane_info.image_center_x, 0), (lane_info.image_center_x, height), (0, 0, 255), 2)
        cv2.line(debug, (lane_info.lane_center_x, 0), (lane_info.lane_center_x, height), (0, 255, 255), 2)
        cv2.putText(
            debug,
            f"offset={lane_info.center_offset_norm:.3f} conf={lane_info.confidence:.2f} {lane_info.reason}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        return debug
