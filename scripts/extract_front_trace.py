#!/usr/bin/env python3
"""Extract the rendered ``front=...`` ultrasonic trace from a debug video.

The runtime debug overlay is drawn with OpenCV's Hershey font.  Matching that
font directly is more reliable than a general OCR engine for the small yellow
status line and keeps this replay helper dependency-free.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
SEARCH_TOP = 312
SEARCH_BOTTOM = 358
TEXT_TOP = 14
TEXT_BOTTOM = 43


@dataclass(frozen=True)
class ExtractionProfile:
    font_scale: float
    value_offset_from_front: int
    reference_samples: Tuple[Tuple[int, str], ...]
    no_value_frames: Tuple[int, ...]


# The two supplied recordings use different debug font scales.  Each reference
# set covers all decimal glyphs and learns only their compressed appearance.
PROFILES_BY_FRAME_COUNT = {
    1258: ExtractionProfile(
        font_scale=0.68,
        value_offset_from_front=71,
        reference_samples=(
            (192, "2205"),
            (216, "2080"),
            (336, "1537"),
            (384, "1971"),
            (504, "1428"),
            (672, "1276"),
        ),
        no_value_frames=(4, 240, 360, 432, 480, 700),
    ),
    1540: ExtractionProfile(
        font_scale=0.85,
        value_offset_from_front=80,
        reference_samples=(
            (24, "1379"),
            (72, "884"),
            (96, "833"),
            (120, "824"),
            (144, "618"),
            (192, "1326"),
            (216, "705"),
            (240, "1307"),
            (264, "1546"),
            (288, "2195"),
            (312, "2084"),
            (360, "474"),
        ),
        no_value_frames=(0, 48, 168, 336),
    ),
}


@dataclass(frozen=True)
class Glyph:
    text: str
    mask: np.ndarray


@dataclass(frozen=True)
class OverlayCalibration:
    front_template: np.ndarray
    quorum_template: np.ndarray
    glyphs: Dict[str, Tuple[Glyph, ...]]
    no_value_templates: Tuple[np.ndarray, ...]
    value_offset_from_front: int


def rendered_text_mask(text: str, font_scale: float) -> np.ndarray:
    (width, height), baseline = cv2.getTextSize(
        text,
        FONT,
        font_scale,
        FONT_THICKNESS,
    )
    canvas = np.zeros((height + baseline + 8, width + 8), dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (4, height + 2),
        FONT,
        font_scale,
        255,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return tight_crop(canvas)


def tight_crop(mask: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask)
    if points is None:
        return mask[:0, :0]
    x, y, width, height = cv2.boundingRect(points)
    return mask[y : y + height, x : x + width]


def yellow_text_mask(frame: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame)
    # The overlay is BGR (0, 255, 255).  Keep its compressed antialiasing while
    # rejecting beige walls and the green segmentation overlay.
    selected = (
        (green >= 165)
        & (red >= 165)
        & (blue <= 145)
        & (np.abs(green.astype(np.int16) - red.astype(np.int16)) <= 80)
    )
    return selected.astype(np.uint8) * 255


def best_template_match(
    image: np.ndarray,
    template: np.ndarray,
) -> Tuple[float, Tuple[int, int]]:
    if (
        image.shape[0] < template.shape[0]
        or image.shape[1] < template.shape[1]
    ):
        return -1.0, (0, 0)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    return float(score), location


def column_groups(mask: np.ndarray, minimum_gap: int = 2) -> Tuple[Tuple[int, int], ...]:
    occupied = np.flatnonzero(np.any(mask > 0, axis=0))
    if occupied.size == 0:
        return ()
    groups = []
    start = int(occupied[0])
    previous = start
    for value in occupied[1:]:
        current = int(value)
        if current - previous > minimum_gap:
            groups.append((start, previous + 1))
            start = current
        previous = current
    groups.append((start, previous + 1))
    return tuple(groups)


def value_glyph_groups(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    if mask.size == 0:
        return ()
    # The green/yellow road overlay can touch the bottoms of adjacent rendered
    # digits.  The upper half still contains a clean inter-character gap.
    probe_height = max(8, int(round(mask.shape[0] * 0.55)))
    return column_groups(mask[:probe_height], minimum_gap=1)


def normalized_similarity(left: np.ndarray, right: np.ndarray) -> float:
    size = (32, 32)
    a = cv2.resize(left, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(right, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    a /= 255.0
    b /= 255.0
    numerator = float(np.sum(a * b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return numerator / denominator if denominator > 0.0 else 0.0


def recognize_glyph(
    glyph_mask: np.ndarray,
    glyphs: Dict[str, Tuple[Glyph, ...]],
) -> Tuple[str, float]:
    cropped = tight_crop(glyph_mask)
    best_character = ""
    best_score = -1.0
    for character, variants in glyphs.items():
        for glyph in variants:
            score = normalized_similarity(cropped, glyph.mask)
            if score > best_score:
                best_character = character
                best_score = score
    return best_character, best_score


def recognize_value(
    value_mask: np.ndarray,
    glyphs: Dict[str, Tuple[Glyph, ...]],
    no_value_templates: Tuple[np.ndarray, ...] = (),
) -> Tuple[Optional[int], str, float]:
    if value_mask.size == 0:
        return None, "", 0.0
    no_value_score = max(
        (
            normalized_similarity(value_mask, template)
            for template in no_value_templates
            if template.size > 0
        ),
        default=0.0,
    )
    if no_value_score >= 0.72:
        return None, "n/a", no_value_score
    groups = value_glyph_groups(value_mask)
    if not groups:
        return None, "", 0.0
    # In the compressed overlay the slash touches one of the adjacent letters,
    # so ``n/a`` consistently forms two connected horizontal groups.
    if len(groups) == 2:
        return None, "n/a", 1.0
    characters = []
    scores = []
    for start, end in groups:
        character, score = recognize_glyph(value_mask[:, start:end], glyphs)
        characters.append(character)
        scores.append(score)
    text = "".join(characters)
    confidence = min(scores) if scores else 0.0
    if text == "n/a":
        return None, text, confidence
    if text.isdigit():
        return int(text), text, confidence
    return None, text, confidence


def locate_status_fields(
    row: np.ndarray,
    front_template: np.ndarray,
    quorum_template: np.ndarray,
) -> Tuple[float, int, float, int]:
    front_score, (front_x, _) = best_template_match(row, front_template)
    quorum_search_left = min(row.shape[1], front_x + 75)
    quorum_search_right = min(row.shape[1], front_x + 190)
    quorum_region = row[:, quorum_search_left:quorum_search_right]
    quorum_score, (relative_q_x, _) = best_template_match(
        quorum_region,
        quorum_template,
    )
    quorum_x = quorum_search_left + relative_q_x
    return front_score, front_x, quorum_score, quorum_x


def status_value_mask(
    row: np.ndarray,
    front_x: int,
    quorum_x: int,
    value_offset_from_front: int,
) -> np.ndarray:
    value_left = min(
        row.shape[1],
        front_x + int(value_offset_from_front),
    )
    return tight_crop(row[TEXT_TOP:TEXT_BOTTOM, value_left:quorum_x])


def extract_front(
    frame: np.ndarray,
    calibration: OverlayCalibration,
) -> Tuple[Optional[int], str, float, float]:
    row = yellow_text_mask(frame)[SEARCH_TOP:SEARCH_BOTTOM]
    front_score, front_x, quorum_score, quorum_x = locate_status_fields(
        row,
        calibration.front_template,
        calibration.quorum_template,
    )
    if front_score < 0.60 or quorum_score < 0.60:
        return None, "", 0.0, min(front_score, quorum_score)
    value_region = status_value_mask(
        row,
        front_x,
        quorum_x,
        calibration.value_offset_from_front,
    )
    value, text, glyph_score = recognize_value(
        value_region,
        calibration.glyphs,
        calibration.no_value_templates,
    )
    return value, text, glyph_score, min(front_score, quorum_score)


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("cannot read calibration frame %d" % frame_index)
    return frame


def calibrate_overlay(
    capture: cv2.VideoCapture,
    profile: ExtractionProfile,
) -> OverlayCalibration:
    generated_quorum = rendered_text_mask("q", profile.font_scale)
    anchor = None
    quorum_score = -1.0
    quorum_x = 0
    quorum_y = 0
    for frame_index, _ in profile.reference_samples:
        candidate = yellow_text_mask(read_frame(capture, frame_index))[
            SEARCH_TOP:SEARCH_BOTTOM
        ]
        score, (x, y) = best_template_match(candidate, generated_quorum)
        if score > quorum_score:
            anchor = candidate
            quorum_score = score
            quorum_x = x
            quorum_y = y
    if anchor is None or quorum_score < 0.55:
        raise RuntimeError(
            "cannot locate q field in calibration frame (score=%.3f)"
            % quorum_score
        )

    generated_front = rendered_text_mask("front", profile.font_scale)
    front_search_left = max(0, quorum_x - 210)
    front_search_right = max(front_search_left, quorum_x - 55)
    front_score, (relative_front_x, front_y) = best_template_match(
        anchor[:, front_search_left:front_search_right],
        generated_front,
    )
    if front_score < 0.35:
        raise RuntimeError(
            "cannot locate front field in calibration frame (score=%.3f)"
            % front_score
        )
    front_x = front_search_left + relative_front_x

    front_template = anchor[
        front_y : front_y + generated_front.shape[0],
        front_x : front_x + generated_front.shape[1],
    ].copy()
    quorum_template = anchor[
        quorum_y : quorum_y + generated_quorum.shape[0],
        quorum_x : quorum_x + generated_quorum.shape[1],
    ].copy()

    learned: Dict[str, list] = {}
    for frame_index, expected in profile.reference_samples:
        row = yellow_text_mask(read_frame(capture, frame_index))[
            SEARCH_TOP:SEARCH_BOTTOM
        ]
        _, current_front_x, _, current_quorum_x = locate_status_fields(
            row,
            front_template,
            quorum_template,
        )
        value_mask = status_value_mask(
            row,
            current_front_x,
            current_quorum_x,
            profile.value_offset_from_front,
        )
        groups = value_glyph_groups(value_mask)
        if len(groups) != len(expected):
            raise RuntimeError(
                "calibration frame %d expected %r but found %d glyphs"
                % (frame_index, expected, len(groups))
            )
        for character, (start, end) in zip(expected, groups):
            learned.setdefault(character, []).append(
                Glyph(character, tight_crop(value_mask[:, start:end]))
            )

    no_value_templates = []
    for frame_index in profile.no_value_frames:
        row = yellow_text_mask(read_frame(capture, frame_index))[
            SEARCH_TOP:SEARCH_BOTTOM
        ]
        _, current_front_x, _, current_quorum_x = locate_status_fields(
            row,
            front_template,
            quorum_template,
        )
        no_value_templates.append(
            status_value_mask(
                row,
                current_front_x,
                current_quorum_x,
                profile.value_offset_from_front,
            )
        )

    return OverlayCalibration(
        front_template=front_template,
        quorum_template=quorum_template,
        glyphs={
            character: tuple(variants)
            for character, variants in learned.items()
        },
        no_value_templates=tuple(no_value_templates),
        value_offset_from_front=profile.value_offset_from_front,
    )


def frame_rows(
    video_path: Path,
) -> Iterable[Tuple[int, float, Optional[int], str, float, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot open video: %s" % video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    profile = PROFILES_BY_FRAME_COUNT.get(frame_count)
    if profile is None:
        raise RuntimeError(
            "no overlay calibration profile for %d frames" % frame_count
        )
    calibration = calibrate_overlay(capture, profile)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            value, text, glyph_score, match_score = extract_front(
                frame,
                calibration,
            )
            yield (
                frame_index,
                frame_index / fps,
                value,
                text,
                glyph_score,
                match_score,
            )
            frame_index += 1
    finally:
        capture.release()


def write_trace(video_path: Path, output_path: Path) -> None:
    rows = list(frame_rows(video_path))
    raw_values = [row[2] for row in rows]
    texts = [row[3] for row in rows]
    cleaned_values = clean_front_values(raw_values, texts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame",
                "time_s",
                "front_mm",
                "front_mm_raw",
                "ocr_text",
                "glyph_confidence",
                "match_confidence",
            )
        )
        for cleaned, row in zip(cleaned_values, rows):
            frame, time_s, value, text, glyph_score, match_score = row
            writer.writerow(
                (
                    frame,
                    "%.6f" % time_s,
                    "" if cleaned is None else cleaned,
                    "" if value is None else value,
                    text,
                    "%.4f" % glyph_score,
                    "%.4f" % match_score,
                )
            )


def clean_front_values(
    raw_values: Sequence[Optional[int]],
    texts: Sequence[str],
) -> Tuple[Optional[int], ...]:
    valid = [
        value if value is not None and 50 <= value <= 3200 else None
        for value in raw_values
    ]
    cleaned = list(valid)
    count = len(cleaned)

    def nearest_valid(index: int, radius: int = 12) -> Optional[int]:
        candidates = []
        for distance in range(1, radius + 1):
            for candidate_index in (index - distance, index + distance):
                if 0 <= candidate_index < count:
                    candidate = valid[candidate_index]
                    if candidate is not None:
                        candidates.append((distance, candidate))
            if candidates:
                return int(round(np.median([item[1] for item in candidates])))
        return None

    for index, (raw, text) in enumerate(zip(raw_values, texts)):
        if cleaned[index] is not None or text == "n/a":
            continue
        # A digit string outside the Arduino's range is still positive evidence
        # that the overlay contained a distance.  Replace only its corrupted
        # magnitude with the nearest valid rendered sample.
        if raw is not None and raw > 0:
            cleaned[index] = nearest_valid(index)
            continue
        # Fill a one-frame field-location miss only when both adjacent frames
        # contain valid values.  Explicit n/a frames are never interpolated.
        if (
            text == ""
            and 0 < index < count - 1
            and valid[index - 1] is not None
            and valid[index + 1] is not None
        ):
            cleaned[index] = int(
                round((valid[index - 1] + valid[index + 1]) / 2.0)
            )
    return tuple(cleaned)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract front ultrasonic values from a YOLO debug overlay"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    write_trace(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
