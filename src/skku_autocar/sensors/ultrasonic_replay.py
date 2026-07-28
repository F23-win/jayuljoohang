from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class UltrasonicFrontTrace:
    """Frame-aligned filtered front distances extracted from a debug video."""

    values: Tuple[Optional[int], ...]

    @classmethod
    def from_csv(cls, path: str) -> "UltrasonicFrontTrace":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                "ultrasonic replay trace not found: %s" % source
            )

        indexed = {}
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"frame", "front_mm"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(
                    "ultrasonic replay trace requires frame,front_mm columns"
                )
            for row in reader:
                frame = int(row["frame"])
                if frame < 0:
                    raise ValueError("trace frame must be non-negative")
                raw_value = str(row.get("front_mm", "")).strip()
                value = int(raw_value) if raw_value else None
                # The Arduino filter accepts positive echoes only in this range.
                # OCR failures outside it are equivalent to no usable echo.
                if value is not None and not 50 <= value <= 3200:
                    value = None
                indexed[frame] = value

        if not indexed:
            return cls(values=())
        last_frame = max(indexed)
        values = tuple(indexed.get(frame) for frame in range(last_frame + 1))
        return cls(values=values)

    def front_mm(self, frame_index: int) -> Optional[int]:
        index = int(frame_index)
        if index < 0 or index >= len(self.values):
            return None
        return self.values[index]
