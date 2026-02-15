from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sketch2rhino.types import Polyline2D


def save_path_overlay(gray: np.ndarray, polyline: Polyline2D | list[Polyline2D], out_path: str | Path) -> None:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    polylines = polyline if isinstance(polyline, list) else [polyline]
    colors = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 140, 255),
        (180, 0, 180),
    ]

    for idx, pl in enumerate(polylines):
        pts = []
        for x, y in pl.points:
            col = int(round(x))
            row = int(round(-y))
            pts.append((col, row))

        if len(pts) >= 2:
            arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(base, [arr], isClosed=False, color=colors[idx % len(colors)], thickness=1)

    cv2.imwrite(str(Path(out_path)), base)
