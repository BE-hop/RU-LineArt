from __future__ import annotations

import numpy as np

from sketch2rhino.config import SimplifyConfig
from sketch2rhino.types import Polyline2D


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0.0:
        return float(np.linalg.norm(point - start))
    return float(np.abs(np.cross(line, point - start)) / norm)


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]

    max_dist = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = _point_line_distance(points[i], start, end)
        if d > max_dist:
            max_dist = d
            idx = i

    if max_dist <= epsilon:
        return np.vstack([start, end])

    left = _rdp(points[: idx + 1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack([left[:-1], right])


def _moving_average_smooth(points: np.ndarray, window: int, passes: int) -> np.ndarray:
    if len(points) < 3:
        return points

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    p = max(1, int(passes))

    out = points.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2

    for _ in range(p):
        # Use edge-padding to avoid zero-padding artifacts near both ends.
        xpad = np.pad(out[:, 0], (pad, pad), mode="edge")
        ypad = np.pad(out[:, 1], (pad, pad), mode="edge")
        xs = np.convolve(xpad, kernel, mode="valid")
        ys = np.convolve(ypad, kernel, mode="valid")
        out[:, 0] = xs
        out[:, 1] = ys
        # Keep the open-curve endpoints fixed.
        out[0] = points[0]
        out[-1] = points[-1]

    return out


def simplify_polyline(polyline: Polyline2D, cfg: SimplifyConfig) -> Polyline2D:
    points = polyline.as_array()
    if len(points) < 3 or cfg.method != "rdp":
        return polyline

    simplified = _rdp(points, float(cfg.epsilon_px))
    if cfg.smooth_enable:
        simplified = _moving_average_smooth(
            simplified,
            window=cfg.smooth_window,
            passes=cfg.smooth_passes,
        )
    out = [(float(x), float(y)) for x, y in simplified]
    return Polyline2D(points=out)
