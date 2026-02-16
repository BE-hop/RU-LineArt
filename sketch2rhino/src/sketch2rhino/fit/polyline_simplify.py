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


def _protected_indices(
    points: np.ndarray,
    protected_points: list[tuple[float, float]] | None,
    tol: float = 2.0,
) -> list[int]:
    if not protected_points or len(points) == 0:
        return []

    protected: set[int] = set()
    for px, py in protected_points:
        target = np.array([px, py], dtype=np.float64)
        d = np.linalg.norm(points - target, axis=1)
        idx = int(np.argmin(d))
        if float(d[idx]) <= tol:
            protected.add(idx)
    return sorted(protected)


def _rdp_with_protected(points: np.ndarray, epsilon: float, keep_indices: list[int]) -> np.ndarray:
    if len(points) < 3:
        return points

    keep = sorted(set([0, len(points) - 1] + keep_indices))
    if len(keep) <= 2:
        return _rdp(points, epsilon)

    chunks: list[np.ndarray] = []
    for i in range(1, len(keep)):
        a = keep[i - 1]
        b = keep[i]
        if b <= a:
            continue
        seg = points[a : b + 1]
        seg_simplified = _rdp(seg, epsilon)
        if not chunks:
            chunks.append(seg_simplified)
        else:
            chunks.append(seg_simplified[1:])
    if not chunks:
        return points[[0, -1]]
    return np.vstack(chunks)


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


def simplify_polyline(
    polyline: Polyline2D,
    cfg: SimplifyConfig,
    protected_points: list[tuple[float, float]] | None = None,
) -> Polyline2D:
    points = polyline.as_array()
    if len(points) < 3 or cfg.method != "rdp":
        return polyline

    keep_idx = _protected_indices(points, protected_points, tol=max(1.0, float(cfg.epsilon_px) * 2.0))
    simplified = _rdp_with_protected(points, float(cfg.epsilon_px), keep_idx)
    if cfg.smooth_enable:
        simplified = _moving_average_smooth(
            simplified,
            window=cfg.smooth_window,
            passes=cfg.smooth_passes,
        )
        # Re-pin protected points after smoothing to preserve topology anchors.
        if protected_points:
            targets = np.asarray(protected_points, dtype=np.float64)
            for idx in _protected_indices(simplified, protected_points, tol=3.0):
                d = np.linalg.norm(targets - simplified[idx], axis=1)
                simplified[idx] = targets[int(np.argmin(d))]

    out = [(float(x), float(y)) for x, y in simplified]
    return Polyline2D(points=out)
