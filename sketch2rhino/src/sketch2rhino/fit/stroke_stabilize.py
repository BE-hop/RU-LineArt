from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from sketch2rhino.config import StabilizeConfig
from sketch2rhino.types import Polyline2D


def _resample_by_arclength(points: np.ndarray, n_samples: int) -> np.ndarray:
    if len(points) == 0 or n_samples <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points.astype(np.float64), n_samples, axis=0)

    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total == 0.0:
        idx = np.linspace(0, len(points) - 1, n_samples).astype(int)
        return points[idx].astype(np.float64)

    targets = np.linspace(0.0, total, int(n_samples), dtype=np.float64)
    out = np.zeros((int(n_samples), 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, s, points[:, 0])
    out[:, 1] = np.interp(targets, s, points[:, 1])
    return out


def _resample_by_step(points: np.ndarray, step_px: float) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float64)

    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    total = float(seg.sum())
    if total == 0.0:
        return points.astype(np.float64)

    step = max(0.5, float(step_px))
    n_samples = max(len(points), int(np.ceil(total / step)) + 1)
    return _resample_by_arclength(points, n_samples)


def _dedupe_anchors(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in anchors:
        key = (round(float(x), 4), round(float(y), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x), float(y)))
    return out


def _snap_anchors(points: np.ndarray, anchors: list[tuple[float, float]], snap_radius: float) -> None:
    if len(points) == 0 or not anchors:
        return
    radius = max(0.5, float(snap_radius))
    for ax, ay in anchors:
        p = np.array([ax, ay], dtype=np.float64)
        d = np.linalg.norm(points - p, axis=1)
        idx = int(np.argmin(d))
        if float(d[idx]) <= radius:
            points[idx] = p


def _valid_savgol_window(n_points: int, requested: int, polyorder: int) -> int:
    if n_points < 5:
        return 0

    max_odd = n_points if n_points % 2 == 1 else n_points - 1
    w = int(requested)
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, max_odd))

    min_valid = int(polyorder) + 3
    if min_valid % 2 == 0:
        min_valid += 1
    if w < min_valid:
        w = min_valid
        if w > max_odd:
            return 0
    return w


def stabilize_polyline(
    polyline: Polyline2D,
    cfg: StabilizeConfig,
    protected_points: list[tuple[float, float]] | None = None,
) -> Polyline2D:
    points = polyline.as_array()
    if not bool(cfg.enable) or len(points) < 5:
        return polyline
    if str(cfg.method).lower() != "savgol":
        return polyline

    work = _resample_by_step(points, cfg.resample_step_px)
    if len(work) < 5:
        return polyline

    polyorder = max(1, int(cfg.polyorder))
    window = _valid_savgol_window(len(work), int(cfg.window), polyorder)
    if window == 0:
        return polyline
    polyorder = min(polyorder, window - 2)

    blend = float(np.clip(float(cfg.blend), 0.0, 1.0))
    passes = max(1, int(cfg.passes))
    snap_radius = max(0.5, float(cfg.anchor_snap_radius_px))

    anchors: list[tuple[float, float]] = [
        (float(points[0, 0]), float(points[0, 1])),
        (float(points[-1, 0]), float(points[-1, 1])),
    ]
    if protected_points:
        anchors.extend((float(x), float(y)) for x, y in protected_points)
    anchors = _dedupe_anchors(anchors)

    out = work.copy()
    for _ in range(passes):
        sx = savgol_filter(out[:, 0], window_length=window, polyorder=polyorder, mode="interp")
        sy = savgol_filter(out[:, 1], window_length=window, polyorder=polyorder, mode="interp")
        smooth = np.column_stack([sx, sy])
        out = (1.0 - blend) * out + blend * smooth
        out[0] = points[0]
        out[-1] = points[-1]
        _snap_anchors(out, anchors, snap_radius)

    out = _resample_by_arclength(out, len(points))
    out[0] = points[0]
    out[-1] = points[-1]
    _snap_anchors(out, anchors, snap_radius * 1.25)

    return Polyline2D(points=[(float(x), float(y)) for x, y in out])
