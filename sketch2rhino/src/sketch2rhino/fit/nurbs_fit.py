from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep

from sketch2rhino.config import FitConfig
from sketch2rhino.types import NurbsSpec, Polyline2D


def _remove_duplicate_neighbors(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if not np.allclose(points[i], points[keep[-1]]):
            keep.append(i)
    return points[keep]


def _resample_by_arclength(points: np.ndarray, n_samples: int) -> np.ndarray:
    if len(points) <= n_samples:
        return points

    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    if total == 0:
        idx = np.linspace(0, len(points) - 1, n_samples).astype(int)
        return points[idx]

    targets = np.linspace(0, total, n_samples)
    out = np.zeros((n_samples, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, s, points[:, 0])
    out[:, 1] = np.interp(targets, s, points[:, 1])
    return out


def _clamped_uniform_knots(n_control: int, degree: int) -> list[float]:
    if n_control <= degree:
        raise ValueError("n_control must be greater than degree")

    m = n_control + degree
    knots: list[float] = []
    for i in range(m + 1):
        if i <= degree:
            knots.append(0.0)
        elif i >= m - degree:
            knots.append(1.0)
        else:
            knots.append((i - degree) / (m - 2 * degree))
    return knots


def _fallback_spec(points: np.ndarray, degree: int, max_control_points: int) -> NurbsSpec:
    n_ctrl = min(max_control_points, len(points))
    n_ctrl = max(n_ctrl, degree + 1)
    sampled = _resample_by_arclength(points, n_ctrl)
    knots = _clamped_uniform_knots(len(sampled), degree)
    return NurbsSpec(
        degree=degree,
        control_points=[(float(x), float(y)) for x, y in sampled],
        knots=knots,
        weights=None,
    )


def _controls_reasonable(control_points: list[tuple[float, float]], ref_points: np.ndarray) -> bool:
    if not control_points:
        return False

    cps = np.asarray(control_points, dtype=np.float64)
    ref_min = ref_points.min(axis=0)
    ref_max = ref_points.max(axis=0)
    diag = float(np.linalg.norm(ref_max - ref_min))
    margin = max(5.0, 0.25 * diag)

    low = ref_min - margin
    high = ref_max + margin
    in_bbox = np.all((cps >= low) & (cps <= high))
    if not bool(in_bbox):
        return False

    if len(cps) >= 2:
        cp_steps = np.linalg.norm(np.diff(cps, axis=0), axis=1)
        ref_steps = np.linalg.norm(np.diff(ref_points, axis=0), axis=1)
        ref_median = float(np.median(ref_steps)) if len(ref_steps) else 0.0
        if ref_median > 0 and float(cp_steps.max(initial=0.0)) > 12.0 * ref_median:
            return False

    return True


def fit_open_nurbs(polyline: Polyline2D, cfg: FitConfig) -> NurbsSpec:
    points = _remove_duplicate_neighbors(polyline.as_array())
    if len(points) < 2:
        raise ValueError("Polyline has fewer than 2 valid points")

    if len(points) < cfg.degree + 1:
        degree = max(1, len(points) - 1)
        return _fallback_spec(points, degree=degree, max_control_points=cfg.max_control_points)

    degree = max(1, min(int(cfg.degree), len(points) - 1))
    smoothing = float(cfg.spline.smoothing)

    for _ in range(max(1, int(cfg.spline.max_iter))):
        try:
            tck, _ = splprep(
                [points[:, 0], points[:, 1]],
                s=smoothing,
                k=degree,
                per=False,
            )
            knots, coeffs, k = tck
            control_points = list(zip(coeffs[0], coeffs[1], strict=True))
            if len(control_points) <= cfg.max_control_points and _controls_reasonable(control_points, points):
                return NurbsSpec(
                    degree=int(k),
                    control_points=[(float(x), float(y)) for x, y in control_points],
                    knots=[float(v) for v in knots],
                    weights=None,
                )
        except Exception:
            if not cfg.spline.fallback_if_fail:
                raise

        smoothing *= 2.0
        points = _resample_by_arclength(points, max(cfg.max_control_points * 2, degree + 2))

    if cfg.spline.fallback_if_fail:
        return _fallback_spec(points, degree=degree, max_control_points=cfg.max_control_points)

    raise RuntimeError("Unable to fit NURBS under control-point limit")
