from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep

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


def _dedupe_anchor_points(anchors: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    if not anchors:
        return []
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in anchors:
        key = (round(float(x), 4), round(float(y), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x), float(y)))
    return out


def _apply_anchor_constraints(
    points: np.ndarray,
    anchors: list[tuple[float, float]],
    snap_max_dist_px: float,
    anchor_weight: float,
    anchor_neighbor_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(points) == 0:
        return points.copy(), np.ones(0, dtype=np.float64), 0

    out = points.copy()
    weights = np.ones(len(out), dtype=np.float64)
    snapped = 0

    max_dist = max(0.5, float(snap_max_dist_px))
    w_anchor = max(1.0, float(anchor_weight))
    w_neighbor = max(1.0, min(w_anchor, float(anchor_neighbor_weight)))

    for ax, ay in anchors:
        anchor = np.array([ax, ay], dtype=np.float64)
        d = np.linalg.norm(out - anchor, axis=1)
        idx = int(np.argmin(d))
        if float(d[idx]) > max_dist:
            continue

        out[idx] = anchor
        weights[idx] = max(weights[idx], w_anchor)
        snapped += 1

        if idx > 0:
            weights[idx - 1] = max(weights[idx - 1], w_neighbor)
        if idx + 1 < len(out):
            weights[idx + 1] = max(weights[idx + 1], w_neighbor)

    return out, weights, snapped


def _anchor_errors(tck: tuple, anchors: list[tuple[float, float]], sample_count: int) -> list[float]:
    if not anchors:
        return []

    n_samples = max(64, int(sample_count))
    u = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    sx, sy = splev(u, tck)
    curve = np.column_stack([sx, sy])

    errors: list[float] = []
    for ax, ay in anchors:
        p = np.array([ax, ay], dtype=np.float64)
        d = np.linalg.norm(curve - p, axis=1)
        errors.append(float(d.min(initial=float("inf"))))
    return errors


def fit_open_nurbs(
    polyline: Polyline2D,
    cfg: FitConfig,
    anchors: list[tuple[float, float]] | None = None,
) -> NurbsSpec:
    points = _remove_duplicate_neighbors(polyline.as_array())
    if len(points) < 2:
        raise ValueError("Polyline has fewer than 2 valid points")

    if len(points) < cfg.degree + 1:
        degree = max(1, len(points) - 1)
        return _fallback_spec(points, degree=degree, max_control_points=cfg.max_control_points)

    degree = max(1, min(int(cfg.degree), len(points) - 1))
    smoothing = float(cfg.spline.smoothing)
    work_points = points.copy()

    anchor_points = _dedupe_anchor_points(anchors)
    endpoint_anchors = [
        (float(work_points[0, 0]), float(work_points[0, 1])),
        (float(work_points[-1, 0]), float(work_points[-1, 1])),
    ]
    anchor_points = _dedupe_anchor_points(endpoint_anchors + anchor_points)
    use_anchor_correction = bool(cfg.spline.anchor_correction_enable and anchor_points)
    anchor_refit_left = max(0, int(cfg.spline.anchor_refit_max_iter))

    for _ in range(max(1, int(cfg.spline.max_iter))):
        constrained = work_points
        weights: np.ndarray | None = None
        if use_anchor_correction:
            factor = 1.0 + 0.5 * (int(cfg.spline.anchor_refit_max_iter) - anchor_refit_left)
            constrained, weights, _ = _apply_anchor_constraints(
                work_points,
                anchor_points,
                snap_max_dist_px=float(cfg.spline.anchor_snap_max_dist_px),
                anchor_weight=float(cfg.spline.anchor_weight) * factor,
                anchor_neighbor_weight=float(cfg.spline.anchor_neighbor_weight) * factor,
            )

        try:
            tck, _ = splprep(
                [constrained[:, 0], constrained[:, 1]],
                w=weights,
                s=smoothing,
                k=degree,
                per=False,
            )
            knots, coeffs, k = tck
            control_points = list(zip(coeffs[0], coeffs[1], strict=True))
            if len(control_points) <= cfg.max_control_points and _controls_reasonable(control_points, constrained):
                if use_anchor_correction and anchor_refit_left > 0:
                    errors = _anchor_errors(tck, anchor_points, int(cfg.spline.anchor_sample_count))
                    if errors and max(errors) > float(cfg.spline.anchor_tolerance_px):
                        anchor_refit_left -= 1
                        smoothing = max(0.0, smoothing * 0.5)
                        target_samples = max(cfg.max_control_points * 3, degree + 2)
                        work_points = _resample_by_arclength(constrained, target_samples)
                        continue

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
        target_samples = max(cfg.max_control_points * 2, degree + 2)
        if use_anchor_correction:
            target_samples = max(target_samples, len(anchor_points) * 8)
        work_points = _resample_by_arclength(constrained, target_samples)

    if cfg.spline.fallback_if_fail:
        return _fallback_spec(work_points, degree=degree, max_control_points=cfg.max_control_points)

    raise RuntimeError("Unable to fit NURBS under control-point limit")
