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


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm == 0.0:
        return float(np.linalg.norm(point - start))
    return float(np.abs(np.cross(line, point - start)) / norm)


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]
    max_dist = -1.0
    split_idx = -1

    for i in range(1, len(points) - 1):
        d = _point_line_distance(points[i], start, end)
        if d > max_dist:
            max_dist = d
            split_idx = i

    if max_dist <= float(epsilon):
        return np.vstack([start, end])

    left = _rdp(points[: split_idx + 1], epsilon)
    right = _rdp(points[split_idx:], epsilon)
    return np.vstack([left[:-1], right])


def _reduce_points_to_budget(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    budget = max(2, int(max_points))
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    low = 0.0
    high = max(1.0, diag)
    best: np.ndarray | None = None

    for _ in range(24):
        mid = 0.5 * (low + high)
        candidate = _rdp(points, mid)
        if len(candidate) > budget:
            low = mid
        else:
            best = candidate
            high = mid

    if best is None:
        best = _rdp(points, high)

    if len(best) > budget:
        best = _resample_by_arclength(points, budget)
    return best


def _hard_edge_spec(points: np.ndarray, max_control_points: int) -> NurbsSpec:
    control = _reduce_points_to_budget(points, max_control_points)
    if len(control) < 2:
        control = _resample_by_arclength(points, 2)
    knots = _clamped_uniform_knots(len(control), degree=1)
    return NurbsSpec(
        degree=1,
        control_points=[(float(x), float(y)) for x, y in control],
        knots=knots,
        weights=None,
    )


def _is_near_straight(points: np.ndarray, cfg: FitConfig) -> bool:
    if len(points) < 2:
        return False

    start = points[0]
    end = points[-1]
    chord = float(np.linalg.norm(end - start))
    if chord <= 1e-6:
        return False

    seg_min = float(getattr(cfg.segment, "straight_min_chord_px", cfg.spline.hard_edge_straight_min_length_px))
    min_length = max(1.0, min(float(cfg.spline.hard_edge_straight_min_length_px), seg_min))
    short_candidate = False
    if chord < min_length:
        # For small segments (typical rectangle edges after splitting), allow a
        # stricter short-line gate instead of rejecting immediately.
        if len(points) <= 8:
            short_candidate = True
        else:
            return False

    max_dev = 0.0
    if len(points) > 2:
        max_dev = max(_point_line_distance(p, start, end) for p in points[1:-1])
    dev_limit = max(
        float(cfg.spline.hard_edge_straight_max_deviation_px),
        float(cfg.spline.hard_edge_straight_max_deviation_ratio) * chord,
    )
    if short_candidate:
        dev_limit = min(dev_limit, 0.8)
    if max_dev > dev_limit:
        return False

    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 1e-6
    seg = seg[valid]
    seg_len = seg_len[valid]
    if len(seg) < 2:
        return True

    unit = seg / seg_len[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    turns = np.degrees(np.arccos(dots))
    if len(turns) == 0:
        return True

    turn_limit = max(1.0, float(cfg.spline.hard_edge_straight_max_turn_deg))
    if short_candidate:
        turn_limit = min(turn_limit, 10.0)
    return float(np.percentile(turns, 95.0)) <= turn_limit


def _should_use_hard_edge(points: np.ndarray, cfg: FitConfig) -> bool:
    mode = str(getattr(cfg.spline, "mode", "auto")).strip().lower()
    if mode == "hard_edge":
        return True
    if mode == "smooth":
        return False
    if mode != "auto":
        return False

    if _is_near_straight(points, cfg):
        return True

    if len(points) < 4:
        return False

    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 1e-6
    seg = seg[valid]
    seg_len = seg_len[valid]
    if len(seg) < 3:
        return False

    unit = seg / seg_len[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    if len(angles) == 0:
        return False

    corner_min = max(5.0, float(cfg.spline.hard_edge_corner_angle_deg))
    corner_mask = angles >= corner_min
    corner_count = int(np.count_nonzero(corner_mask))
    if corner_count < max(1, int(cfg.spline.hard_edge_min_corners)):
        return False

    right_tol = max(5.0, float(cfg.spline.hard_edge_right_angle_tolerance_deg))
    right_count = int(np.count_nonzero(np.abs(angles[corner_mask] - 90.0) <= right_tol))
    right_ratio = float(right_count / corner_count) if corner_count > 0 else 0.0
    corner_ratio = float(corner_count / max(1, len(angles)))

    max_vertices = max(4, int(cfg.spline.hard_edge_max_vertices))
    right_ratio_min = float(cfg.spline.hard_edge_right_angle_ratio_min)
    corner_ratio_min = float(cfg.spline.hard_edge_corner_ratio_min)

    return right_ratio >= right_ratio_min and (
        len(points) <= max_vertices or corner_ratio >= corner_ratio_min
    )


def should_use_polyline_geometry(polyline: Polyline2D, cfg: FitConfig) -> bool:
    points = _remove_duplicate_neighbors(polyline.as_array())
    if len(points) < 2:
        return False
    return _should_use_hard_edge(points, cfg)


def fit_open_polyline(polyline: Polyline2D, cfg: FitConfig) -> Polyline2D:
    points = _remove_duplicate_neighbors(polyline.as_array())
    if len(points) < 2:
        raise ValueError("Polyline has fewer than 2 valid points")

    # Export near-straight long segments as true two-point line-like polylines.
    chord = float(np.linalg.norm(points[-1] - points[0]))
    seg_min = float(getattr(cfg.segment, "straight_min_chord_px", cfg.spline.hard_edge_straight_min_length_px))
    collapse_min = max(8.0, min(float(cfg.spline.hard_edge_straight_min_length_px), seg_min))
    if len(points) >= 5 and chord >= collapse_min and _is_near_straight(points, cfg):
        p0 = (float(points[0, 0]), float(points[0, 1]))
        p1 = (float(points[-1, 0]), float(points[-1, 1]))
        return Polyline2D(points=[p0, p1])

    reduced = _reduce_points_to_budget(points, max(2, int(cfg.max_control_points)))
    if len(reduced) < 2:
        reduced = _resample_by_arclength(points, 2)
    return Polyline2D(points=[(float(x), float(y)) for x, y in reduced])


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

    if _should_use_hard_edge(points, cfg):
        return _hard_edge_spec(points, max_control_points=cfg.max_control_points)

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
