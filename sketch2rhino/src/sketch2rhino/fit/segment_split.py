from __future__ import annotations

import math

import numpy as np

from sketch2rhino.config import SegmentConfig
from sketch2rhino.types import Polyline2D


def _remove_duplicate_neighbors(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = [0]
    for i in range(1, len(points)):
        dx = float(points[i, 0] - points[keep[-1], 0])
        dy = float(points[i, 1] - points[keep[-1], 1])
        if abs(dx) <= tol and abs(dy) <= tol:
            continue
        keep.append(i)
    return points[keep]


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm == 0.0:
        return float(np.linalg.norm(point - start))
    cross = float(line[0] * (point[1] - start[1]) - line[1] * (point[0] - start[0]))
    return abs(cross) / norm


def _turn_angle_deg(prev_pt: np.ndarray, curr_pt: np.ndarray, next_pt: np.ndarray) -> float:
    v1 = curr_pt - prev_pt
    v2 = next_pt - curr_pt
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-8 or n2 <= 1e-8:
        return 0.0
    dot = float(np.dot(v1 / n1, v2 / n2))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _polyline_arclen(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(points) == 1:
        return np.zeros(1, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_by_step(points: np.ndarray, step_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 2:
        s = _polyline_arclen(points)
        return points.copy(), s.copy(), s.copy()

    s_orig = _polyline_arclen(points)
    total = float(s_orig[-1])
    if total <= 1e-9:
        return points.copy(), s_orig.copy(), s_orig.copy()

    step = max(0.5, float(step_px))
    n_samples = max(2, int(math.ceil(total / step)) + 1)
    s_res = np.linspace(0.0, total, n_samples, dtype=np.float64)
    xs = np.interp(s_res, s_orig, points[:, 0])
    ys = np.interp(s_res, s_orig, points[:, 1])
    out = np.column_stack([xs, ys]).astype(np.float64)
    return out, s_res, s_orig


def _line_fit_stats(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0, 0.0

    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm

    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    d = np.abs(centered @ normal)
    return direction, float(np.mean(d)), float(np.max(d, initial=0.0))


def _fit_circle_stats(points: np.ndarray) -> tuple[np.ndarray, float, float, float] | None:
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([x, y, np.ones(len(points), dtype=np.float64)])
    b = -(x * x + y * y)
    try:
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    aa, bb, cc = (float(coef[0]), float(coef[1]), float(coef[2]))
    cx = -0.5 * aa
    cy = -0.5 * bb
    r2 = cx * cx + cy * cy - cc
    if not np.isfinite(r2) or r2 <= 1e-9:
        return None

    radius = float(math.sqrt(r2))
    center = np.array([cx, cy], dtype=np.float64)
    dist = np.linalg.norm(points - center, axis=1)
    err = np.abs(dist - radius)
    return center, radius, float(np.mean(err)), float(np.max(err, initial=0.0))


def _compute_multiscale_turn_matrix(points: np.ndarray, scales_idx: list[int]) -> np.ndarray:
    n = len(points)
    if n == 0 or not scales_idx:
        return np.zeros((n, 0), dtype=np.float64)

    out = np.full((n, len(scales_idx)), np.nan, dtype=np.float64)
    for j, k in enumerate(scales_idx):
        kk = max(1, int(k))
        for i in range(kk, n - kk):
            out[i, j] = _turn_angle_deg(points[i - kk], points[i], points[i + kk])
    return out


def _group_true_regions(mask: np.ndarray, gap_points: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []

    regions: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    max_step = max(1, int(gap_points) + 1)
    for raw in idx[1:]:
        cur = int(raw)
        if cur - prev <= max_step:
            prev = cur
            continue
        regions.append((start, prev))
        start = cur
        prev = cur
    regions.append((start, prev))
    return regions


def _arclen_to_nearest_index(target_s: float, s_orig: np.ndarray) -> int:
    if len(s_orig) == 0:
        return 0
    if target_s <= float(s_orig[0]):
        return 0
    if target_s >= float(s_orig[-1]):
        return len(s_orig) - 1

    right = int(np.searchsorted(s_orig, target_s, side="left"))
    left = max(0, right - 1)
    if right >= len(s_orig):
        return len(s_orig) - 1
    if abs(float(s_orig[left]) - target_s) <= abs(float(s_orig[right]) - target_s):
        return left
    return right


def _is_fillet_like_region(
    samples: np.ndarray,
    start: int,
    end: int,
    cfg: SegmentConfig,
) -> bool:
    if not bool(cfg.fillet_enable):
        return False
    if end <= start:
        return False

    n = len(samples)
    w = max(3, int(cfg.fillet_line_window_points))
    pre_a = max(0, start - w)
    pre_b = start + 1
    post_a = end
    post_b = min(n, end + w + 1)
    if pre_b - pre_a < 3 or post_b - post_a < 3:
        return False

    pre = samples[pre_a:pre_b]
    post = samples[post_a:post_b]
    pre_flow = pre[-1] - pre[0]
    post_flow = post[-1] - post[0]
    if float(np.linalg.norm(pre_flow)) <= 1e-8 or float(np.linalg.norm(post_flow)) <= 1e-8:
        return False

    _, pre_mean, _ = _line_fit_stats(pre)
    _, post_mean, _ = _line_fit_stats(post)
    line_limit = max(0.2, float(cfg.fillet_line_residual_px))
    if pre_mean > line_limit or post_mean > line_limit:
        return False

    seg = samples[max(0, start - 1) : min(n, end + 2)]
    if len(seg) < 4:
        return False
    span = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    if span > float(cfg.fillet_max_span_px):
        return False

    circle = _fit_circle_stats(seg)
    if circle is None:
        return False
    _, radius, circle_mean, _ = circle
    if radius < float(cfg.fillet_radius_min_px) or radius > float(cfg.fillet_radius_max_px):
        return False
    if circle_mean > float(cfg.fillet_circle_residual_px):
        return False

    u = pre_flow / float(np.linalg.norm(pre_flow))
    v = post_flow / float(np.linalg.norm(post_flow))
    dot = float(np.dot(u, v))
    dot = max(-1.0, min(1.0, dot))
    turn = float(math.degrees(math.acos(dot)))
    if turn < float(cfg.fillet_min_turn_deg) or turn > float(cfg.fillet_max_turn_deg):
        return False

    return True


def _merge_near_breaks(points: np.ndarray, breaks: list[int], merge_distance_px: float) -> list[int]:
    if len(breaks) <= 2:
        return breaks

    ordered = sorted(set(int(v) for v in breaks))
    if not ordered:
        return ordered

    s = _polyline_arclen(points)
    min_gap = max(0.0, float(merge_distance_px))
    out: list[int] = [ordered[0]]
    for idx in ordered[1:]:
        if idx >= len(points) - 1:
            if out[-1] != len(points) - 1:
                out.append(len(points) - 1)
            continue
        if idx <= 0:
            continue
        if float(s[idx] - s[out[-1]]) < min_gap:
            continue
        out.append(idx)

    if out[-1] != len(points) - 1:
        out.append(len(points) - 1)
    if out[0] != 0:
        out.insert(0, 0)
    return out


def _multiscale_corner_break_indices(points: np.ndarray, cfg: SegmentConfig) -> set[int]:
    if len(points) < 5:
        return set()

    samples, s_res, s_orig = _resample_by_step(points, float(cfg.corner_resample_step_px))
    if len(samples) < 5:
        return set()

    scales_px = [float(v) for v in cfg.corner_scales_px if float(v) > 0.0]
    if not scales_px:
        scales_px = [max(2.0, 3.0 * float(cfg.corner_resample_step_px))]
    scales_idx = sorted({max(1, int(round(v / max(0.5, float(cfg.corner_resample_step_px))))) for v in scales_px})
    max_k = max(scales_idx, default=1)

    mat = _compute_multiscale_turn_matrix(samples, scales_idx)
    if mat.size == 0:
        return set()

    valid = np.isfinite(mat)
    if not np.any(valid):
        return set()

    strong = max(5.0, float(cfg.corner_split_angle_deg))
    weak = strong * max(0.2, min(1.0, float(cfg.corner_multiscale_ratio)))
    hit = np.sum((mat >= weak) & valid, axis=1)
    max_angle = np.zeros(len(samples), dtype=np.float64)
    med_angle = np.zeros(len(samples), dtype=np.float64)
    for i in range(len(samples)):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue
        row = mat[i, row_valid]
        max_angle[i] = float(np.max(row))
        med_angle[i] = float(np.median(row))

    mask = (max_angle >= strong) & (med_angle >= weak) & (hit >= max(1, int(cfg.corner_min_scales_hit)))
    if max_k > 0 and len(mask) > 2 * max_k:
        mask[:max_k] = False
        mask[-max_k:] = False

    regions = _group_true_regions(mask, int(cfg.corner_region_gap_points))
    breaks: set[int] = set()
    for start, end in regions:
        if end - start < 1:
            continue
        local = max_angle[start : end + 1]
        if len(local) == 0:
            continue
        peak = int(start + int(np.argmax(local)))

        candidates_s: list[float]
        if _is_fillet_like_region(samples, start, end, cfg):
            candidates_s = [float(s_res[start]), float(s_res[end])]
        else:
            candidates_s = [float(s_res[peak])]

        for target_s in candidates_s:
            idx = _arclen_to_nearest_index(target_s, s_orig)
            if 0 < idx < len(points) - 1:
                breaks.add(int(idx))
    return breaks


def _window_is_straight(points: np.ndarray, cfg: SegmentConfig) -> bool:
    if len(points) < 3:
        return False

    start = points[0]
    end = points[-1]
    chord = float(np.linalg.norm(end - start))
    if chord < max(1.0, float(cfg.straight_min_chord_px)):
        return False

    dev95 = 0.0
    if len(points) > 2:
        dev = np.asarray([_point_line_distance(p, start, end) for p in points[1:-1]], dtype=np.float64)
        if len(dev) > 0:
            dev95 = float(np.percentile(dev, 95.0))
    dev_limit = max(
        float(cfg.straight_max_deviation_px),
        float(cfg.straight_max_deviation_ratio) * chord,
    )
    if dev95 > dev_limit:
        return False

    turns: list[float] = []
    for i in range(1, len(points) - 1):
        turns.append(_turn_angle_deg(points[i - 1], points[i], points[i + 1]))
    if not turns:
        return True
    return float(np.percentile(np.asarray(turns, dtype=np.float64), 95.0)) <= float(cfg.straight_max_turn_deg)


def _local_straight_labels(points: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    n = len(points)
    if n < 3:
        return np.zeros(n, dtype=bool)

    w = max(3, int(cfg.window_points))
    if w % 2 == 0:
        w += 1
    half = w // 2

    labels = np.zeros(n, dtype=bool)
    for i in range(n):
        a = max(0, i - half)
        b = min(n - 1, i + half)
        if b - a + 1 < 3:
            continue
        labels[i] = _window_is_straight(points[a : b + 1], cfg)

    # Remove one-point oscillation to avoid over-splitting.
    for _ in range(2):
        for i in range(1, n - 1):
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                labels[i] = labels[i - 1]

    return labels


def _forced_break_indices(
    points: np.ndarray,
    forced_break_points: list[tuple[float, float]] | None,
    tolerance_px: float,
) -> set[int]:
    out: set[int] = set()
    if not forced_break_points or len(points) < 3:
        return out

    tol = max(0.5, float(tolerance_px))
    for fx, fy in forced_break_points:
        target = np.array([float(fx), float(fy)], dtype=np.float64)
        d = np.linalg.norm(points - target, axis=1)
        idx = int(np.argmin(d))
        if float(d[idx]) <= tol and 0 < idx < len(points) - 1:
            out.add(idx)
    return out


def _dedupe_breaks(breaks: list[int], n_points: int) -> list[int]:
    out: list[int] = []
    prev = -1
    for b in sorted(int(v) for v in breaks):
        bb = max(0, min(n_points - 1, b))
        if bb == prev:
            continue
        out.append(bb)
        prev = bb
    if not out or out[0] != 0:
        out.insert(0, 0)
    if out[-1] != n_points - 1:
        out.append(n_points - 1)
    return out


def split_polyline_into_segments(
    polyline: Polyline2D,
    cfg: SegmentConfig,
    forced_break_points: list[tuple[float, float]] | None = None,
) -> list[Polyline2D]:
    points = _remove_duplicate_neighbors(polyline.as_array())
    if len(points) < 3 or not bool(cfg.enable):
        return [Polyline2D(points=[(float(x), float(y)) for x, y in points])]

    breaks: set[int] = {0, len(points) - 1}
    corner_breaks = _multiscale_corner_break_indices(points, cfg)
    if corner_breaks:
        breaks.update(corner_breaks)
    else:
        # Fallback to local corner detection when multi-scale evidence is weak.
        corner_thresh = max(5.0, float(cfg.corner_split_angle_deg))
        for i in range(1, len(points) - 1):
            if _turn_angle_deg(points[i - 1], points[i], points[i + 1]) >= corner_thresh:
                breaks.add(i)

    breaks.update(
        _forced_break_indices(
            points,
            forced_break_points=forced_break_points,
            tolerance_px=float(cfg.forced_break_tolerance_px),
        )
    )

    labels = _local_straight_labels(points, cfg)
    for i in range(2, len(points) - 1):
        if labels[i] != labels[i - 1]:
            breaks.add(i)

    ordered = _dedupe_breaks(list(breaks), len(points))
    ordered = _merge_near_breaks(points, ordered, float(cfg.break_merge_distance_px))
    min_points = max(2, int(cfg.min_segment_points))

    segments_arr: list[np.ndarray] = []
    for i in range(1, len(ordered)):
        a = ordered[i - 1]
        b = ordered[i]
        if b <= a:
            continue
        seg = points[a : b + 1]
        if len(seg) < 2:
            continue
        if segments_arr and len(seg) < min_points:
            segments_arr[-1] = np.vstack([segments_arr[-1], seg[1:]])
        else:
            segments_arr.append(seg)

    if not segments_arr:
        segments_arr = [points]

    if len(segments_arr) >= 2 and len(segments_arr[-1]) < min_points:
        tail = segments_arr.pop()
        segments_arr[-1] = np.vstack([segments_arr[-1], tail[1:]])

    out: list[Polyline2D] = []
    for seg in segments_arr:
        seg_clean = _remove_duplicate_neighbors(seg)
        if len(seg_clean) < 2:
            continue
        out.append(Polyline2D(points=[(float(x), float(y)) for x, y in seg_clean]))
    return out or [Polyline2D(points=[(float(x), float(y)) for x, y in points])]


def anchors_for_segment(
    polyline: Polyline2D,
    anchors: list[tuple[float, float]] | None,
    tolerance_px: float,
) -> list[tuple[float, float]]:
    arr = polyline.as_array()
    if len(arr) == 0:
        return []

    out: list[tuple[float, float]] = [
        (float(arr[0, 0]), float(arr[0, 1])),
        (float(arr[-1, 0]), float(arr[-1, 1])),
    ]
    if anchors:
        tol = max(0.5, float(tolerance_px))
        for ax, ay in anchors:
            target = np.array([float(ax), float(ay)], dtype=np.float64)
            d = np.linalg.norm(arr - target, axis=1)
            idx = int(np.argmin(d))
            if float(d[idx]) <= tol:
                out.append((float(arr[idx, 0]), float(arr[idx, 1])))

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in out:
        key = (round(float(x), 4), round(float(y), 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((float(x), float(y)))
    return deduped


class _Dsu:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def snap_segment_endpoints(
    segments: list[Polyline2D],
    tolerance_px: float,
) -> tuple[list[Polyline2D], list[tuple[int, int]], list[tuple[float, float]]]:
    if not segments:
        return [], [], []

    arrays = [seg.as_array() for seg in segments]
    endpoints: list[np.ndarray] = []
    for arr in arrays:
        if len(arr) < 2:
            continue
        endpoints.append(arr[0].copy())
        endpoints.append(arr[-1].copy())

    if not endpoints:
        return segments, [(-1, -1)] * len(segments), []

    tol = max(0.0, float(tolerance_px))
    dsu = _Dsu(len(endpoints))
    if tol > 0.0:
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if float(np.linalg.norm(endpoints[i] - endpoints[j])) <= tol:
                    dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(endpoints)):
        groups.setdefault(dsu.find(i), []).append(i)

    ordered_groups = sorted(groups.values(), key=lambda g: min(g))
    endpoint_to_node: dict[int, int] = {}
    node_centers: list[tuple[float, float]] = []
    for node_id, group in enumerate(ordered_groups):
        coords = np.asarray([endpoints[idx] for idx in group], dtype=np.float64)
        center = coords.mean(axis=0)
        node_centers.append((float(center[0]), float(center[1])))
        for idx in group:
            endpoint_to_node[idx] = node_id

    snapped_segments: list[Polyline2D] = []
    node_pairs: list[tuple[int, int]] = []
    ep_index = 0
    for arr in arrays:
        if len(arr) < 2:
            snapped_segments.append(Polyline2D(points=[(float(x), float(y)) for x, y in arr]))
            node_pairs.append((-1, -1))
            continue

        start_node = endpoint_to_node[ep_index]
        end_node = endpoint_to_node[ep_index + 1]
        ep_index += 2

        out = arr.copy()
        out[0] = np.asarray(node_centers[start_node], dtype=np.float64)
        out[-1] = np.asarray(node_centers[end_node], dtype=np.float64)
        out = _remove_duplicate_neighbors(out)

        snapped_segments.append(Polyline2D(points=[(float(x), float(y)) for x, y in out]))
        node_pairs.append((start_node, end_node))

    return snapped_segments, node_pairs, node_centers
