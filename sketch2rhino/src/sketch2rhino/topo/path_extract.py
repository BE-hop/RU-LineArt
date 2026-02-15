from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage as ndi

from sketch2rhino.config import PathExtractConfig
from sketch2rhino.types import Polyline2D, SkeletonImage, StrokeGraph

_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

Pixel = tuple[int, int]


def _pixel_to_xy(pixel: Pixel) -> tuple[float, float]:
    r, c = pixel
    return float(c), float(-r)


def _neighbors_on(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    h, w = mask.shape
    r, c = pixel
    out: list[Pixel] = []
    for dr, dc in _NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def choose_main_component(skeleton: SkeletonImage, choose_component: str = "largest") -> SkeletonImage:
    if choose_component != "largest":
        return skeleton.copy()

    components = split_components(skeleton)
    if not components:
        return skeleton.copy()
    return components[0]


def split_components(skeleton: SkeletonImage) -> list[SkeletonImage]:
    labels, n_labels = ndi.label(skeleton > 0, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return []
    if n_labels == 1:
        return [skeleton.copy()]

    out: list[tuple[int, SkeletonImage]] = []
    for label in range(1, n_labels + 1):
        comp = (labels == label).astype(np.uint8)
        out.append((int(comp.sum()), comp))

    out.sort(key=lambda x: x[0], reverse=True)
    return [comp for _, comp in out]


def _build_pixel_adjacency(mask: np.ndarray) -> dict[Pixel, list[Pixel]]:
    coords = np.argwhere(mask)
    adjacency: dict[Pixel, list[Pixel]] = {}
    for r, c in coords:
        pix = (int(r), int(c))
        adjacency[pix] = _neighbors_on(pix, mask)
    return adjacency


def _edge_weight(a: Pixel, b: Pixel) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _dijkstra_pixels(
    start: Pixel,
    adjacency: dict[Pixel, list[Pixel]],
) -> tuple[dict[Pixel, float], dict[Pixel, Pixel]]:
    dist: dict[Pixel, float] = {start: 0.0}
    prev: dict[Pixel, Pixel] = {}
    heap: list[tuple[float, Pixel]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue

        for v in adjacency.get(u, []):
            nd = d + _edge_weight(u, v)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev


def _reconstruct_pixel_path(start: Pixel, end: Pixel, prev: dict[Pixel, Pixel]) -> list[Pixel]:
    out: list[Pixel] = [end]
    cur = end

    while cur != start:
        if cur not in prev:
            return []
        cur = prev[cur]
        out.append(cur)

    out.reverse()
    return out


def _path_length(path: list[Pixel]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        total += _edge_weight(path[i - 1], path[i])
    return total


def _edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return tuple(sorted((a, b)))


def _unvisited_neighbors(
    pixel: Pixel,
    adjacency: dict[Pixel, list[Pixel]],
    visited_edges: set[tuple[Pixel, Pixel]],
    prev: Pixel | None = None,
) -> list[Pixel]:
    out: list[Pixel] = []
    for nb in adjacency.get(pixel, []):
        if prev is not None and nb == prev:
            continue
        if _edge_key(pixel, nb) in visited_edges:
            continue
        out.append(nb)
    return out


def _trace_from_endpoint_overpass(
    start: Pixel,
    adjacency: dict[Pixel, list[Pixel]],
) -> list[Pixel]:
    nbs = adjacency.get(start, [])
    if not nbs:
        return [start]

    prev = start
    curr = nbs[0]
    path: list[Pixel] = [start, curr]
    visited_edges: set[tuple[Pixel, Pixel]] = {tuple(sorted((start, curr)))}
    max_steps = max(8, len(adjacency) * 3)

    for _ in range(max_steps):
        candidates = [p for p in adjacency.get(curr, []) if p != prev]
        if not candidates:
            break

        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            incoming = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)

            def score(p: Pixel) -> float:
                out = np.array([p[0] - curr[0], p[1] - curr[1]], dtype=np.float64)
                denom = float(np.linalg.norm(incoming) * np.linalg.norm(out))
                if denom == 0.0:
                    return -10.0
                cos = float(np.dot(incoming, out) / denom)
                edge = tuple(sorted((curr, p)))
                if edge in visited_edges:
                    cos -= 2.0
                return cos

            candidates.sort(key=score, reverse=True)
            nxt = candidates[0]

        edge = tuple(sorted((curr, nxt)))
        if edge in visited_edges:
            break
        visited_edges.add(edge)

        prev, curr = curr, nxt
        path.append(curr)

        if len(adjacency.get(curr, [])) <= 1:
            break

    return path


def _trace_path_from_edge(
    start: Pixel,
    first_next: Pixel,
    adjacency: dict[Pixel, list[Pixel]],
    visited_edges: set[tuple[Pixel, Pixel]],
    crossing_policy: str,
) -> list[Pixel]:
    prev = start
    curr = first_next
    path: list[Pixel] = [start, curr]
    visited_edges.add(_edge_key(start, curr))
    max_steps = max(8, len(adjacency) * 3)

    for _ in range(max_steps):
        candidates = _unvisited_neighbors(curr, adjacency, visited_edges, prev=prev)
        if not candidates:
            break

        if len(candidates) == 1:
            nxt = candidates[0]
        elif crossing_policy == "overpass":
            incoming = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)

            def score(p: Pixel) -> float:
                out = np.array([p[0] - curr[0], p[1] - curr[1]], dtype=np.float64)
                denom = float(np.linalg.norm(incoming) * np.linalg.norm(out))
                if denom == 0.0:
                    return -10.0
                return float(np.dot(incoming, out) / denom)

            candidates.sort(key=score, reverse=True)
            nxt = candidates[0]
        else:
            nxt = sorted(candidates)[0]

        ek = _edge_key(curr, nxt)
        if ek in visited_edges:
            break
        visited_edges.add(ek)
        prev, curr = curr, nxt
        path.append(curr)

        # If the trace closes a loop, keep one closed cycle and let caller cut to open.
        if curr == start:
            break

    return path


def _cut_loop_open(path: list[Pixel], loop_cut: str) -> list[Pixel]:
    if len(path) < 3 or path[0] != path[-1]:
        return path

    cycle = path[:-1]
    if not cycle:
        return path

    if loop_cut == "topmost":
        idx = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    elif loop_cut == "leftmost":
        idx = min(range(len(cycle)), key=lambda i: (cycle[i][1], cycle[i][0]))
    else:
        # min_curvature: prefer a straight-ish point as cut location.
        if len(cycle) < 3:
            idx = 0
        else:
            scores: list[float] = []
            n = len(cycle)
            for i in range(n):
                p0 = np.array(cycle[(i - 1) % n], dtype=np.float64)
                p1 = np.array(cycle[i], dtype=np.float64)
                p2 = np.array(cycle[(i + 1) % n], dtype=np.float64)
                v1 = p1 - p0
                v2 = p2 - p1
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom == 0.0:
                    scores.append(-1.0)
                else:
                    scores.append(float(np.dot(v1, v2) / denom))
            idx = int(np.argmax(scores))

    rotated = cycle[idx:] + cycle[:idx]
    return rotated


def _polyline_len_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _extract_endpoint_pair_paths(adjacency: dict[Pixel, list[Pixel]]) -> list[list[Pixel]]:
    endpoints = sorted([pix for pix, nbs in adjacency.items() if len(nbs) == 1])
    if len(endpoints) < 2:
        return []

    dist_cache: dict[Pixel, dict[Pixel, float]] = {}
    prev_cache: dict[Pixel, dict[Pixel, Pixel]] = {}
    for ep in endpoints:
        dist, prev = _dijkstra_pixels(ep, adjacency)
        dist_cache[ep] = dist
        prev_cache[ep] = prev

    remaining = set(endpoints)
    pairs: list[tuple[Pixel, Pixel, float]] = []

    while len(remaining) >= 2:
        ordered = sorted(remaining)
        best_pair: tuple[Pixel, Pixel, float] | None = None
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                d = dist_cache.get(a, {}).get(b)
                if d is None:
                    continue
                if best_pair is None or d > best_pair[2]:
                    best_pair = (a, b, float(d))

        if best_pair is None:
            break

        a, b, d = best_pair
        pairs.append((a, b, d))
        remaining.remove(a)
        remaining.remove(b)

    paths: list[list[Pixel]] = []
    for a, b, _ in pairs:
        prev = prev_cache[a]
        path = _reconstruct_pixel_path(a, b, prev)
        if len(path) >= 2:
            paths.append(path)

    return paths


def extract_open_paths(
    skeleton: SkeletonImage,
    graph: StrokeGraph,
    cfg: PathExtractConfig,
    min_length_px: float = 0.0,
    max_curves: int | None = None,
    include_loops: bool = True,
) -> list[Polyline2D]:
    del graph

    mask = skeleton.astype(bool)
    if int(mask.sum()) == 0:
        return []

    adjacency = _build_pixel_adjacency(mask)
    pixel_paths = _extract_endpoint_pair_paths(adjacency)

    if not pixel_paths and include_loops:
        loop_path = _trace_loop(mask, cfg.loop_cut)
        if len(loop_path) >= 2:
            pixel_paths = [loop_path]

    out: list[Polyline2D] = []
    for path in pixel_paths:
        if len(path) < 2:
            continue
        if path[0] == path[-1]:
            path = _cut_loop_open(path, cfg.loop_cut)
            if len(path) < 2:
                continue

        points = [_pixel_to_xy(p) for p in path]
        if _polyline_len_xy(points) < float(min_length_px):
            continue
        out.append(Polyline2D(points=points))

    if out and max_curves is not None and max_curves > 0:
        out.sort(key=lambda pl: _polyline_len_xy(pl.points), reverse=True)
        out = out[:max_curves]

    return out


def _extract_overpass_main_path(adjacency: dict[Pixel, list[Pixel]], endpoints: list[Pixel]) -> list[Pixel]:
    best_path: list[Pixel] = []
    best_len = -1.0

    # Track best candidate for each unordered endpoint pair to avoid duplicates.
    per_pair_best: dict[tuple[Pixel, Pixel], tuple[float, list[Pixel]]] = {}

    for ep in endpoints:
        candidate = _trace_from_endpoint_overpass(ep, adjacency)
        if len(candidate) < 2:
            continue

        a, b = candidate[0], candidate[-1]
        pair = tuple(sorted((a, b)))
        plen = _path_length(candidate)

        prev_best = per_pair_best.get(pair)
        if prev_best is None or plen > prev_best[0]:
            per_pair_best[pair] = (plen, candidate)

    for plen, path in per_pair_best.values():
        if plen > best_len:
            best_len = plen
            best_path = path

    return best_path


def _trace_loop(mask: np.ndarray, loop_cut: str) -> list[Pixel]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []

    pixels = [(int(r), int(c)) for r, c in coords]
    if loop_cut == "topmost":
        start = min(pixels, key=lambda p: (p[0], p[1]))
    else:
        start = min(pixels, key=lambda p: (p[1], p[0]))

    path: list[Pixel] = [start]
    prev: Pixel | None = None
    curr = start
    used_edges: set[tuple[Pixel, Pixel]] = set()

    for _ in range(len(pixels) + 5):
        nbs = _neighbors_on(curr, mask)
        candidates = [p for p in nbs if prev is None or p != prev]
        if not candidates:
            break

        if prev is None:
            nxt = candidates[0]
        else:
            in_vec = (curr[0] - prev[0], curr[1] - prev[1])

            def score(p: Pixel) -> float:
                out_vec = (p[0] - curr[0], p[1] - curr[1])
                dot = in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]
                norm = math.hypot(*in_vec) * math.hypot(*out_vec)
                if norm == 0:
                    return -1.0
                return dot / norm

            candidates.sort(key=score, reverse=True)
            nxt = candidates[0]

        edge = tuple(sorted((curr, nxt)))
        if edge in used_edges:
            break
        used_edges.add(edge)

        prev, curr = curr, nxt
        if curr == start:
            break
        path.append(curr)

    return path


def extract_main_open_path(
    skeleton: SkeletonImage,
    graph: StrokeGraph,
    cfg: PathExtractConfig,
) -> Polyline2D:
    paths = extract_open_paths(
        skeleton=skeleton,
        graph=graph,
        cfg=cfg,
        min_length_px=0.0,
        max_curves=1,
        include_loops=True,
    )
    if not paths:
        raise ValueError("Empty skeleton component")
    return paths[0]
