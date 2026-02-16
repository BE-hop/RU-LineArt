from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class _SuperEdge:
    a: int
    b: int
    pixels: list[Pixel]  # oriented from a -> b
    length_px: float


@dataclass(slots=True)
class _SuperGraph:
    edges: list[_SuperEdge]
    incident: dict[int, list[int]]
    centers: dict[int, tuple[float, float]]


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

    for _ in range(len(pixels) + 8):
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
                if norm == 0.0:
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
            path.append(curr)
            break
        path.append(curr)

    return path


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


def _polyline_len_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _cut_loop_open(path: list[Pixel], loop_cut: str) -> list[Pixel]:
    if len(path) < 3:
        return path

    closed = path[0] == path[-1]
    cycle = path[:-1] if closed else path
    if len(cycle) < 2:
        return path

    if loop_cut == "topmost":
        idx = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    elif loop_cut == "leftmost":
        idx = min(range(len(cycle)), key=lambda i: (cycle[i][1], cycle[i][0]))
    else:
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

    return cycle[idx:] + cycle[:idx]


def _build_super_graph(graph: StrokeGraph, cfg: PathExtractConfig) -> _SuperGraph:
    if not graph.node_points:
        return _SuperGraph(edges=[], incident={}, centers={})

    node_ids = sorted(graph.node_points.keys())
    deg = {nid: len(graph.adjacency.get(nid, set())) for nid in node_ids}
    junctions = [nid for nid in node_ids if deg[nid] >= 3]
    radius = max(1.0, float(getattr(cfg, "cluster_radius_px", 4.0)))

    parent = {nid: nid for nid in node_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # Spatial clustering of junction pixels.
    for i in range(len(junctions)):
        a = junctions[i]
        ar, ac = graph.node_points[a]
        for j in range(i + 1, len(junctions)):
            b = junctions[j]
            br, bc = graph.node_points[b]
            if math.hypot(float(ar - br), float(ac - bc)) <= radius:
                union(a, b)

    # Also merge tiny edges inside junction neighborhoods.
    short_thresh = max(2.0, 0.75 * radius)
    for edge in graph.edges:
        if edge.length_px <= short_thresh and deg.get(edge.start, 0) >= 3 and deg.get(edge.end, 0) >= 3:
            union(edge.start, edge.end)

    # Merge nearby junction clusters connected by a short bridge. This prevents
    # one visual crossing from being split into multiple super-nodes.
    bridge_thresh = max(short_thresh, float(getattr(cfg, "junction_bridge_max_px", 12.0)))
    for edge in graph.edges:
        if edge.length_px <= bridge_thresh and deg.get(edge.start, 0) >= 3 and deg.get(edge.end, 0) >= 3:
            union(edge.start, edge.end)

    root_key: dict[int, tuple[str, int]] = {}
    for nid in node_ids:
        if deg[nid] >= 3:
            root_key[nid] = ("j", find(nid))
        else:
            root_key[nid] = ("n", nid)

    key_to_super: dict[tuple[str, int], int] = {}
    super_members: dict[int, list[int]] = {}
    for nid in node_ids:
        key = root_key[nid]
        sid = key_to_super.setdefault(key, len(key_to_super))
        super_members.setdefault(sid, []).append(nid)

    centers: dict[int, tuple[float, float]] = {}
    for sid, members in super_members.items():
        pts = np.asarray([graph.node_points[m] for m in members], dtype=np.float64)
        centers[sid] = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    # Keep parallel edges between the same super-node pair: they may represent
    # distinct strokes crossing the same junction neighborhood.
    edges: list[_SuperEdge] = []
    for edge in graph.edges:
        sa = key_to_super[root_key[edge.start]]
        sb = key_to_super[root_key[edge.end]]
        if sa == sb:
            continue

        a, b = (sa, sb) if sa < sb else (sb, sa)
        if sa == a and sb == b:
            pixels = edge.pixels
        else:
            pixels = list(reversed(edge.pixels))

        edges.append(_SuperEdge(a=a, b=b, pixels=pixels, length_px=float(edge.length_px)))

    incident: dict[int, list[int]] = {sid: [] for sid in centers}
    for eid, edge in enumerate(edges):
        incident.setdefault(edge.a, []).append(eid)
        incident.setdefault(edge.b, []).append(eid)

    return _SuperGraph(edges=edges, incident=incident, centers=centers)


def _edge_pixels_from_node(edge: _SuperEdge, node: int) -> tuple[list[Pixel], int]:
    if node == edge.a:
        return edge.pixels, edge.b
    return list(reversed(edge.pixels)), edge.a


def _edge_outward_direction(edge: _SuperEdge, node: int, centers: dict[int, tuple[float, float]], tangent_k: int) -> np.ndarray:
    pixels, other = _edge_pixels_from_node(edge, node)
    if len(pixels) >= 2:
        k = min(len(pixels) - 1, max(1, int(tangent_k)))
        p0 = np.array(pixels[0], dtype=np.float64)
        p1 = np.array(pixels[k], dtype=np.float64)
        vec = p1 - p0
    else:
        c0 = np.array(centers[node], dtype=np.float64)
        c1 = np.array(centers[other], dtype=np.float64)
        vec = c1 - c0

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return vec / norm


def _build_continuation_map(super_graph: _SuperGraph, cfg: PathExtractConfig) -> dict[tuple[int, int], int]:
    continuation: dict[tuple[int, int], int] = {}
    tangent_k = int(getattr(cfg, "tangent_k", 10))

    def edge_pair_score(node: int, e1: int, e2: int) -> float:
        v1 = _edge_outward_direction(super_graph.edges[e1], node, super_graph.centers, tangent_k)
        v2 = _edge_outward_direction(super_graph.edges[e2], node, super_graph.centers, tangent_k)
        # Lower is better. Opposite directions (straight-through) -> dot ~= -1.
        return float(np.dot(v1, v2))

    def solve_pairs_exact(node: int, eids: list[int]) -> list[tuple[int, int]]:
        # Brute-force with memoization; degrees here are usually small.
        sorted_eids = tuple(sorted(eids))
        cache: dict[tuple[int, ...], tuple[float, tuple[tuple[int, int], ...]]] = {}

        def solve_even(state: tuple[int, ...]) -> tuple[float, tuple[tuple[int, int], ...]]:
            if not state:
                return 0.0, tuple()
            if state in cache:
                return cache[state]

            first = state[0]
            best_cost = float("inf")
            best_pairs: tuple[tuple[int, int], ...] = tuple()

            for i in range(1, len(state)):
                second = state[i]
                rest = state[1:i] + state[i + 1 :]
                rest_cost, rest_pairs = solve_even(rest)
                pair_cost = edge_pair_score(node, first, second)
                total = pair_cost + rest_cost
                if total < best_cost:
                    pair = (first, second)
                    best_cost = total
                    best_pairs = (pair, *rest_pairs)

            cache[state] = (best_cost, best_pairs)
            return cache[state]

        if len(sorted_eids) % 2 == 0:
            return list(solve_even(sorted_eids)[1])

        # Odd degree: leave one branch unmatched, pair the rest optimally.
        best_cost = float("inf")
        best_pairs: list[tuple[int, int]] = []
        best_left_len = float("inf")
        for leftover in sorted_eids:
            rest = tuple(e for e in sorted_eids if e != leftover)
            cost, pairs = solve_even(rest)
            left_len = super_graph.edges[leftover].length_px
            if cost < best_cost or (math.isclose(cost, best_cost) and left_len < best_left_len):
                best_cost = cost
                best_pairs = list(pairs)
                best_left_len = left_len
        return best_pairs

    def solve_pairs_greedy(node: int, eids: list[int]) -> list[tuple[int, int]]:
        remaining = list(eids)
        pairs: list[tuple[int, int]] = []
        while len(remaining) >= 2:
            best_pair: tuple[int, int] | None = None
            best_score = float("inf")
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    e1 = remaining[i]
                    e2 = remaining[j]
                    score = edge_pair_score(node, e1, e2)
                    if score < best_score:
                        best_score = score
                        best_pair = (e1, e2)
            if best_pair is None:
                break
            e1, e2 = best_pair
            pairs.append((e1, e2))
            remaining.remove(e1)
            remaining.remove(e2)
        return pairs

    for node, eids in super_graph.incident.items():
        if len(eids) < 2:
            continue
        if len(eids) <= 10:
            pairs = solve_pairs_exact(node, eids)
        else:
            pairs = solve_pairs_greedy(node, eids)

        for e1, e2 in pairs:
            continuation[(node, e1)] = e2
            continuation[(node, e2)] = e1

    return continuation


def _trace_stroke_from(
    start_node: int,
    start_edge: int,
    super_graph: _SuperGraph,
    continuation: dict[tuple[int, int], int],
    visited_edges: set[int],
) -> list[Pixel]:
    path: list[Pixel] = []
    curr_node = start_node
    curr_edge = start_edge
    step_cap = max(16, len(super_graph.edges) * 3)

    for _ in range(step_cap):
        if curr_edge in visited_edges:
            break
        visited_edges.add(curr_edge)

        edge = super_graph.edges[curr_edge]
        seg, next_node = _edge_pixels_from_node(edge, curr_node)

        if not path:
            path.extend(seg)
        else:
            path.extend(seg[1:])

        next_edge = continuation.get((next_node, curr_edge))
        if next_edge is None or next_edge in visited_edges:
            break

        curr_node = next_node
        curr_edge = next_edge

    return path


def extract_open_paths(
    skeleton: SkeletonImage,
    graph: StrokeGraph,
    cfg: PathExtractConfig,
    min_length_px: float = 0.0,
    max_curves: int | None = None,
    include_loops: bool = True,
) -> list[Polyline2D]:
    mask = skeleton.astype(bool)
    if int(mask.sum()) == 0:
        return []

    super_graph = _build_super_graph(graph, cfg)

    # Fallback for pure loops / degenerate graphs.
    if not super_graph.edges:
        if not include_loops:
            return []
        loop_path = _trace_loop(mask, cfg.loop_cut)
        if len(loop_path) < 2:
            return []
        open_loop = _cut_loop_open(loop_path, cfg.loop_cut)
        points = [_pixel_to_xy(p) for p in open_loop]
        return [Polyline2D(points=points)] if _polyline_len_xy(points) >= float(min_length_px) else []

    continuation = _build_continuation_map(super_graph, cfg)
    visited_edges: set[int] = set()
    pixel_paths: list[list[Pixel]] = []

    endpoints = sorted([node for node, eids in super_graph.incident.items() if len(eids) == 1])

    # Trace from endpoints first (open strokes).
    for node in endpoints:
        for eid in super_graph.incident.get(node, []):
            if eid in visited_edges:
                continue
            path = _trace_stroke_from(node, eid, super_graph, continuation, visited_edges)
            if len(path) >= 2:
                pixel_paths.append(path)

    # Then consume leftover edges (loops or unmatched branches).
    for eid, edge in enumerate(super_graph.edges):
        if eid in visited_edges:
            continue
        path = _trace_stroke_from(edge.a, eid, super_graph, continuation, visited_edges)
        if len(path) >= 2:
            pixel_paths.append(path)

    out: list[Polyline2D] = []
    for path in pixel_paths:
        if len(path) < 2:
            continue

        if path[0] == path[-1]:
            if not include_loops:
                continue
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
        max_curves=None,
        include_loops=True,
    )
    if not paths:
        raise ValueError("Empty skeleton component")

    paths.sort(key=lambda pl: _polyline_len_xy(pl.points), reverse=True)
    return paths[0]


def extract_junction_centers(
    graph: StrokeGraph,
    cfg: PathExtractConfig,
) -> list[tuple[float, float]]:
    """Return stable XY junction anchors from the same super-graph used for path extraction."""
    super_graph = _build_super_graph(graph, cfg)
    out: list[tuple[float, float]] = []
    for node, eids in super_graph.incident.items():
        if len(eids) < 3:
            continue
        r, c = super_graph.centers[node]
        out.append((float(c), float(-r)))
    return out
