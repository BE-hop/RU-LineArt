from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math

import numpy as np

from sketch2rhino.types import GraphEdge, SkeletonImage, StrokeGraph

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


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors8(pixel: Pixel, h: int, w: int) -> Iterable[Pixel]:
    r, c = pixel
    for dr, dc in _NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if _in_bounds(nr, nc, h, w):
            yield nr, nc


def _on_neighbors(pixel: Pixel, mask: np.ndarray) -> list[Pixel]:
    h, w = mask.shape
    return [p for p in _neighbors8(pixel, h, w) if mask[p[0], p[1]]]


def _edge_length(path: list[Pixel]) -> float:
    total = 0.0
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        total += math.hypot(r1 - r0, c1 - c0)
    return total


def build_stroke_graph(skeleton: SkeletonImage) -> StrokeGraph:
    mask = skeleton.astype(bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return StrokeGraph(node_points={}, adjacency={}, edges=[], endpoints=set(), component_size_px=0)

    degree_map = np.zeros_like(skeleton, dtype=np.uint8)
    for r, c in coords:
        neighbors = _on_neighbors((int(r), int(c)), mask)
        degree_map[r, c] = len(neighbors)

    node_pixels: set[Pixel] = {
        (int(r), int(c))
        for r, c in coords
        if degree_map[r, c] != 2
    }

    # Closed loops may have no degree!=2 nodes; inject one stable seed node.
    if not node_pixels:
        seed = tuple(int(v) for v in coords[np.lexsort((coords[:, 1], coords[:, 0]))[0]])
        node_pixels.add(seed)

    sorted_nodes = sorted(node_pixels)
    pixel_to_id = {p: i for i, p in enumerate(sorted_nodes)}
    node_points = {i: p for i, p in enumerate(sorted_nodes)}

    adjacency: dict[int, set[int]] = defaultdict(set)
    edges: list[GraphEdge] = []
    visited_segments: set[tuple[Pixel, Pixel]] = set()

    for start_pix, start_id in pixel_to_id.items():
        for nb in _on_neighbors(start_pix, mask):
            seg = tuple(sorted((start_pix, nb)))
            if seg in visited_segments:
                continue

            path = [start_pix, nb]
            visited_segments.add(seg)

            prev = start_pix
            curr = nb
            while curr not in node_pixels:
                candidates = [p for p in _on_neighbors(curr, mask) if p != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                seg2 = tuple(sorted((curr, nxt)))
                if seg2 in visited_segments:
                    break
                visited_segments.add(seg2)
                path.append(nxt)
                prev, curr = curr, nxt

            if curr not in pixel_to_id:
                continue

            end_id = pixel_to_id[curr]
            if start_id == end_id and len(path) < 3:
                continue

            edge = GraphEdge(start=start_id, end=end_id, pixels=path, length_px=_edge_length(path))
            edges.append(edge)
            adjacency[start_id].add(end_id)
            adjacency[end_id].add(start_id)

    endpoints = {
        nid
        for pix, nid in pixel_to_id.items()
        if degree_map[pix[0], pix[1]] == 1
    }

    return StrokeGraph(
        node_points=node_points,
        adjacency={k: set(v) for k, v in adjacency.items()},
        edges=edges,
        endpoints=endpoints,
        component_size_px=int(coords.shape[0]),
    )
