import numpy as np
import pytest

pytest.importorskip("scipy")

from sketch2rhino.config import PathExtractConfig
from sketch2rhino.topo.graph_build import build_stroke_graph
from sketch2rhino.topo.path_extract import (
    _build_super_graph,
    choose_main_component,
    extract_main_open_path,
    extract_open_paths,
)
from sketch2rhino.types import GraphEdge, StrokeGraph


def test_graph_and_path_extract_on_open_stroke():
    skel = np.zeros((40, 60), dtype=np.uint8)
    skel[20, 5:55] = 1
    skel[15:21, 30] = 1  # branch spur

    component = choose_main_component(skel, "largest")
    graph = build_stroke_graph(component)
    poly = extract_main_open_path(component, graph, PathExtractConfig())

    assert len(graph.node_points) >= 2
    assert len(graph.edges) >= 1
    assert len(poly.points) >= 10

    x0, _ = poly.points[0]
    x1, _ = poly.points[-1]
    assert abs(x1 - x0) > 20


def test_overpass_policy_prefers_straight_crossing_path():
    skel = np.zeros((40, 40), dtype=np.uint8)

    # Diagonal A
    for i in range(6, 34):
        skel[i, i] = 1
    # Diagonal B
    for i in range(6, 34):
        skel[i, 39 - i] = 1

    component = choose_main_component(skel, "largest")
    graph = build_stroke_graph(component)
    poly = extract_main_open_path(component, graph, PathExtractConfig(crossing_policy="overpass"))

    assert len(poly.points) > 10

    start = np.asarray(poly.points[0], dtype=float)
    end = np.asarray(poly.points[-1], dtype=float)
    endpoint_dist = float(np.linalg.norm(end - start))

    # Straight-through diagonals connect opposite corners (longer than adjacent-corner route).
    assert endpoint_dist > 25.0


def test_junction_split_breaks_crossing_into_branches():
    skel = np.zeros((40, 40), dtype=np.uint8)

    for i in range(6, 34):
        skel[i, i] = 1
    for i in range(6, 34):
        skel[i, 39 - i] = 1

    component = choose_main_component(skel, "largest")
    graph = build_stroke_graph(component)
    paths = extract_open_paths(
        component,
        graph,
        PathExtractConfig(crossing_policy="junction_split"),
        min_length_px=0.0,
        max_curves=None,
        include_loops=True,
    )

    assert len(paths) >= 4
    lengths = []
    for pl in paths:
        arr = np.asarray(pl.points, dtype=float)
        if len(arr) < 2:
            continue
        lengths.append(float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()))
    assert lengths
    # With split policy, no path should span both opposite corners.
    assert max(lengths) < 25.0


def test_extract_open_paths_returns_multiple_strokes():
    skel = np.zeros((60, 60), dtype=np.uint8)
    # Two disjoint open strokes.
    skel[20, 5:30] = 1
    skel[40, 30:55] = 1

    component = choose_main_component(skel, "all")
    graph = build_stroke_graph(component)
    paths = extract_open_paths(
        component,
        graph,
        PathExtractConfig(crossing_policy="overpass"),
        min_length_px=5.0,
        max_curves=10,
        include_loops=True,
    )

    assert len(paths) >= 2
    for poly in paths:
        assert len(poly.points) >= 2


def test_extract_open_paths_preserves_internal_cluster_edges():
    # Two nearby junction nodes become one super-node under default cluster
    # radius, but the short visible link between them should still be kept.
    node_points = {
        0: (10, 10),
        1: (10, 13),
        2: (8, 10),
        3: (12, 10),
        4: (8, 13),
        5: (12, 13),
    }
    adjacency = {
        0: {1, 2, 3},
        1: {0, 4, 5},
        2: {0},
        3: {0},
        4: {1},
        5: {1},
    }
    edges = [
        GraphEdge(start=0, end=1, pixels=[(10, 10), (10, 11), (10, 12), (10, 13)], length_px=3.0),
        GraphEdge(start=0, end=2, pixels=[(10, 10), (9, 10), (8, 10)], length_px=2.0),
        GraphEdge(start=0, end=3, pixels=[(10, 10), (11, 10), (12, 10)], length_px=2.0),
        GraphEdge(start=1, end=4, pixels=[(10, 13), (9, 13), (8, 13)], length_px=2.0),
        GraphEdge(start=1, end=5, pixels=[(10, 13), (11, 13), (12, 13)], length_px=2.0),
    ]
    graph = StrokeGraph(
        node_points=node_points,
        adjacency=adjacency,
        edges=edges,
        endpoints={2, 3, 4, 5},
        component_size_px=0,
    )

    skeleton = np.zeros((25, 25), dtype=np.uint8)
    for e in edges:
        for r, c in e.pixels:
            skeleton[r, c] = 1

    cfg = PathExtractConfig(cluster_radius_px=4.0, junction_bridge_max_px=12.0, crossing_policy="overpass")
    super_graph = _build_super_graph(graph, cfg)
    assert len(super_graph.internal_paths) >= 1

    paths = extract_open_paths(
        skeleton=skeleton,
        graph=graph,
        cfg=cfg,
        min_length_px=0.0,
        max_curves=None,
        include_loops=True,
    )

    found_internal = False
    for pl in paths:
        arr = np.asarray(pl.points, dtype=float)
        if len(arr) < 4:
            continue
        if not np.allclose(arr[:, 1], -10.0):
            continue
        x0 = float(arr[0, 0])
        x1 = float(arr[-1, 0])
        if {round(x0, 6), round(x1, 6)} == {10.0, 13.0}:
            found_internal = True
            break

    assert found_internal is True


def test_super_graph_filters_short_internal_cluster_edges():
    node_points = {
        0: (10, 10),
        1: (10, 11),
        2: (8, 10),
        3: (12, 10),
        4: (8, 11),
        5: (12, 11),
    }
    adjacency = {
        0: {1, 2, 3},
        1: {0, 4, 5},
        2: {0},
        3: {0},
        4: {1},
        5: {1},
    }
    edges = [
        GraphEdge(start=0, end=1, pixels=[(10, 10), (10, 11)], length_px=1.0),
        GraphEdge(start=0, end=2, pixels=[(10, 10), (9, 10), (8, 10)], length_px=2.0),
        GraphEdge(start=0, end=3, pixels=[(10, 10), (11, 10), (12, 10)], length_px=2.0),
        GraphEdge(start=1, end=4, pixels=[(10, 11), (9, 11), (8, 11)], length_px=2.0),
        GraphEdge(start=1, end=5, pixels=[(10, 11), (11, 11), (12, 11)], length_px=2.0),
    ]
    graph = StrokeGraph(
        node_points=node_points,
        adjacency=adjacency,
        edges=edges,
        endpoints={2, 3, 4, 5},
        component_size_px=0,
    )

    cfg = PathExtractConfig(
        cluster_radius_px=4.0,
        junction_bridge_max_px=12.0,
        crossing_policy="overpass",
        internal_path_min_length_px=2.0,
    )
    super_graph = _build_super_graph(graph, cfg)
    assert super_graph.internal_paths == []
