import numpy as np
import pytest

pytest.importorskip("scipy")

from sketch2rhino.config import PathExtractConfig
from sketch2rhino.topo.graph_build import build_stroke_graph
from sketch2rhino.topo.path_extract import choose_main_component, extract_main_open_path, extract_open_paths


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
