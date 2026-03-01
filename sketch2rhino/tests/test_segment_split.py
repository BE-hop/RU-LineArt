import numpy as np
import pytest

from sketch2rhino.config import SegmentConfig
from sketch2rhino.fit.nurbs_fit import should_use_polyline_geometry
from sketch2rhino.fit.segment_split import (
    _window_is_straight,
    join_collinear_segments,
    snap_segment_endpoints,
    split_polyline_into_segments,
)
from sketch2rhino.types import Polyline2D
from sketch2rhino.config import FitConfig


def _mixed_line_arc_line_polyline() -> Polyline2D:
    line1 = [(float(x), 0.0) for x in np.linspace(0.0, 30.0, 16)]
    # quarter-like arc from (30,0) to (60,30), center at (30,30), radius=30
    theta = np.linspace(-np.pi / 2.0, 0.0, 22)
    arc = [(30.0 + 30.0 * np.cos(t), 30.0 + 30.0 * np.sin(t)) for t in theta]
    line2 = [(float(x), 30.0) for x in np.linspace(60.0, 95.0, 18)]
    points = line1 + arc[1:] + line2[1:]
    return Polyline2D(points=points)


def test_split_polyline_into_segments_for_mixed_geometry():
    polyline = _mixed_line_arc_line_polyline()
    seg_cfg = SegmentConfig(
        enable=True,
        corner_split_angle_deg=35.0,
        corner_scales_px=[4.0, 8.0, 16.0],
        min_segment_points=3,
        fillet_enable=True,
    )
    segments = split_polyline_into_segments(polyline, seg_cfg)

    assert len(segments) >= 3

    fit_cfg = FitConfig(max_control_points=50)
    fit_cfg.spline.mode = "auto"
    kinds = [should_use_polyline_geometry(seg, fit_cfg) for seg in segments]
    assert any(k for k in kinds)
    assert any(not k for k in kinds)


def test_split_respects_forced_break_points():
    points = [(float(x), 0.0) for x in np.linspace(0.0, 100.0, 51)]
    polyline = Polyline2D(points=points)
    seg_cfg = SegmentConfig(enable=True, corner_split_angle_deg=80.0, min_segment_points=2)
    segments = split_polyline_into_segments(
        polyline,
        seg_cfg,
        forced_break_points=[(50.0, 0.0)],
    )

    assert len(segments) >= 2
    mid = segments[0].points[-1]
    assert mid == pytest.approx(segments[1].points[0], abs=1e-8)


def test_snap_segment_endpoints_unifies_join_nodes():
    seg1 = Polyline2D(points=[(0.0, 0.0), (10.0, 0.0)])
    seg2 = Polyline2D(points=[(10.6, 0.2), (20.0, 5.0)])
    snapped, node_pairs, nodes = snap_segment_endpoints([seg1, seg2], tolerance_px=1.0)

    assert len(snapped) == 2
    assert len(node_pairs) == 2
    assert len(nodes) >= 3

    end1 = snapped[0].points[-1]
    start2 = snapped[1].points[0]
    assert end1 == pytest.approx(start2, abs=1e-8)
    assert node_pairs[0][1] == node_pairs[1][0]


def test_snap_segment_endpoints_keeps_transitive_cluster_connectivity():
    seg1 = Polyline2D(points=[(-10.0, 0.0), (0.0, 0.0)])
    seg2 = Polyline2D(points=[(0.8, 0.0), (12.0, 0.0)])
    seg3 = Polyline2D(points=[(1.6, 0.0), (14.0, 0.0)])

    snapped, node_pairs, _ = snap_segment_endpoints([seg1, seg2, seg3], tolerance_px=1.0)

    e1 = snapped[0].points[-1]
    s2 = snapped[1].points[0]
    s3 = snapped[2].points[0]
    assert e1 == pytest.approx(s2, abs=1e-8)
    assert e1 == pytest.approx(s3, abs=1e-8)
    assert node_pairs[0][1] == node_pairs[1][0] == node_pairs[2][0]


def test_join_collinear_segments_merges_degree2_chain():
    segments = [
        Polyline2D(points=[(0.0, 0.0), (5.0, 0.0)]),
        Polyline2D(points=[(5.0, 0.0), (10.0, 0.0)]),
        Polyline2D(points=[(10.0, 0.0), (15.0, 0.0)]),
    ]
    node_pairs = [(0, 1), (1, 2), (2, 3)]

    merged, merged_pairs, groups = join_collinear_segments(segments, node_pairs, angle_tolerance_deg=8.0)

    assert len(merged) == 1
    assert len(groups) == 1
    assert groups[0] == [0, 1, 2]
    assert merged_pairs[0] == (0, 3)
    assert merged[0].points[0] == pytest.approx((0.0, 0.0), abs=1e-8)
    assert merged[0].points[-1] == pytest.approx((15.0, 0.0), abs=1e-8)


def test_join_collinear_segments_keeps_junction_unmerged():
    segments = [
        Polyline2D(points=[(0.0, 0.0), (5.0, 0.0)]),
        Polyline2D(points=[(5.0, 0.0), (10.0, 0.0)]),
        Polyline2D(points=[(5.0, 0.0), (5.0, 4.0)]),
    ]
    node_pairs = [(0, 1), (1, 2), (1, 3)]

    merged, merged_pairs, groups = join_collinear_segments(segments, node_pairs, angle_tolerance_deg=8.0)

    assert len(merged) == 3
    assert len(groups) == 3
    assert sorted(len(g) for g in groups) == [1, 1, 1]
    assert sorted(merged_pairs) == sorted(node_pairs)


def test_window_straight_is_robust_to_single_outlier_point():
    xs = np.linspace(0.0, 120.0, 41)
    ys = np.zeros_like(xs)
    ys[len(ys) // 2] = 3.0
    points = np.column_stack([xs, ys]).astype(np.float64)

    cfg = SegmentConfig(
        straight_min_chord_px=8.0,
        straight_max_deviation_px=2.0,
        straight_max_deviation_ratio=0.0,
        straight_max_turn_deg=180.0,  # isolate deviation metric in this regression test
    )

    assert _window_is_straight(points, cfg) is True
