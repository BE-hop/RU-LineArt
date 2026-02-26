import numpy as np
import pytest

from sketch2rhino.config import SegmentConfig
from sketch2rhino.fit.nurbs_fit import should_use_polyline_geometry
from sketch2rhino.fit.segment_split import snap_segment_endpoints, split_polyline_into_segments
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
