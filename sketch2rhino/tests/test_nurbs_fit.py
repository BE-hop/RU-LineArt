import numpy as np
import pytest

pytest.importorskip("scipy")

from sketch2rhino.config import FitConfig
from sketch2rhino.types import Polyline2D
from sketch2rhino.fit.nurbs_fit import fit_open_nurbs, fit_open_polyline, should_use_polyline_geometry


def test_nurbs_fit_control_point_budget():
    xs = np.linspace(0, 100, 150)
    ys = 10.0 * np.sin(xs * 0.08)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = FitConfig(max_control_points=25)
    nurbs = fit_open_nurbs(poly, cfg)

    assert nurbs.degree >= 1
    assert len(nurbs.control_points) <= 25
    assert len(nurbs.knots) > 0


def test_nurbs_fit_auto_mode_uses_degree1_for_rect_like_path():
    poly = Polyline2D(
        points=[
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 25.0),
            (0.0, 25.0),
            (0.0, 0.0),
        ]
    )

    cfg = FitConfig(max_control_points=20)
    cfg.spline.mode = "auto"
    nurbs = fit_open_nurbs(poly, cfg)

    assert nurbs.degree == 1
    assert len(nurbs.control_points) <= 20


def test_auto_mode_detects_near_straight_as_polyline():
    xs = np.linspace(0.0, 120.0, 60)
    ys = 0.6 * np.sin(xs * 0.05)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = FitConfig(max_control_points=30)
    cfg.spline.mode = "auto"
    assert should_use_polyline_geometry(poly, cfg) is True

    reduced = fit_open_polyline(poly, cfg)
    assert len(reduced.points) >= 2
    assert len(reduced.points) <= 30


def test_fit_open_polyline_collapses_near_straight_to_two_points():
    xs = np.linspace(0.0, 60.0, 40)
    ys = 0.4 * np.sin(xs * 0.07)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = FitConfig(max_control_points=50)
    cfg.spline.mode = "auto"
    cfg.spline.hard_edge_straight_min_length_px = 8.0
    cfg.segment.straight_min_chord_px = 8.0

    assert should_use_polyline_geometry(poly, cfg) is True
    reduced = fit_open_polyline(poly, cfg)
    assert len(reduced.points) == 2


def test_short_straight_prefers_polyline_but_not_forced_two_points():
    points = [(0.0, 0.0), (2.0, 0.1), (4.0, 0.0), (6.0, -0.1)]
    poly = Polyline2D(points=points)

    cfg = FitConfig(max_control_points=50)
    cfg.spline.mode = "auto"
    cfg.spline.hard_edge_straight_min_length_px = 20.0
    cfg.segment.straight_min_chord_px = 12.0

    assert should_use_polyline_geometry(poly, cfg) is True
    reduced = fit_open_polyline(poly, cfg)
    assert len(reduced.points) >= 3


def test_auto_mode_keeps_curvy_shape_as_nurbs():
    xs = np.linspace(0.0, 120.0, 80)
    ys = 15.0 * np.sin(xs * 0.08)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = FitConfig(max_control_points=30)
    cfg.spline.mode = "auto"
    assert should_use_polyline_geometry(poly, cfg) is False
