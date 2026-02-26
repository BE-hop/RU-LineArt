import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("scipy")

from sketch2rhino.config import AppConfig, FitConfig
from sketch2rhino.pipeline import _fit_geometry_for_segment
from sketch2rhino.types import NurbsSpec, Polyline2D


def _near_straight_polyline() -> Polyline2D:
    xs = np.linspace(0.0, 80.0, 20)
    ys = 0.35 * np.sin(xs * 0.05)
    return Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])


def _curvy_polyline() -> Polyline2D:
    xs = np.linspace(0.0, 80.0, 40)
    ys = 12.0 * np.sin(xs * 0.1)
    return Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])


def test_fit_geometry_mode_rejects_invalid_value():
    with pytest.raises(ValidationError):
        FitConfig(geometry_mode="bad_mode")


def test_polyline_only_forces_polyline_even_for_curvy_stroke():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "polyline_only"

    geom, kind = _fit_geometry_for_segment(_curvy_polyline(), anchors=[], cfg=cfg)

    assert kind == "polyline"
    assert isinstance(geom, Polyline2D)


def test_nurbs_only_forces_nurbs_even_for_straight_stroke():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "nurbs_only"
    cfg.fit.spline.mode = "auto"

    geom, kind = _fit_geometry_for_segment(_near_straight_polyline(), anchors=[], cfg=cfg)

    assert kind == "nurbs"
    assert isinstance(geom, NurbsSpec)


def test_mixed_keeps_auto_classification():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "mixed"
    cfg.fit.spline.mode = "auto"
    cfg.fit.spline.hard_edge_straight_min_length_px = 8.0
    cfg.fit.segment.straight_min_chord_px = 8.0

    geom, kind = _fit_geometry_for_segment(_near_straight_polyline(), anchors=[], cfg=cfg)

    assert kind == "polyline"
    assert isinstance(geom, Polyline2D)
