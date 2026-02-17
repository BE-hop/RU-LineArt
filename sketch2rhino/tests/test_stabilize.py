import numpy as np
import pytest

pytest.importorskip("scipy")

from sketch2rhino.config import StabilizeConfig
from sketch2rhino.fit.stroke_stabilize import stabilize_polyline
from sketch2rhino.types import Polyline2D


def _roughness(points: list[tuple[float, float]]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    d2 = np.diff(arr, n=2, axis=0)
    return float(np.mean(np.linalg.norm(d2, axis=1)))


def test_stabilize_reduces_jitter_and_keeps_endpoints():
    xs = np.linspace(0.0, 120.0, 241)
    base = 12.0 * np.sin(xs * 0.06)
    jitter = 1.4 * np.sin(xs * 1.9) + 0.8 * np.sin(xs * 3.7)
    ys = base + jitter
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = StabilizeConfig(
        enable=True,
        method="savgol",
        window=15,
        polyorder=2,
        passes=2,
        blend=0.8,
        resample_step_px=0.8,
        anchor_snap_radius_px=3.0,
    )
    out = stabilize_polyline(poly, cfg)

    assert out.points[0] == pytest.approx(poly.points[0], abs=1e-8)
    assert out.points[-1] == pytest.approx(poly.points[-1], abs=1e-8)
    assert _roughness(out.points) < _roughness(poly.points) * 0.65


def test_stabilize_respects_protected_anchor():
    xs = np.linspace(0.0, 100.0, 101)
    ys = 0.15 * (xs - 50.0) + 1.2 * np.sin(xs * 1.6)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    anchor = (50.0, 0.0)
    cfg = StabilizeConfig(
        enable=True,
        method="savgol",
        window=13,
        polyorder=2,
        passes=2,
        blend=0.7,
        resample_step_px=1.0,
        anchor_snap_radius_px=6.0,
    )
    out = stabilize_polyline(poly, cfg, protected_points=[anchor])

    arr = np.asarray(out.points, dtype=np.float64)
    d = np.linalg.norm(arr - np.array(anchor, dtype=np.float64), axis=1)
    assert float(d.min()) <= 1.0
