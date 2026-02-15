import numpy as np
import pytest

pytest.importorskip("scipy")

from sketch2rhino.config import FitConfig
from sketch2rhino.types import Polyline2D
from sketch2rhino.fit.nurbs_fit import fit_open_nurbs


def test_nurbs_fit_control_point_budget():
    xs = np.linspace(0, 100, 150)
    ys = 10.0 * np.sin(xs * 0.08)
    poly = Polyline2D(points=[(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])

    cfg = FitConfig(max_control_points=25)
    nurbs = fit_open_nurbs(poly, cfg)

    assert nurbs.degree >= 1
    assert len(nurbs.control_points) <= 25
    assert len(nurbs.knots) > 0
