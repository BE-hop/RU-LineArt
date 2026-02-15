import pytest

from sketch2rhino.config import ExportConfig
from sketch2rhino.fit.nurbs_fit import fit_open_nurbs
from sketch2rhino.types import Polyline2D
from sketch2rhino.config import FitConfig

rhino3dm = pytest.importorskip("rhino3dm")
from sketch2rhino.export.rhino3dm_writer import write_3dm, write_3dm_many


def test_export_writes_3dm(tmp_path):
    poly = Polyline2D(points=[(0.0, 0.0), (20.0, 5.0), (40.0, -4.0), (60.0, 0.0)])
    nurbs = fit_open_nurbs(poly, FitConfig(max_control_points=10))

    out = tmp_path / "curve.3dm"
    result = write_3dm(nurbs, out, ExportConfig())

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0

    model = rhino3dm.File3dm.Read(str(out))
    assert model is not None
    assert len(model.Objects) == 1

    geometry = model.Objects[0].Geometry
    assert hasattr(geometry, "IsClosed")
    assert geometry.IsClosed is False


def test_export_writes_multiple_curves(tmp_path):
    poly1 = Polyline2D(points=[(0.0, 0.0), (20.0, 0.0), (40.0, 10.0)])
    poly2 = Polyline2D(points=[(0.0, -20.0), (20.0, -30.0), (40.0, -20.0)])
    nurbs1 = fit_open_nurbs(poly1, FitConfig(max_control_points=10))
    nurbs2 = fit_open_nurbs(poly2, FitConfig(max_control_points=10))

    out = tmp_path / "curves.3dm"
    result = write_3dm_many(specs=[nurbs1, nurbs2], output_path=out, cfg=ExportConfig())

    assert result.output_path.exists()
    model = rhino3dm.File3dm.Read(str(out))
    assert model is not None
    assert len(model.Objects) == 2
    assert model.Objects[0].Attributes.Name.startswith("stroke_")
    assert model.Objects[1].Attributes.Name.startswith("stroke_")
