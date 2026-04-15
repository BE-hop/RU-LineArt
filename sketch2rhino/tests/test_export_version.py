import pytest

from sketch2rhino.config import ExportConfig
from sketch2rhino.export.rhino3dm_writer import write_3dm
from sketch2rhino.types import Polyline2D

rhino3dm = pytest.importorskip("rhino3dm")


def test_export_defaults_to_rhino7_archive(tmp_path):
    polyline = Polyline2D(points=[(0.0, 0.0), (10.0, 2.0), (20.0, 0.0)])
    out = tmp_path / "default_v7.3dm"

    write_3dm(polyline, out, ExportConfig())

    model = rhino3dm.File3dm.Read(str(out))
    assert model is not None
    assert int(model.ArchiveVersion) == 70


def test_export_can_write_rhino8_archive(tmp_path):
    polyline = Polyline2D(points=[(0.0, 0.0), (10.0, 2.0), (20.0, 0.0)])
    out = tmp_path / "explicit_v8.3dm"

    write_3dm(polyline, out, ExportConfig(file_version=8))

    model = rhino3dm.File3dm.Read(str(out))
    assert model is not None
    assert int(model.ArchiveVersion) == 80
