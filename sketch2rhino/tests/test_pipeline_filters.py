import numpy as np

from sketch2rhino.pipeline import _filter_multi_output_polylines
from sketch2rhino.pipeline import _polyline_pixel_coverage
from sketch2rhino.types import Polyline2D


def _line(length: float) -> Polyline2D:
    return Polyline2D(points=[(0.0, 0.0), (float(length), 0.0)])


def test_filter_multi_output_polylines_reports_min_and_max_drops():
    candidates = [_line(2.0), _line(5.0), _line(10.0), _line(20.0)]
    kept, stats, warnings = _filter_multi_output_polylines(
        candidates=candidates,
        min_length_px=6.0,
        max_curves=1,
        sort_by="length",
    )

    assert len(kept) == 1
    assert kept[0].points[-1][0] == 20.0

    assert stats["raw_candidate_count"] == 4
    assert stats["dropped_by_min_length"] == 2
    assert stats["kept_after_min_length"] == 2
    assert stats["dropped_by_max_curves"] == 1
    assert stats["kept_final_count"] == 1

    assert len(warnings) == 2
    assert "min_length_px" in warnings[0]
    assert "max_curves" in warnings[1]


def test_filter_multi_output_polylines_without_max_cap_keeps_all():
    candidates = [_line(4.0), _line(9.0), _line(3.0)]
    kept, stats, warnings = _filter_multi_output_polylines(
        candidates=candidates,
        min_length_px=0.0,
        max_curves=0,
        sort_by="length",
    )

    assert len(kept) == 3
    assert [pl.points[-1][0] for pl in kept] == [9.0, 4.0, 3.0]
    assert stats["dropped_by_min_length"] == 0
    assert stats["dropped_by_max_curves"] == 0
    assert warnings == []


def test_polyline_pixel_coverage_reports_recall_against_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 2:18] = 1
    mask[5:16, 5] = 1

    horizontal = Polyline2D(points=[(2.0, -10.0), (17.0, -10.0)])
    cov = _polyline_pixel_coverage([horizontal], mask)

    assert cov["total_px"] == int(mask.sum())
    assert cov["covered_px"] > 0
    assert 0.0 < float(cov["ratio"]) < 1.0
