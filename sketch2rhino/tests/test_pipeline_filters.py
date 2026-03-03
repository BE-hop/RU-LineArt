import numpy as np

from sketch2rhino.pipeline import _adaptive_spur_min_length
from sketch2rhino.pipeline import _enhance_binary_for_geometry_mode
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


def test_enhance_binary_for_geometry_mode_keeps_thin_strokes():
    binary = np.zeros((40, 40), dtype=np.uint8)
    binary[20, 4:36] = 1
    binary[6:34, 10] = 1

    for mode in ("polyline_only", "nurbs_only", "mixed"):
        out = _enhance_binary_for_geometry_mode(binary, mode)
        assert out.dtype == np.uint8
        assert int(out.sum()) >= int(binary.sum())
        assert int(out[20, 4]) == 1
        assert int(out[20, 35]) == 1


def test_adaptive_spur_min_length_relaxes_when_ink_is_sparse():
    sparse = np.zeros((100, 100), dtype=np.uint8)
    sparse[50, 10:70] = 1  # 0.6% foreground
    dense = np.zeros((100, 100), dtype=np.uint8)
    dense[20:80, 20:80] = 1  # 36% foreground

    assert _adaptive_spur_min_length(sparse, 18) == 3
    assert _adaptive_spur_min_length(dense, 18) == 18


def test_enhance_binary_mixed_dense_avoids_false_bridge_between_parallel_lines():
    binary = np.zeros((120, 120), dtype=np.uint8)
    binary[:, 40] = 1
    binary[:, 42] = 1
    # Raise foreground ratio above dense threshold so mixed mode skips bridging close-op.
    binary[20:100, 70:115] = 1

    out = _enhance_binary_for_geometry_mode(binary, "mixed")
    assert out.dtype == np.uint8
    # Gap column between the two vertical lines should stay empty.
    assert int(np.count_nonzero(out[:, 41])) == 0
