import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("scipy")

from sketch2rhino.config import AppConfig, FitConfig
from sketch2rhino.pipeline import (
    _fit_geometry_for_segment,
    _resolve_effective_component_choice,
    _resolve_effective_path_filter,
    _resolve_effective_segment_enable,
)
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


def test_segment_strategy_nurbs_only_forces_disable():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "nurbs_only"
    cfg.fit.segment.enable = True

    enabled, reason = _resolve_effective_segment_enable(cfg)

    assert enabled is False
    assert reason == "forced_off_by_geometry_mode_nurbs_only"


def test_segment_strategy_polyline_only_forces_enable():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "polyline_only"
    cfg.fit.segment.enable = False

    enabled, reason = _resolve_effective_segment_enable(cfg)

    assert enabled is True
    assert reason == "forced_on_by_geometry_mode_polyline_only"


def test_segment_strategy_mixed_follows_config():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "mixed"

    cfg.fit.segment.enable = False
    enabled0, reason0 = _resolve_effective_segment_enable(cfg)
    assert enabled0 is False
    assert reason0 == "from_fit.segment.enable"

    cfg.fit.segment.enable = True
    enabled1, reason1 = _resolve_effective_segment_enable(cfg)
    assert enabled1 is True
    assert reason1 == "from_fit.segment.enable"


def test_path_filter_strategy_polyline_only_disables_filter():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "polyline_only"
    cfg.output.multi.min_length_px = 12.0
    cfg.output.multi.max_curves = 77

    min_len, max_curves, reason = _resolve_effective_path_filter(cfg)

    assert min_len == 0.0
    assert max_curves == 0
    assert reason == "disabled_for_geometry_mode_polyline_only"


def test_path_filter_strategy_mixed_disables_filter_and_hard_cap():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "mixed"
    cfg.output.multi.min_length_px = 9.5
    cfg.output.multi.max_curves = 123

    min_len, max_curves, reason = _resolve_effective_path_filter(cfg)

    assert min_len == 0.0
    assert max_curves == 0
    assert reason == "disabled_for_geometry_mode_mixed"


def test_path_filter_strategy_nurbs_only_relaxes_filter_and_disables_hard_cap():
    cfg = AppConfig()
    cfg.fit.geometry_mode = "nurbs_only"
    cfg.output.multi.min_length_px = 5.0
    cfg.output.multi.max_curves = 66

    min_len, max_curves, reason = _resolve_effective_path_filter(cfg)

    assert min_len == 4.0
    assert max_curves == 0
    assert reason == "relaxed_for_geometry_mode_nurbs_only"


def test_component_choice_strategy_mixed_forces_all_for_multi():
    cfg = AppConfig()
    cfg.output.mode = "multi"
    cfg.fit.geometry_mode = "mixed"
    cfg.path_extract.choose_component = "largest"

    effective, reason = _resolve_effective_component_choice(cfg)

    assert effective == "all"
    assert reason == "forced_all_for_geometry_mode_mixed"


def test_component_choice_strategy_non_mixed_respects_config():
    cfg = AppConfig()
    cfg.output.mode = "multi"
    cfg.fit.geometry_mode = "polyline_only"
    cfg.path_extract.choose_component = "largest"

    effective, reason = _resolve_effective_component_choice(cfg)

    assert effective == "largest"
    assert reason == "from_path_extract.choose_component"
