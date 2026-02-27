from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DenoiseConfig(BaseModel):
    enable: bool = False
    method: str = "gaussian"
    strength: float = 1.0


class BinarizeConfig(BaseModel):
    method: str = "otsu"
    block_size: int = 31
    C: int = 7
    otsu_offset: float = 0.0


class MorphConfig(BaseModel):
    erode_iter: int = 0
    close_iter: int = 0
    open_iter: int = 0


class PreprocessConfig(BaseModel):
    target_black_on_white: bool = True
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    binarize: BinarizeConfig = Field(default_factory=BinarizeConfig)
    morph: MorphConfig = Field(default_factory=MorphConfig)


class SkeletonConfig(BaseModel):
    prune_spurs: bool = True
    spur_min_length_px: int = 18


class PathExtractConfig(BaseModel):
    mode: str = "main_open_path"
    crossing_policy: str = "overpass"
    choose_component: str = "largest"
    loop_cut: str = "min_curvature"
    cluster_radius_px: float = 4.0
    junction_bridge_max_px: float = 12.0
    tangent_k: int = 10
    internal_path_min_length_px: float = 0.0


class SimplifyConfig(BaseModel):
    method: str = "rdp"
    epsilon_px: float = 1.2
    smooth_enable: bool = False
    smooth_window: int = 7
    smooth_passes: int = 1


class StabilizeConfig(BaseModel):
    enable: bool = False
    method: str = "savgol"
    window: int = 11
    polyorder: int = 2
    passes: int = 2
    blend: float = 0.7
    resample_step_px: float = 1.0
    anchor_snap_radius_px: float = 4.0


class SplineConfig(BaseModel):
    mode: str = "auto"  # "auto" | "smooth" | "hard_edge"
    smoothing: float = 6.0
    max_iter: int = 6
    fallback_if_fail: bool = True
    anchor_correction_enable: bool = True
    anchor_tolerance_px: float = 2.0
    anchor_snap_max_dist_px: float = 10.0
    anchor_weight: float = 20.0
    anchor_neighbor_weight: float = 6.0
    anchor_refit_max_iter: int = 2
    anchor_sample_count: int = 400
    hard_edge_corner_angle_deg: float = 45.0
    hard_edge_right_angle_tolerance_deg: float = 30.0
    hard_edge_min_corners: int = 3
    hard_edge_max_vertices: int = 20
    hard_edge_corner_ratio_min: float = 0.35
    hard_edge_right_angle_ratio_min: float = 0.4
    hard_edge_straight_min_length_px: float = 20.0
    hard_edge_straight_max_deviation_px: float = 2.0
    hard_edge_straight_max_deviation_ratio: float = 0.02
    hard_edge_straight_max_turn_deg: float = 12.0


class SegmentConfig(BaseModel):
    enable: bool = True
    window_points: int = 9
    corner_split_angle_deg: float = 45.0
    corner_resample_step_px: float = 2.0
    corner_scales_px: list[float] = Field(default_factory=lambda: [6.0, 12.0, 24.0])
    corner_multiscale_ratio: float = 0.65
    corner_min_scales_hit: int = 2
    corner_region_gap_points: int = 1
    fillet_enable: bool = True
    fillet_line_window_points: int = 6
    fillet_line_residual_px: float = 1.0
    fillet_circle_residual_px: float = 1.5
    fillet_min_turn_deg: float = 35.0
    fillet_max_turn_deg: float = 170.0
    fillet_radius_min_px: float = 2.0
    fillet_radius_max_px: float = 2000.0
    fillet_max_span_px: float = 80.0
    break_merge_distance_px: float = 2.0
    straight_min_chord_px: float = 12.0
    straight_max_deviation_px: float = 2.0
    straight_max_deviation_ratio: float = 0.02
    straight_max_turn_deg: float = 12.0
    min_segment_points: int = 3
    min_length_px: float = 0.0
    forced_break_tolerance_px: float = 3.0
    endpoint_snap_tolerance_px: float = 2.5


class FitConfig(BaseModel):
    geometry_mode: str = "mixed"  # "mixed" | "polyline_only" | "nurbs_only"
    degree: int = 3
    max_control_points: int = 50
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    stabilize: StabilizeConfig = Field(default_factory=StabilizeConfig)
    simplify: SimplifyConfig = Field(default_factory=SimplifyConfig)
    spline: SplineConfig = Field(default_factory=SplineConfig)

    @field_validator("geometry_mode")
    @classmethod
    def _validate_geometry_mode(cls, value: str) -> str:
        mode = str(value).strip().lower()
        if mode not in {"mixed", "polyline_only", "nurbs_only"}:
            raise ValueError("fit.geometry_mode must be one of: mixed, polyline_only, nurbs_only")
        return mode


class ExportConfig(BaseModel):
    plane: str = "XY"
    layer_name: str = "sketch_curve"
    object_name: str = "main_nurbs"
    multi_object_prefix: str = "stroke"


class MultiOutputConfig(BaseModel):
    min_length_px: float = 20.0
    max_curves: int = 50
    include_loops: bool = True
    sort_by: str = "length"
    preserve_junctions: bool = True
    junction_snap_radius_px: float = 4.0


class OutputConfig(BaseModel):
    mode: str = "multi"  # "single" | "multi"
    multi: MultiOutputConfig = Field(default_factory=MultiOutputConfig)


class AppConfig(BaseModel):
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    path_extract: PathExtractConfig = Field(default_factory=PathExtractConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
