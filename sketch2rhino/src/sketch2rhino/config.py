from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DenoiseConfig(BaseModel):
    enable: bool = False
    method: str = "gaussian"
    strength: float = 1.0


class BinarizeConfig(BaseModel):
    method: str = "otsu"
    block_size: int = 31
    C: int = 7


class MorphConfig(BaseModel):
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


class SimplifyConfig(BaseModel):
    method: str = "rdp"
    epsilon_px: float = 1.5
    smooth_enable: bool = True
    smooth_window: int = 5
    smooth_passes: int = 1


class SplineConfig(BaseModel):
    smoothing: float = 0.0
    max_iter: int = 5
    fallback_if_fail: bool = True


class FitConfig(BaseModel):
    degree: int = 3
    max_control_points: int = 50
    simplify: SimplifyConfig = Field(default_factory=SimplifyConfig)
    spline: SplineConfig = Field(default_factory=SplineConfig)


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
