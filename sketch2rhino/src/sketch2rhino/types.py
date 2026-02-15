from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

BinaryImage: TypeAlias = npt.NDArray[np.uint8]
SkeletonImage: TypeAlias = npt.NDArray[np.uint8]
Point2D: TypeAlias = tuple[float, float]


@dataclass(slots=True)
class GraphEdge:
    start: int
    end: int
    pixels: list[tuple[int, int]]
    length_px: float


@dataclass(slots=True)
class StrokeGraph:
    node_points: dict[int, tuple[int, int]]
    adjacency: dict[int, set[int]]
    edges: list[GraphEdge]
    endpoints: set[int]
    component_size_px: int = 0


@dataclass(slots=True)
class Polyline2D:
    points: list[Point2D]

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.points, dtype=np.float64)


@dataclass(slots=True)
class NurbsSpec:
    degree: int
    control_points: list[Point2D]
    knots: list[float]
    weights: list[float] | None = None


@dataclass(slots=True)
class ExportResult:
    output_path: Path
    report: dict[str, Any] = field(default_factory=dict)
