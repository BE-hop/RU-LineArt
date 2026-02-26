from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from sketch2rhino.types import BinaryImage, CurveGeometry, NurbsSpec, Polyline2D, SkeletonImage


def ensure_debug_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_binary_png(binary: BinaryImage, out_path: str | Path) -> None:
    img = (binary > 0).astype("uint8") * 255
    cv2.imwrite(str(out_path), img)


def save_skeleton_png(skeleton: SkeletonImage, out_path: str | Path) -> None:
    img = (skeleton > 0).astype("uint8") * 255
    cv2.imwrite(str(out_path), img)


def save_polyline_json(polyline: Polyline2D | list[Polyline2D], out_path: str | Path) -> None:
    if isinstance(polyline, list):
        data = {
            "polylines": [{"points": pl.points, "count": len(pl.points)} for pl in polyline],
            "count": len(polyline),
        }
    else:
        data = {"points": polyline.points, "count": len(polyline.points)}
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _curve_payload(spec: CurveGeometry) -> dict[str, Any]:
    if isinstance(spec, NurbsSpec):
        return {
            "geometry_type": "nurbs",
            "degree": spec.degree,
            "control_points": spec.control_points,
            "control_points_count": len(spec.control_points),
            "knots": spec.knots,
            "weights": spec.weights,
        }
    return {
        "geometry_type": "polyline",
        "points": spec.points,
        "points_count": len(spec.points),
    }


def save_nurbs_json(spec: CurveGeometry | list[CurveGeometry], out_path: str | Path) -> None:
    if isinstance(spec, list):
        data: dict[str, Any] = {
            "curves": [_curve_payload(s) for s in spec],
            "count": len(spec),
        }
    else:
        data = _curve_payload(spec)
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_report_json(report: dict[str, Any], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
