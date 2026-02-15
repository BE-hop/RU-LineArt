from __future__ import annotations

from pathlib import Path

import rhino3dm

from sketch2rhino.config import ExportConfig
from sketch2rhino.types import ExportResult, NurbsSpec


def _uniform_knots_for_rhino(n_ctrl: int, degree: int) -> list[float]:
    # Rhino knot list omits one duplicated start/end knot compared with full clamped vectors.
    n_knots = n_ctrl + degree - 1
    if n_knots <= 0:
        return []
    if n_knots == 1:
        return [0.0]

    knots = [0.0] * n_knots
    for i in range(n_knots):
        if i < degree:
            knots[i] = 0.0
        elif i >= n_knots - degree:
            knots[i] = 1.0
        else:
            knots[i] = (i - degree + 1) / (n_knots - 2 * degree + 1)
    return knots


def _to_rhino_knots(spec: NurbsSpec) -> list[float]:
    n_ctrl = len(spec.control_points)
    target = n_ctrl + spec.degree - 1
    if target <= 0:
        return []

    if len(spec.knots) == target:
        return list(spec.knots)

    # SciPy full knot vectors are usually n_ctrl + degree + 1.
    if len(spec.knots) == n_ctrl + spec.degree + 1:
        trimmed = spec.knots[1:-1]
        if len(trimmed) == target:
            return [float(v) for v in trimmed]

    return _uniform_knots_for_rhino(n_ctrl, spec.degree)


def _build_nurbs_curve(spec: NurbsSpec) -> rhino3dm.NurbsCurve:
    degree = int(spec.degree)
    order = degree + 1
    n_ctrl = len(spec.control_points)

    if n_ctrl < order:
        raise ValueError("Not enough control points for requested degree")

    rational = spec.weights is not None and len(spec.weights) == n_ctrl
    curve = rhino3dm.NurbsCurve(3, rational, order, n_ctrl)

    for i, (x, y) in enumerate(spec.control_points):
        if rational:
            w = float(spec.weights[i])
            curve.Points[i] = rhino3dm.Point4d(float(x), float(y), 0.0, w)
        else:
            curve.Points[i] = rhino3dm.Point4d(float(x), float(y), 0.0, 1.0)

    rhino_knots = _to_rhino_knots(spec)
    if len(rhino_knots) != len(curve.Knots):
        rhino_knots = _uniform_knots_for_rhino(n_ctrl, degree)

    for i, kv in enumerate(rhino_knots):
        curve.Knots[i] = float(kv)

    return curve


def _ensure_layer(model: rhino3dm.File3dm, name: str) -> int:
    for i, layer in enumerate(model.Layers):
        if layer.Name == name:
            return i

    layer = rhino3dm.Layer()
    layer.Name = name
    return model.Layers.Add(layer)


def write_3dm(spec: NurbsSpec, output_path: str | Path, cfg: ExportConfig) -> ExportResult:
    return write_3dm_many([spec], output_path, cfg)


def write_3dm_many(specs: list[NurbsSpec], output_path: str | Path, cfg: ExportConfig) -> ExportResult:
    if not specs:
        raise ValueError("No NURBS curves to export")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = rhino3dm.File3dm()
    layer_idx = _ensure_layer(model, cfg.layer_name)

    for i, spec in enumerate(specs, start=1):
        curve = _build_nurbs_curve(spec)

        attr = rhino3dm.ObjectAttributes()
        if len(specs) == 1:
            attr.Name = cfg.object_name
        else:
            attr.Name = f"{cfg.multi_object_prefix}_{i:03d}"
        attr.LayerIndex = layer_idx
        model.Objects.AddCurve(curve, attr)

    ok = model.Write(str(out), 8)
    if not ok:
        raise RuntimeError(f"Failed to write 3dm file: {out}")

    return ExportResult(
        output_path=out,
        report={
            "curve_count": len(specs),
            "control_points": [len(spec.control_points) for spec in specs],
            "degree": [spec.degree for spec in specs],
        },
    )
