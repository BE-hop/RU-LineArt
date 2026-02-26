from __future__ import annotations

from pathlib import Path

import rhino3dm

from sketch2rhino import __version__
from sketch2rhino.brand import brand_signature, generated_by_text, generator_label
from sketch2rhino.config import ExportConfig
from sketch2rhino.types import CurveGeometry, ExportResult, NurbsSpec, Polyline2D


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


def _dedupe_polyline_points(points: list[tuple[float, float]], tol: float = 1e-6) -> list[tuple[float, float]]:
    if not points:
        return []
    out = [points[0]]
    for x, y in points[1:]:
        px, py = out[-1]
        if abs(float(x) - float(px)) <= tol and abs(float(y) - float(py)) <= tol:
            continue
        out.append((float(x), float(y)))
    return out


def _build_polyline_curve(spec: Polyline2D) -> rhino3dm.Curve:
    points = _dedupe_polyline_points(spec.points)
    if len(points) < 2:
        raise ValueError("Polyline geometry requires at least 2 points")

    if len(points) == 2:
        p0 = rhino3dm.Point3d(float(points[0][0]), float(points[0][1]), 0.0)
        p1 = rhino3dm.Point3d(float(points[1][0]), float(points[1][1]), 0.0)
        return rhino3dm.LineCurve(p0, p1)

    pts3 = [rhino3dm.Point3d(float(x), float(y), 0.0) for x, y in points]
    return rhino3dm.PolylineCurve(pts3)


def _build_curve(spec: CurveGeometry) -> tuple[rhino3dm.Curve, str, int]:
    if isinstance(spec, NurbsSpec):
        curve = _build_nurbs_curve(spec)
        return curve, "nurbs", len(spec.control_points)
    if isinstance(spec, Polyline2D):
        curve = _build_polyline_curve(spec)
        return curve, "polyline", len(spec.points)
    raise TypeError(f"Unsupported curve geometry type: {type(spec)!r}")


def _ensure_layer(model: rhino3dm.File3dm, name: str) -> int:
    for i, layer in enumerate(model.Layers):
        if layer.Name == name:
            return i

    layer = rhino3dm.Layer()
    layer.Name = name
    return model.Layers.Add(layer)


def write_3dm(
    spec: CurveGeometry,
    output_path: str | Path,
    cfg: ExportConfig,
    node_ids: tuple[int, int] | None = None,
) -> ExportResult:
    ids = [node_ids] if node_ids is not None else None
    return write_3dm_many([spec], output_path, cfg, node_ids=ids)


def write_3dm_many(
    specs: list[CurveGeometry],
    output_path: str | Path,
    cfg: ExportConfig,
    node_ids: list[tuple[int, int]] | None = None,
) -> ExportResult:
    if not specs:
        raise ValueError("No curve geometry to export")
    if node_ids is not None and len(node_ids) != len(specs):
        raise ValueError("node_ids length must match number of curves")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = rhino3dm.File3dm()
    layer_idx = _ensure_layer(model, cfg.layer_name)
    signature = brand_signature()
    model.Strings["generated_by"] = generated_by_text()
    model.Strings["generator"] = generator_label()
    model.Strings["behop.version"] = __version__
    for key, value in signature.items():
        model.Strings[f"behop.{key}"] = value

    geometry_types: list[str] = []
    point_counts: list[int] = []
    degree: list[int | None] = []
    for i, spec in enumerate(specs, start=1):
        curve, geometry_type, points_count = _build_curve(spec)
        geometry_types.append(geometry_type)
        point_counts.append(points_count)
        degree.append(spec.degree if isinstance(spec, NurbsSpec) else None)

        attr = rhino3dm.ObjectAttributes()
        if len(specs) == 1:
            attr.Name = cfg.object_name
        else:
            attr.Name = f"{cfg.multi_object_prefix}_{i:03d}"
        attr.LayerIndex = layer_idx
        attr.SetUserString("geometry_type", geometry_type)
        if node_ids is not None:
            start_node, end_node = node_ids[i - 1]
            attr.SetUserString("start_node_id", str(int(start_node)))
            attr.SetUserString("end_node_id", str(int(end_node)))
        attr.SetUserString("generated_by", generated_by_text())
        attr.SetUserString("generator", generator_label())
        attr.SetUserString("behop.version", __version__)
        for key, value in signature.items():
            attr.SetUserString(f"behop.{key}", value)
        model.Objects.AddCurve(curve, attr)

    ok = model.Write(str(out), 8)
    if not ok:
        raise RuntimeError(f"Failed to write 3dm file: {out}")

    return ExportResult(
        output_path=out,
        report={
            "curve_count": len(specs),
            "geometry_types": geometry_types,
            "points_count": point_counts,
            "degree": degree,
            "node_ids": node_ids if node_ids is not None else [],
            "generated_by": generated_by_text(),
            "generator": generator_label(),
            "brand": signature,
        },
    )
