from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline

from sketch2rhino.config import AppConfig
from sketch2rhino.debug.artifacts import (
    ensure_debug_dir,
    save_binary_png,
    save_nurbs_json,
    save_polyline_json,
    save_report_json,
    save_skeleton_png,
)
from sketch2rhino.debug.overlays import save_path_overlay
from sketch2rhino.export.rhino3dm_writer import write_3dm, write_3dm_many
from sketch2rhino.fit.nurbs_fit import fit_open_nurbs
from sketch2rhino.fit.polyline_simplify import simplify_polyline
from sketch2rhino.fit.stroke_stabilize import stabilize_polyline
from sketch2rhino.topo.graph_build import build_stroke_graph
from sketch2rhino.topo.path_extract import (
    choose_main_component,
    extract_junction_centers,
    extract_main_open_path,
    extract_open_paths,
    split_components,
)
from sketch2rhino.types import ExportResult, NurbsSpec, Polyline2D
from sketch2rhino.vision.preprocess import load_grayscale_image, preprocess_image
from sketch2rhino.vision.skeletonize import skeletonize_image

def _polyline_length(polyline: Polyline2D) -> float:
    arr = polyline.as_array()
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _snap_and_collect_protected_points(
    polylines: list[Polyline2D],
    junctions_xy: list[tuple[float, float]],
    snap_radius: float,
) -> tuple[list[Polyline2D], list[list[tuple[float, float]]]]:
    if not polylines:
        return polylines, []

    arrays = [pl.as_array().copy() for pl in polylines]
    protected: list[list[tuple[float, float]]] = [[] for _ in polylines]
    radius = max(0.5, float(snap_radius))

    for jx, jy in junctions_xy:
        targets: list[tuple[int, int]] = []  # (polyline_idx, point_idx)
        for i, arr in enumerate(arrays):
            if len(arr) == 0:
                continue
            d = np.linalg.norm(arr - np.array([jx, jy], dtype=np.float64), axis=1)
            idx = int(np.argmin(d))
            if float(d[idx]) <= radius:
                targets.append((i, idx))

        if len(targets) < 2:
            continue

        for i, idx in targets:
            arrays[i][idx] = np.array([jx, jy], dtype=np.float64)
            protected[i].append((jx, jy))

    snapped = [Polyline2D(points=[(float(x), float(y)) for x, y in arr]) for arr in arrays]
    return snapped, protected


def _polyline_anchor_points(
    polyline: Polyline2D,
    protected_points: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    anchors: list[tuple[float, float]] = []
    if polyline.points:
        anchors.append(polyline.points[0])
        anchors.append(polyline.points[-1])
    if protected_points:
        anchors.extend(protected_points)

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in anchors:
        key = (round(float(x), 4), round(float(y), 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((float(x), float(y)))

    # Merge near-duplicate anchors to avoid over-constraining one local junction area.
    merged: list[tuple[float, float]] = []
    merge_radius = 2.5
    for ax, ay in deduped:
        found = False
        for i, (mx, my) in enumerate(merged):
            if np.hypot(ax - mx, ay - my) <= merge_radius:
                merged[i] = ((mx + ax) * 0.5, (my + ay) * 0.5)
                found = True
                break
        if not found:
            merged.append((ax, ay))
    return merged


def _sample_nurbs_curve(spec: NurbsSpec, n_samples: int) -> np.ndarray:
    n = max(64, int(n_samples))
    knots = np.asarray(spec.knots, dtype=np.float64)
    cps = np.asarray(spec.control_points, dtype=np.float64)
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)

    bx = BSpline(knots, cps[:, 0], int(spec.degree))
    by = BSpline(knots, cps[:, 1], int(spec.degree))
    return np.column_stack([bx(u), by(u)])


def _anchor_error_stats(
    spec: NurbsSpec,
    anchors: list[tuple[float, float]],
    sample_count: int,
) -> dict[str, object]:
    if not anchors:
        return {"count": 0, "errors_px": [], "max_px": 0.0, "mean_px": 0.0}

    curve = _sample_nurbs_curve(spec, sample_count)
    errors: list[float] = []
    for ax, ay in anchors:
        d = np.linalg.norm(curve - np.array([ax, ay], dtype=np.float64), axis=1)
        errors.append(float(d.min(initial=float("inf"))))

    return {
        "count": len(errors),
        "errors_px": errors,
        "max_px": float(max(errors) if errors else 0.0),
        "mean_px": float(np.mean(errors) if errors else 0.0),
    }


def run_pipeline(
    image_path: Path,
    output_path: Path,
    cfg: AppConfig,
    debug_dir: Path | None = None,
) -> ExportResult:
    report: dict[str, object] = {"warnings": [], "timings": {}}

    t0 = perf_counter()
    gray = load_grayscale_image(image_path)
    report["timings"]["load_image"] = perf_counter() - t0

    t1 = perf_counter()
    binary = preprocess_image(gray, cfg.preprocess)
    report["timings"]["preprocess"] = perf_counter() - t1

    t2 = perf_counter()
    skeleton = skeletonize_image(binary, cfg.skeleton)
    report["timings"]["skeletonize"] = perf_counter() - t2

    t3 = perf_counter()
    mode = cfg.output.mode.lower()
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unsupported output mode: {cfg.output.mode}")

    if mode == "single":
        components = [choose_main_component(skeleton, cfg.path_extract.choose_component)]
    else:
        if cfg.path_extract.choose_component == "largest":
            components = [choose_main_component(skeleton, "largest")]
        else:
            components = split_components(skeleton)
            if not components:
                components = [skeleton.copy()]

    graphs = [build_stroke_graph(comp) for comp in components]
    polylines: list[Polyline2D] = []

    if mode == "single":
        polylines.append(extract_main_open_path(components[0], graphs[0], cfg.path_extract))
    else:
        for comp, graph in zip(components, graphs, strict=True):
            polylines.extend(
                extract_open_paths(
                    skeleton=comp,
                    graph=graph,
                    cfg=cfg.path_extract,
                    min_length_px=cfg.output.multi.min_length_px,
                    max_curves=None,
                    include_loops=cfg.output.multi.include_loops,
                )
            )
        if cfg.output.multi.sort_by == "length":
            polylines.sort(key=_polyline_length, reverse=True)
        if cfg.output.multi.max_curves > 0:
            polylines = polylines[: cfg.output.multi.max_curves]

    if not polylines:
        raise ValueError("No valid path extracted from skeleton")

    protected_points: list[list[tuple[float, float]]] = [[] for _ in polylines]
    junctions_xy: list[tuple[float, float]] = []
    if mode == "multi" and cfg.output.multi.preserve_junctions and len(polylines) >= 2:
        for graph in graphs:
            junctions_xy.extend(extract_junction_centers(graph, cfg.path_extract))
        polylines, protected_points = _snap_and_collect_protected_points(
            polylines=polylines,
            junctions_xy=junctions_xy,
            snap_radius=cfg.output.multi.junction_snap_radius_px,
        )

    report["timings"]["path_extract"] = perf_counter() - t3

    t4 = perf_counter()
    stabilized_list = [
        stabilize_polyline(
            polyline,
            cfg.fit.stabilize,
            protected_points=protected_points[i] if i < len(protected_points) else None,
        )
        for i, polyline in enumerate(polylines)
    ]
    simplified_list = [
        simplify_polyline(
            polyline,
            cfg.fit.simplify,
            protected_points=protected_points[i] if i < len(protected_points) else None,
        )
        for i, polyline in enumerate(stabilized_list)
    ]
    anchor_lists = [
        _polyline_anchor_points(
            polyline=simplified_list[i],
            protected_points=protected_points[i] if i < len(protected_points) else None,
        )
        for i in range(len(simplified_list))
    ]
    nurbs_list: list[NurbsSpec] = [
        fit_open_nurbs(polyline, cfg.fit, anchors=anchor_lists[i]) for i, polyline in enumerate(simplified_list)
    ]
    report["timings"]["fit"] = perf_counter() - t4

    t5 = perf_counter()
    if len(nurbs_list) == 1:
        export_result = write_3dm(nurbs_list[0], output_path, cfg.export)
    else:
        export_result = write_3dm_many(nurbs_list, output_path, cfg.export)
    report["timings"]["export"] = perf_counter() - t5

    if len(simplified_list) == 1:
        report["polyline_points"] = len(simplified_list[0].points)
        report["control_points"] = len(nurbs_list[0].control_points)
    else:
        report["polyline_points"] = [len(pl.points) for pl in simplified_list]
        report["control_points"] = [len(ns.control_points) for ns in nurbs_list]
        report["curve_count"] = len(nurbs_list)

    anchor_stats = [
        _anchor_error_stats(
            spec=nurbs_list[i],
            anchors=anchor_lists[i],
            sample_count=int(cfg.fit.spline.anchor_sample_count),
        )
        for i in range(len(nurbs_list))
    ]
    report["anchor_error_px"] = {
        "per_curve": anchor_stats,
        "max_px": float(max((s["max_px"] for s in anchor_stats), default=0.0)),
        "mean_px": float(np.mean([s["mean_px"] for s in anchor_stats]) if anchor_stats else 0.0),
    }

    report["graph_nodes"] = sum(len(g.node_points) for g in graphs)
    report["graph_edges"] = sum(len(g.edges) for g in graphs)
    report["component_size_px"] = sum(g.component_size_px for g in graphs)
    report["components"] = len(components)
    report["junctions_detected"] = len(junctions_xy)

    if debug_dir is not None:
        dbg = ensure_debug_dir(debug_dir)
        save_binary_png(binary, dbg / "01_binarized.png")
        save_skeleton_png(skeleton, dbg / "02_skeleton.png")
        save_path_overlay(gray, polylines if len(polylines) > 1 else polylines[0], dbg / "03_path_overlay.png")
        save_polyline_json(simplified_list if len(simplified_list) > 1 else simplified_list[0], dbg / "04_polyline.json")
        save_nurbs_json(nurbs_list if len(nurbs_list) > 1 else nurbs_list[0], dbg / "05_nurbs.json")
        save_report_json(report, dbg / "report.json")

    return ExportResult(output_path=export_result.output_path, report=report)
