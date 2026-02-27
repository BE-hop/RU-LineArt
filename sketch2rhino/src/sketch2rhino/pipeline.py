from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.interpolate import BSpline

from sketch2rhino.brand import brand_signature
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
from sketch2rhino.fit.nurbs_fit import fit_open_nurbs, fit_open_polyline, should_use_polyline_geometry
from sketch2rhino.fit.polyline_simplify import simplify_polyline
from sketch2rhino.fit.segment_split import anchors_for_segment, snap_segment_endpoints, split_polyline_into_segments
from sketch2rhino.fit.stroke_stabilize import stabilize_polyline
from sketch2rhino.topo.graph_build import build_stroke_graph
from sketch2rhino.topo.path_extract import (
    choose_main_component,
    extract_junction_centers,
    extract_main_open_path,
    extract_open_paths,
    split_components,
)
from sketch2rhino.types import CurveGeometry, ExportResult, NurbsSpec, Polyline2D
from sketch2rhino.vision.preprocess import load_grayscale_image, preprocess_image
from sketch2rhino.vision.skeletonize import skeletonize_image

def _polyline_length(polyline: Polyline2D) -> float:
    arr = polyline.as_array()
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _geometry_point_count(geometry: CurveGeometry) -> int:
    if isinstance(geometry, NurbsSpec):
        return len(geometry.control_points)
    return len(geometry.points)


def _polyline_has_2_unique_points(polyline: Polyline2D, tol: float = 1e-6) -> bool:
    arr = polyline.as_array()
    if len(arr) < 2:
        return False
    d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return bool(np.any(d > float(tol)))


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


def _fit_geometry_for_segment(
    polyline: Polyline2D,
    anchors: list[tuple[float, float]],
    cfg: AppConfig,
) -> tuple[CurveGeometry, str]:
    mode = str(cfg.fit.geometry_mode).strip().lower()
    if mode == "polyline_only":
        return fit_open_polyline(polyline, cfg.fit), "polyline"
    if mode == "nurbs_only":
        return fit_open_nurbs(polyline, cfg.fit, anchors=anchors), "nurbs"
    if should_use_polyline_geometry(polyline, cfg.fit):
        return fit_open_polyline(polyline, cfg.fit), "polyline"
    return fit_open_nurbs(polyline, cfg.fit, anchors=anchors), "nurbs"


def _filter_multi_output_polylines(
    candidates: list[Polyline2D],
    min_length_px: float,
    max_curves: int,
    sort_by: str,
) -> tuple[list[Polyline2D], dict[str, object], list[str]]:
    raw_count = len(candidates)
    min_len = max(0.0, float(min_length_px))
    filtered = [pl for pl in candidates if _polyline_length(pl) >= min_len]
    dropped_by_min = raw_count - len(filtered)

    if str(sort_by).strip().lower() == "length":
        filtered.sort(key=_polyline_length, reverse=True)

    before_truncate = len(filtered)
    max_keep = int(max_curves)
    if max_keep > 0:
        filtered = filtered[:max_keep]
    dropped_by_max = before_truncate - len(filtered)

    stats: dict[str, object] = {
        "raw_candidate_count": int(raw_count),
        "min_length_px": float(min_len),
        "kept_after_min_length": int(before_truncate),
        "dropped_by_min_length": int(dropped_by_min),
        "max_curves": int(max_keep),
        "kept_final_count": int(len(filtered)),
        "dropped_by_max_curves": int(dropped_by_max),
    }

    warnings: list[str] = []
    if dropped_by_min > 0:
        warnings.append(
            "polylines filtered by output.multi.min_length_px: "
            f"{raw_count} -> {before_truncate}"
        )
    if dropped_by_max > 0:
        warnings.append(
            "polylines truncated by output.multi.max_curves: "
            f"{before_truncate} -> {len(filtered)}"
        )

    return filtered, stats, warnings


def _polyline_pixel_coverage(
    polylines: list[Polyline2D],
    reference_mask: np.ndarray,
) -> dict[str, object]:
    ref = reference_mask.astype(bool)
    total = int(np.count_nonzero(ref))
    if total <= 0:
        return {"covered_px": 0, "total_px": 0, "ratio": 0.0}

    canvas = np.zeros(reference_mask.shape, dtype=np.uint8)
    for pl in polylines:
        pts: list[tuple[int, int]] = []
        for x, y in pl.points:
            col = int(round(x))
            row = int(round(-y))
            if 0 <= row < canvas.shape[0] and 0 <= col < canvas.shape[1]:
                pts.append((col, row))
        if len(pts) >= 2:
            arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [arr], isClosed=False, color=255, thickness=1)
        elif len(pts) == 1:
            cv2.circle(canvas, pts[0], radius=0, color=255, thickness=-1)

    covered = int(np.count_nonzero((canvas > 0) & ref))
    return {
        "covered_px": covered,
        "total_px": total,
        "ratio": float(covered / total),
    }


def run_pipeline(
    image_path: Path,
    output_path: Path,
    cfg: AppConfig,
    debug_dir: Path | None = None,
) -> ExportResult:
    report: dict[str, object] = {"warnings": [], "timings": {}, "brand": brand_signature()}

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
    raw_path_candidates: list[Polyline2D] = []

    if mode == "single":
        polylines.append(extract_main_open_path(components[0], graphs[0], cfg.path_extract))
        raw_path_candidates = list(polylines)
    else:
        for comp, graph in zip(components, graphs, strict=True):
            raw_path_candidates.extend(
                extract_open_paths(
                    skeleton=comp,
                    graph=graph,
                    cfg=cfg.path_extract,
                    min_length_px=0.0,
                    max_curves=None,
                    include_loops=cfg.output.multi.include_loops,
                )
            )
        polylines, path_stats, path_warnings = _filter_multi_output_polylines(
            candidates=raw_path_candidates,
            min_length_px=float(cfg.output.multi.min_length_px),
            max_curves=int(cfg.output.multi.max_curves),
            sort_by=str(cfg.output.multi.sort_by),
        )
        report["path_extract_stats"] = path_stats
        report["warnings"].extend(path_warnings)
        report["path_extract_stats"]["coverage_raw_vs_skeleton"] = _polyline_pixel_coverage(
            raw_path_candidates,
            skeleton,
        )
        report["path_extract_stats"]["coverage_filtered_vs_skeleton"] = _polyline_pixel_coverage(
            polylines,
            skeleton,
        )

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
    segment_polylines: list[Polyline2D] = []
    segment_anchors: list[list[tuple[float, float]]] = []
    segment_sources: list[tuple[int, int]] = []  # (source_curve_idx, local_segment_idx)
    for i, polyline in enumerate(stabilized_list):
        base_anchors = _polyline_anchor_points(
            polyline=polyline,
            protected_points=protected_points[i] if i < len(protected_points) else None,
        )
        segments = split_polyline_into_segments(
            polyline,
            cfg.fit.segment,
            forced_break_points=base_anchors,
        )
        for j, seg in enumerate(segments):
            seg_anchor_candidates = anchors_for_segment(
                seg,
                anchors=base_anchors,
                tolerance_px=float(cfg.fit.segment.forced_break_tolerance_px),
            )
            seg_simplified = simplify_polyline(
                seg,
                cfg.fit.simplify,
                protected_points=seg_anchor_candidates,
            )
            segment_polylines.append(seg_simplified)
            segment_anchors.append(_polyline_anchor_points(seg_simplified, seg_anchor_candidates))
            segment_sources.append((i, j))

    if not segment_polylines:
        raise ValueError("No valid segments after split")

    node_pairs: list[tuple[int, int]] = [(-1, -1) for _ in segment_polylines]
    node_centers: list[tuple[float, float]] = []
    if float(cfg.fit.segment.endpoint_snap_tolerance_px) > 0.0:
        segment_polylines, node_pairs, node_centers = snap_segment_endpoints(
            segment_polylines,
            tolerance_px=float(cfg.fit.segment.endpoint_snap_tolerance_px),
        )
        segment_anchors = [
            _polyline_anchor_points(segment_polylines[i], segment_anchors[i]) for i in range(len(segment_polylines))
        ]

    filtered_polylines: list[Polyline2D] = []
    filtered_anchors: list[list[tuple[float, float]]] = []
    filtered_sources: list[tuple[int, int]] = []
    filtered_node_pairs: list[tuple[int, int]] = []
    min_seg_len = max(0.0, float(cfg.fit.segment.min_length_px))
    dropped_by_min_seg_len = 0
    dropped_degenerate = 0
    for i, seg in enumerate(segment_polylines):
        if not _polyline_has_2_unique_points(seg):
            dropped_degenerate += 1
            continue
        if min_seg_len > 0.0 and _polyline_length(seg) < min_seg_len:
            dropped_by_min_seg_len += 1
            continue
        filtered_polylines.append(seg)
        filtered_anchors.append(segment_anchors[i])
        filtered_sources.append(segment_sources[i])
        filtered_node_pairs.append(node_pairs[i] if i < len(node_pairs) else (-1, -1))

    segment_polylines = filtered_polylines
    segment_anchors = filtered_anchors
    segment_sources = filtered_sources
    node_pairs = filtered_node_pairs
    report["segment_filter_stats"] = {
        "min_length_px": float(min_seg_len),
        "dropped_by_min_length": int(dropped_by_min_seg_len),
        "dropped_degenerate": int(dropped_degenerate),
    }
    if dropped_by_min_seg_len > 0:
        report["warnings"].append(
            "segments filtered by fit.segment.min_length_px: "
            f"-{dropped_by_min_seg_len}"
        )
    if not segment_polylines:
        raise ValueError("No valid segments after endpoint snap")

    geometry_list: list[CurveGeometry] = []
    geometry_types: list[str] = []
    for i, polyline in enumerate(segment_polylines):
        geometry, geometry_type = _fit_geometry_for_segment(polyline, segment_anchors[i], cfg)
        geometry_list.append(geometry)
        geometry_types.append(geometry_type)
    report["timings"]["fit"] = perf_counter() - t4

    node_ids_for_export: list[tuple[int, int]] | None = None
    if node_pairs and all(a >= 0 and b >= 0 for a, b in node_pairs):
        node_ids_for_export = node_pairs

    t5 = perf_counter()
    if len(geometry_list) == 1:
        export_result = write_3dm(
            geometry_list[0],
            output_path,
            cfg.export,
            node_ids=node_ids_for_export[0] if node_ids_for_export else None,
        )
    else:
        export_result = write_3dm_many(geometry_list, output_path, cfg.export, node_ids=node_ids_for_export)
    report["timings"]["export"] = perf_counter() - t5
    report["export_info"] = export_result.report

    point_counts = [_geometry_point_count(g) for g in geometry_list]
    if len(segment_polylines) == 1:
        report["polyline_points"] = len(segment_polylines[0].points)
        report["control_points"] = point_counts[0]
        report["geometry_type"] = geometry_types[0]
        report["segment_count"] = 1
    else:
        report["polyline_points"] = [len(pl.points) for pl in segment_polylines]
        report["control_points"] = point_counts
        report["curve_count"] = len(geometry_list)
        report["geometry_types"] = geometry_types
        report["segment_count"] = len(segment_polylines)
    report["nurbs_curve_count"] = int(sum(1 for t in geometry_types if t == "nurbs"))
    report["polyline_curve_count"] = int(sum(1 for t in geometry_types if t == "polyline"))

    segments_report: list[dict[str, object]] = []
    for i in range(len(segment_polylines)):
        start_node, end_node = node_pairs[i] if i < len(node_pairs) else (-1, -1)
        source_idx, local_idx = segment_sources[i]
        segments_report.append(
            {
                "segment_idx": i,
                "source_curve_idx": int(source_idx),
                "local_segment_idx": int(local_idx),
                "geometry_type": geometry_types[i],
                "polyline_points": len(segment_polylines[i].points),
                "curve_points": point_counts[i],
                "start_node_id": int(start_node),
                "end_node_id": int(end_node),
            }
        )
    report["segments"] = segments_report

    if node_centers:
        node_degree: dict[int, int] = {}
        for s_id, e_id in node_pairs:
            if s_id >= 0:
                node_degree[s_id] = node_degree.get(s_id, 0) + 1
            if e_id >= 0:
                node_degree[e_id] = node_degree.get(e_id, 0) + 1
        report["nodes"] = [
            {"id": i, "x": float(x), "y": float(y), "degree": int(node_degree.get(i, 0))}
            for i, (x, y) in enumerate(node_centers)
        ]

    anchor_stats: list[dict[str, object]] = []
    for i, geom in enumerate(geometry_list):
        if isinstance(geom, NurbsSpec):
            stat = _anchor_error_stats(
                spec=geom,
                anchors=segment_anchors[i],
                sample_count=int(cfg.fit.spline.anchor_sample_count),
            )
            stat["geometry_type"] = "nurbs"
            anchor_stats.append(stat)
        else:
            anchor_stats.append(
                {
                    "geometry_type": "polyline",
                    "count": len(segment_anchors[i]),
                    "errors_px": [],
                    "max_px": 0.0,
                    "mean_px": 0.0,
                }
            )
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
        if mode == "multi" and raw_path_candidates:
            # Candidate paths before min_length/max_curves filtering.
            save_path_overlay(
                gray,
                raw_path_candidates if len(raw_path_candidates) > 1 else raw_path_candidates[0],
                dbg / "03_path_overlay_raw.png",
            )
        # Raw extracted paths (before simplify/split/fitting).
        save_path_overlay(gray, polylines if len(polylines) > 1 else polylines[0], dbg / "03_path_overlay.png")
        # Segment-level paths after split + per-segment simplify + endpoint snap.
        save_path_overlay(
            gray,
            segment_polylines if len(segment_polylines) > 1 else segment_polylines[0],
            dbg / "03_segment_overlay.png",
        )
        save_polyline_json(
            segment_polylines if len(segment_polylines) > 1 else segment_polylines[0],
            dbg / "04_polyline.json",
        )
        save_nurbs_json(
            geometry_list if len(geometry_list) > 1 else geometry_list[0],
            dbg / "05_nurbs.json",
        )
        save_report_json(report, dbg / "report.json")

    return ExportResult(output_path=export_result.output_path, report=report)
