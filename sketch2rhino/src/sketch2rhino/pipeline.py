from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

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
from sketch2rhino.fit.segment_split import (
    anchors_for_segment,
    join_collinear_segments,
    snap_segment_endpoints,
    split_polyline_into_segments,
)
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


def _moving_average_polyline(polyline: Polyline2D, window: int, passes: int) -> Polyline2D:
    arr = polyline.as_array()
    if len(arr) < 3:
        return polyline

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    p = max(1, int(passes))
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / float(w)

    out = arr.copy()
    for _ in range(p):
        xpad = np.pad(out[:, 0], (pad, pad), mode="edge")
        ypad = np.pad(out[:, 1], (pad, pad), mode="edge")
        out[:, 0] = np.convolve(xpad, kernel, mode="valid")
        out[:, 1] = np.convolve(ypad, kernel, mode="valid")
        out[0] = arr[0]
        out[-1] = arr[-1]

    return Polyline2D(points=[(float(x), float(y)) for x, y in out])


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


def _resolve_effective_segment_enable(cfg: AppConfig) -> tuple[bool, str]:
    mode = str(cfg.fit.geometry_mode).strip().lower()
    if mode == "nurbs_only":
        return False, "forced_off_by_geometry_mode_nurbs_only"
    if mode == "polyline_only":
        return True, "forced_on_by_geometry_mode_polyline_only"
    return bool(cfg.fit.segment.enable), "from_fit.segment.enable"


def _resolve_effective_path_filter(cfg: AppConfig) -> tuple[float, int, str]:
    mode = str(cfg.fit.geometry_mode).strip().lower()
    if mode == "polyline_only":
        return 0.0, 0, "disabled_for_geometry_mode_polyline_only"
    if mode == "nurbs_only":
        # Keep curve mode complete: avoid hard truncation and only trim tiny speckles.
        return min(float(cfg.output.multi.min_length_px), 4.0), 0, "relaxed_for_geometry_mode_nurbs_only"
    if mode == "mixed":
        # Mixed mode in multi-output should keep full recall; downstream segment filtering handles cleanup.
        return 0.0, 0, "disabled_for_geometry_mode_mixed"
    return float(cfg.output.multi.min_length_px), int(cfg.output.multi.max_curves), "from_output.multi"


def _resolve_effective_component_choice(cfg: AppConfig) -> tuple[str, str]:
    mode = str(cfg.output.mode).strip().lower()
    requested = str(cfg.path_extract.choose_component).strip().lower()
    geometry_mode = str(cfg.fit.geometry_mode).strip().lower()

    if mode == "multi" and geometry_mode == "mixed":
        # Mixed mode targets visual completeness; keep all disconnected components.
        return "all", "forced_all_for_geometry_mode_mixed"
    if requested in {"largest", "all"}:
        return requested, "from_path_extract.choose_component"
    return "largest", "fallback_to_largest_for_invalid_path_extract.choose_component"


def _enhance_binary_for_geometry_mode(binary: np.ndarray, geometry_mode: str) -> np.ndarray:
    mode = str(geometry_mode).strip().lower()
    if mode not in {"mixed", "nurbs_only", "polyline_only"}:
        return binary

    mask_u8 = (binary.astype(np.uint8) > 0).astype(np.uint8) * 255

    if mode == "polyline_only":
        # Straight mode: suppress speckles/noisy burrs while keeping long line continuity.
        blurred = cv2.GaussianBlur(mask_u8, (3, 3), sigmaX=0.7)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return (closed > 0).astype(np.uint8)

    if mode == "nurbs_only":
        # Curve mode: favor continuous, cleaner strokes before skeletonization.
        blurred = cv2.GaussianBlur(mask_u8, (3, 3), sigmaX=1.0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(thresh, 3)
        kernel = np.ones((3, 3), dtype=np.uint8)
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
        return (closed > 0).astype(np.uint8)

    # Mixed mode: light smoothing to reduce skeleton jaggies while keeping topology.
    blurred = cv2.GaussianBlur(mask_u8, (3, 3), sigmaX=0.8)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (closed > 0).astype(np.uint8)


def _points_to_pixel_path(points_xy: list[tuple[float, float]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    pts: list[tuple[int, int]] = []
    for x, y in points_xy:
        col = int(round(float(x)))
        row = int(round(-float(y)))
        if 0 <= row < h and 0 <= col < w:
            pts.append((col, row))
    if not pts:
        return []

    deduped: list[tuple[int, int]] = [pts[0]]
    for p in pts[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    return deduped


def _rasterize_geometries_mask(
    geometries: list[CurveGeometry],
    shape: tuple[int, int],
) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for geom in geometries:
        if isinstance(geom, NurbsSpec):
            sample_n = max(64, len(geom.control_points) * 12)
            arr = _sample_nurbs_curve(geom, sample_n)
            points_xy = [(float(x), float(y)) for x, y in arr]
        else:
            points_xy = [(float(x), float(y)) for x, y in geom.points]

        pixel_path = _points_to_pixel_path(points_xy, shape)
        if len(pixel_path) >= 2:
            arr_i = np.asarray(pixel_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [arr_i], isClosed=False, color=255, thickness=1)
        elif len(pixel_path) == 1:
            cv2.circle(canvas, pixel_path[0], radius=0, color=255, thickness=-1)
    return canvas


def _geometry_alignment_metrics(
    geometries: list[CurveGeometry],
    skeleton: np.ndarray,
) -> dict[str, object]:
    skel = (skeleton > 0).astype(np.uint8)
    curve = (_rasterize_geometries_mask(geometries, skeleton.shape) > 0).astype(np.uint8)

    skel_px = int(np.count_nonzero(skel))
    curve_px = int(np.count_nonzero(curve))
    if skel_px == 0 or curve_px == 0:
        return {
            "skeleton_px": skel_px,
            "curve_px": curve_px,
            "overlap_px": 0,
            "precision_exact": 0.0,
            "recall_exact": 0.0,
            "precision_t1px": 0.0,
            "recall_t1px": 0.0,
            "mean_curve_to_skeleton_px": 0.0,
            "p95_curve_to_skeleton_px": 0.0,
            "max_curve_to_skeleton_px": 0.0,
        }

    overlap = int(np.count_nonzero((skel > 0) & (curve > 0)))
    precision_exact = float(overlap / max(1, curve_px))
    recall_exact = float(overlap / max(1, skel_px))

    dist_to_skeleton = cv2.distanceTransform((1 - skel).astype(np.uint8), cv2.DIST_L2, 3)
    curve_dist = dist_to_skeleton[curve > 0]
    dist_to_curve = cv2.distanceTransform((1 - curve).astype(np.uint8), cv2.DIST_L2, 3)
    skel_dist = dist_to_curve[skel > 0]

    if curve_dist.size == 0:
        curve_dist = np.asarray([0.0], dtype=np.float32)
    if skel_dist.size == 0:
        skel_dist = np.asarray([0.0], dtype=np.float32)

    return {
        "skeleton_px": skel_px,
        "curve_px": curve_px,
        "overlap_px": overlap,
        "precision_exact": precision_exact,
        "recall_exact": recall_exact,
        "precision_t1px": float(np.mean(curve_dist <= 1.0)),
        "recall_t1px": float(np.mean(skel_dist <= 1.0)),
        "mean_curve_to_skeleton_px": float(np.mean(curve_dist)),
        "p95_curve_to_skeleton_px": float(np.percentile(curve_dist, 95.0)),
        "max_curve_to_skeleton_px": float(np.max(curve_dist)),
    }


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
    progress_cb: Callable[[float, str], None] | None = None,
) -> ExportResult:
    def _report_progress(value: float, stage: str) -> None:
        if progress_cb is None:
            return
        try:
            clipped = float(np.clip(float(value), 0.0, 1.0))
            progress_cb(clipped, stage)
        except Exception:
            return

    def _report_loop_progress(base: float, span: float, index: int, total: int, stage: str) -> None:
        if total <= 0:
            _report_progress(base + span, stage)
            return
        step = max(1, int(total // 120))
        cur = int(index) + 1
        if cur >= total or cur % step == 0:
            _report_progress(base + span * (float(cur) / float(total)), stage)

    report: dict[str, object] = {"warnings": [], "timings": {}, "brand": brand_signature()}
    geometry_mode = str(cfg.fit.geometry_mode).strip().lower()
    report["geometry_mode"] = geometry_mode
    segment_cfg_effective = cfg.fit.segment.model_copy(deep=True)
    simplify_cfg_effective = cfg.fit.simplify.model_copy(deep=True)
    if geometry_mode == "mixed":
        # Mixed mode focuses on geometry recall/fidelity:
        # avoid operations that can shift lines away from the traced skeleton.
        segment_cfg_effective.pre_simplify_enable = True
        segment_cfg_effective.pre_simplify_epsilon_px = 0.35
        simplify_cfg_effective.epsilon_px = 0.0
        segment_cfg_effective.endpoint_snap_tolerance_px = 1e-6
        segment_cfg_effective.join_enable = True
        segment_cfg_effective.join_angle_tolerance_deg = max(float(segment_cfg_effective.join_angle_tolerance_deg), 179.0)
        segment_cfg_effective.post_join_smooth_enable = False
    report["mixed_fidelity_profile"] = {
        "active": geometry_mode == "mixed",
        "pre_simplify_enable": bool(segment_cfg_effective.pre_simplify_enable),
        "simplify_epsilon_px": float(simplify_cfg_effective.epsilon_px),
        "endpoint_snap_tolerance_px": float(segment_cfg_effective.endpoint_snap_tolerance_px),
        "join_enable": bool(segment_cfg_effective.join_enable),
        "post_join_smooth_enable": bool(segment_cfg_effective.post_join_smooth_enable),
    }
    _report_progress(0.0, "start")

    t0 = perf_counter()
    gray = load_grayscale_image(image_path)
    report["timings"]["load_image"] = perf_counter() - t0
    _report_progress(0.05, "load_image")

    t1 = perf_counter()
    binary = preprocess_image(gray, cfg.preprocess)
    binary = _enhance_binary_for_geometry_mode(binary, geometry_mode)
    report["timings"]["preprocess"] = perf_counter() - t1
    _report_progress(0.10, "preprocess")

    t2 = perf_counter()
    skeleton_cfg = cfg.skeleton.model_copy(deep=True)
    if geometry_mode == "mixed":
        # Mixed mode: prune short burrs a bit more aggressively for smoother skeleton overlays.
        skeleton_cfg.spur_min_length_px = max(int(skeleton_cfg.spur_min_length_px), 22)
    skeleton = skeletonize_image(binary, skeleton_cfg)
    report["timings"]["skeletonize"] = perf_counter() - t2
    _report_progress(0.18, "skeletonize")

    t3 = perf_counter()
    mode = cfg.output.mode.lower()
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unsupported output mode: {cfg.output.mode}")

    component_choice, component_choice_reason = _resolve_effective_component_choice(cfg)
    report["component_choice"] = {
        "requested": str(cfg.path_extract.choose_component),
        "effective": component_choice,
        "reason": component_choice_reason,
    }
    if component_choice_reason == "forced_all_for_geometry_mode_mixed" and str(cfg.path_extract.choose_component).strip().lower() != "all":
        report["warnings"].append("path_extract.choose_component forced to all in mixed mode")

    if mode == "single":
        components = [choose_main_component(skeleton, component_choice)]
    else:
        if component_choice == "largest":
            components = [choose_main_component(skeleton, "largest")]
        else:
            components = split_components(skeleton)
            if not components:
                components = [skeleton.copy()]
    _report_progress(0.22, "split_components")

    graphs: list[object] = []
    for i, comp in enumerate(components):
        graphs.append(build_stroke_graph(comp))
        _report_loop_progress(0.22, 0.08, i, len(components), "build_graphs")
    polylines: list[Polyline2D] = []
    raw_path_candidates: list[Polyline2D] = []

    if mode == "single":
        polylines.append(extract_main_open_path(components[0], graphs[0], cfg.path_extract))
        raw_path_candidates = list(polylines)
        _report_progress(0.40, "extract_paths")
    else:
        for i, (comp, graph) in enumerate(zip(components, graphs, strict=True)):
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
            _report_loop_progress(0.30, 0.08, i, len(components), "extract_paths")
        min_len_effective, max_curves_effective, filter_policy = _resolve_effective_path_filter(cfg)
        if filter_policy == "disabled_for_geometry_mode_polyline_only":
            polylines = list(raw_path_candidates)
            path_stats: dict[str, object] = {
                "raw_candidate_count": int(len(raw_path_candidates)),
                "min_length_px": float(min_len_effective),
                "kept_after_min_length": int(len(raw_path_candidates)),
                "dropped_by_min_length": 0,
                "max_curves": int(max_curves_effective),
                "kept_final_count": int(len(raw_path_candidates)),
                "dropped_by_max_curves": 0,
            }
            path_warnings: list[str] = []
        else:
            polylines, path_stats, path_warnings = _filter_multi_output_polylines(
                candidates=raw_path_candidates,
                min_length_px=min_len_effective,
                max_curves=max_curves_effective,
                sort_by=str(cfg.output.multi.sort_by),
            )
        path_stats["filter_policy"] = filter_policy
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
        _report_progress(0.40, "extract_paths")

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
    _report_progress(0.44, "preserve_junctions")

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
    segment_enable_effective, segment_enable_reason = _resolve_effective_segment_enable(cfg)
    report["segment_strategy"] = {
        "geometry_mode": str(cfg.fit.geometry_mode),
        "segment_enable_config": bool(cfg.fit.segment.enable),
        "segment_enable_effective": bool(segment_enable_effective),
        "reason": segment_enable_reason,
        "effective_overrides": {
            "pre_simplify_enable": bool(segment_cfg_effective.pre_simplify_enable),
            "simplify_epsilon_px": float(simplify_cfg_effective.epsilon_px),
            "endpoint_snap_tolerance_px": float(segment_cfg_effective.endpoint_snap_tolerance_px),
            "join_enable": bool(segment_cfg_effective.join_enable),
            "post_join_smooth_enable": bool(segment_cfg_effective.post_join_smooth_enable),
        },
    }

    node_pairs: list[tuple[int, int]] = []
    node_centers: list[tuple[float, float]] = []
    join_stats: dict[str, object] = {
        "enabled": False,
        "before_count": 0,
        "after_count": 0,
        "merged_segments": 0,
        "chains_joined": 0,
        "max_chain_len": 1,
        "post_join_smoothed": False,
    }

    if segment_enable_effective:
        for i, polyline in enumerate(stabilized_list):
            base_anchors_initial = _polyline_anchor_points(
                polyline=polyline,
                protected_points=protected_points[i] if i < len(protected_points) else None,
            )
            polyline_for_split = polyline
            if bool(segment_cfg_effective.pre_simplify_enable):
                pre_cfg = simplify_cfg_effective.model_copy(deep=True)
                pre_eps = float(segment_cfg_effective.pre_simplify_epsilon_px)
                if pre_eps > 0.0:
                    pre_cfg.epsilon_px = pre_eps
                polyline_for_split = simplify_polyline(
                    polyline_for_split,
                    pre_cfg,
                    protected_points=base_anchors_initial,
                )
            base_anchors = _polyline_anchor_points(
                polyline=polyline_for_split,
                protected_points=base_anchors_initial,
            )
            segments = split_polyline_into_segments(
                polyline_for_split,
                segment_cfg_effective,
                forced_break_points=base_anchors,
            )
            for j, seg in enumerate(segments):
                seg_anchor_candidates = anchors_for_segment(
                    seg,
                    anchors=base_anchors,
                    tolerance_px=float(segment_cfg_effective.forced_break_tolerance_px),
                )
                seg_simplified = simplify_polyline(
                    seg,
                    simplify_cfg_effective,
                    protected_points=seg_anchor_candidates,
                )
                segment_polylines.append(seg_simplified)
                segment_anchors.append(_polyline_anchor_points(seg_simplified, seg_anchor_candidates))
                segment_sources.append((i, j))
            _report_loop_progress(0.44, 0.26, i, len(stabilized_list), "split_and_simplify")

        if not segment_polylines:
            raise ValueError("No valid segments after split")

        node_pairs = [(-1, -1) for _ in segment_polylines]
        if float(segment_cfg_effective.endpoint_snap_tolerance_px) > 0.0:
            segment_polylines, node_pairs, node_centers = snap_segment_endpoints(
                segment_polylines,
                tolerance_px=float(segment_cfg_effective.endpoint_snap_tolerance_px),
            )
            segment_anchors = [
                _polyline_anchor_points(segment_polylines[i], segment_anchors[i]) for i in range(len(segment_polylines))
            ]
        _report_progress(0.75, "endpoint_snap")

        join_stats = {
            "enabled": bool(segment_cfg_effective.join_enable),
            "before_count": int(len(segment_polylines)),
            "after_count": int(len(segment_polylines)),
            "merged_segments": 0,
            "chains_joined": 0,
            "max_chain_len": 1,
            "post_join_smoothed": False,
        }
        if bool(segment_cfg_effective.join_enable) and segment_polylines and node_pairs:
            joined_polylines, joined_node_pairs, joined_groups = join_collinear_segments(
                segments=segment_polylines,
                node_pairs=node_pairs,
                angle_tolerance_deg=float(segment_cfg_effective.join_angle_tolerance_deg),
            )
            if joined_polylines:
                rebuilt_anchors: list[list[tuple[float, float]]] = []
                rebuilt_sources: list[tuple[int, int]] = []
                max_chain = 1
                chains_joined = 0
                for i, group in enumerate(joined_groups):
                    merged_anchor_candidates: list[tuple[float, float]] = []
                    merged_sources: list[tuple[int, int]] = []
                    for idx in group:
                        if 0 <= idx < len(segment_anchors):
                            merged_anchor_candidates.extend(segment_anchors[idx])
                        if 0 <= idx < len(segment_sources):
                            merged_sources.append(segment_sources[idx])
                    rebuilt_anchors.append(_polyline_anchor_points(joined_polylines[i], merged_anchor_candidates))
                    rebuilt_sources.append(merged_sources[0] if merged_sources else (0, 0))
                    chain_len = len(group)
                    if chain_len > 1:
                        chains_joined += 1
                        max_chain = max(max_chain, chain_len)

                segment_polylines = joined_polylines
                node_pairs = joined_node_pairs
                segment_anchors = rebuilt_anchors
                segment_sources = rebuilt_sources
                join_stats = {
                    "enabled": True,
                    "before_count": int(sum(len(g) for g in joined_groups)),
                    "after_count": int(len(joined_polylines)),
                    "merged_segments": int(sum(max(0, len(g) - 1) for g in joined_groups)),
                    "chains_joined": int(chains_joined),
                    "max_chain_len": int(max_chain),
                    "post_join_smoothed": False,
                }
        report["join_stats"] = join_stats
        _report_progress(0.80, "join_segments")

        if bool(segment_cfg_effective.post_join_smooth_enable) and segment_polylines:
            smoothed_polylines: list[Polyline2D] = []
            smoothed_anchors: list[list[tuple[float, float]]] = []
            for i, seg in enumerate(segment_polylines):
                smoothed = _moving_average_polyline(
                    seg,
                    window=int(segment_cfg_effective.post_join_smooth_window),
                    passes=int(segment_cfg_effective.post_join_smooth_passes),
                )
                smoothed_polylines.append(smoothed)
                smoothed_anchors.append(_polyline_anchor_points(smoothed, segment_anchors[i]))
                _report_loop_progress(0.80, 0.04, i, len(segment_polylines), "post_join_smooth")
            segment_polylines = smoothed_polylines
            segment_anchors = smoothed_anchors
            join_stats["post_join_smoothed"] = True
        _report_progress(0.84, "post_join_smooth")
    else:
        for i, polyline in enumerate(stabilized_list):
            full_anchor_candidates = _polyline_anchor_points(
                polyline=polyline,
                protected_points=protected_points[i] if i < len(protected_points) else None,
            )
            simplified = simplify_polyline(
                polyline,
                simplify_cfg_effective,
                protected_points=full_anchor_candidates,
            )
            segment_polylines.append(simplified)
            segment_anchors.append(_polyline_anchor_points(simplified, full_anchor_candidates))
            segment_sources.append((i, 0))
            _report_loop_progress(0.44, 0.26, i, len(stabilized_list), "split_and_simplify")

        node_pairs = [(-1, -1) for _ in segment_polylines]
        report["join_stats"] = join_stats
        _report_progress(0.75, "endpoint_snap")
        _report_progress(0.80, "join_segments")
        _report_progress(0.84, "post_join_smooth")

    filtered_polylines: list[Polyline2D] = []
    filtered_anchors: list[list[tuple[float, float]]] = []
    filtered_sources: list[tuple[int, int]] = []
    filtered_node_pairs: list[tuple[int, int]] = []
    min_seg_len = max(0.0, float(segment_cfg_effective.min_length_px))
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
    _report_progress(0.86, "filter_segments")

    geometry_list: list[CurveGeometry] = []
    geometry_types: list[str] = []
    fitted_polylines: list[Polyline2D] = []
    fitted_anchors: list[list[tuple[float, float]]] = []
    fitted_sources: list[tuple[int, int]] = []
    fitted_node_pairs: list[tuple[int, int]] = []
    dropped_before_fit = 0

    for i, polyline in enumerate(segment_polylines):
        try:
            geometry, geometry_type = _fit_geometry_for_segment(polyline, segment_anchors[i], cfg)
        except ValueError as exc:
            # Keep conversion robust when no-filter modes pass tiny/degenerate candidates.
            if "fewer than 2 valid points" in str(exc):
                dropped_before_fit += 1
                _report_loop_progress(0.86, 0.09, i, len(segment_polylines), "fit_geometry")
                continue
            raise

        geometry_list.append(geometry)
        geometry_types.append(geometry_type)
        fitted_polylines.append(polyline)
        fitted_anchors.append(segment_anchors[i])
        fitted_sources.append(segment_sources[i])
        fitted_node_pairs.append(node_pairs[i] if i < len(node_pairs) else (-1, -1))
        _report_loop_progress(0.86, 0.09, i, len(segment_polylines), "fit_geometry")

    if dropped_before_fit > 0:
        report["warnings"].append(f"segments dropped before fit: -{dropped_before_fit}")
    if not geometry_list:
        raise ValueError("No valid segments for geometry fitting")

    segment_polylines = fitted_polylines
    segment_anchors = fitted_anchors
    segment_sources = fitted_sources
    node_pairs = fitted_node_pairs
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
    _report_progress(0.99, "export_3dm")
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

    report["alignment_px"] = _geometry_alignment_metrics(geometry_list, skeleton)

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

    _report_progress(1.0, "done")
    return ExportResult(output_path=export_result.output_path, report=report)
