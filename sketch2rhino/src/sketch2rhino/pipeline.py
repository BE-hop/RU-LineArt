from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

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
from sketch2rhino.topo.graph_build import build_stroke_graph
from sketch2rhino.topo.path_extract import (
    choose_main_component,
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

    report["timings"]["path_extract"] = perf_counter() - t3

    t4 = perf_counter()
    simplified_list = [simplify_polyline(polyline, cfg.fit.simplify) for polyline in polylines]
    nurbs_list: list[NurbsSpec] = [fit_open_nurbs(polyline, cfg.fit) for polyline in simplified_list]
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

    report["graph_nodes"] = sum(len(g.node_points) for g in graphs)
    report["graph_edges"] = sum(len(g.edges) for g in graphs)
    report["component_size_px"] = sum(g.component_size_px for g in graphs)
    report["components"] = len(components)

    if debug_dir is not None:
        dbg = ensure_debug_dir(debug_dir)
        save_binary_png(binary, dbg / "01_binarized.png")
        save_skeleton_png(skeleton, dbg / "02_skeleton.png")
        save_path_overlay(gray, polylines if len(polylines) > 1 else polylines[0], dbg / "03_path_overlay.png")
        save_polyline_json(simplified_list if len(simplified_list) > 1 else simplified_list[0], dbg / "04_polyline.json")
        save_nurbs_json(nurbs_list if len(nurbs_list) > 1 else nurbs_list[0], dbg / "05_nurbs.json")
        save_report_json(report, dbg / "report.json")

    return ExportResult(output_path=export_result.output_path, report=report)
