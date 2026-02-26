# sketch2rhino

A local tool that converts a planar sketch image into one or more editable open curves in Rhino.

## Versioning

- Rule: every software update must bump the version and be logged in README.
- Current version: `0.2.4` (2026-02-26)
- Latest changes:
  - Fixed packaged-app TLS fallback by shipping `cacert.pem` inside desktop assets.
  - Update check keeps retry + fallback URL flow (`8s`, `2` attempts, `www` -> non-`www`).
  - Update failure logs include concrete exception reason (DNS/SSL/timeout, etc).
  - Windows packaging still follows GitHub Actions workflow `.github/workflows/build-windows-desktop.yml`.

## What it does
Input:
- A sketch image containing one or more contour strokes (can be jittery, messy, varying thickness).
- Crossings are treated as overpasses (not connected), with topology-preserving pairing in multi mode.

Output:
- A `.3dm` file containing one or more open curves, editable in Rhino.
  Geometry type is controlled by `fit.geometry_mode` (`mixed`, `polyline_only`, `nurbs_only`).
- Optional debug artifacts (binarized image, skeleton, extracted path overlay).

## Design constraints (fixed for MVP)
- ✅ Planar only (outputs curve in XY plane).
- ✅ No scale calibration (you can scale in Rhino).
- ✅ No text/semantic hints.
- ✅ Fully automatic (no user interaction).
- ✅ Output can be single-curve or multi-curve (configurable), all open.
- ✅ Straight / hard-edge strokes can be exported as Polyline (PL) geometry.
- ✅ Control points upper bound: 50 (best effort).

## Quick start

### 1) Create environment (Python 3.11+ recommended)

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\activate   # Windows PowerShell
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

### 2) Run

```bash
.venv/bin/sketch2rhino run \
  --image data/samples/sample.png \
  --out   data/outputs/sample.3dm \
  --config configs/default.yaml \
  --debug data/outputs/debug_sample
```

Included clean samples:
- `data/samples/sample_straight.png`
- `data/samples/sample_curve.png`
- `data/samples/sample_crossing.png`

## CLI

```bash
.venv/bin/sketch2rhino run --image <path> --out <path.3dm> [--config <yaml>] [--debug <dir>] [--geometry-mode <mixed|polyline_only|nurbs_only>]
```

## Common Commands

Input image -> output `.3dm` (use absolute paths):

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/default.yaml \
  --debug /tmp/ru_lineart_debug
```

Only export PL/Polyline curves:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output_polyline_only.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/default.yaml \
  --geometry-mode polyline_only \
  --debug /tmp/ru_lineart_debug_polyline
```

Only export NURBS curves:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output_nurbs_only.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/default.yaml \
  --geometry-mode nurbs_only \
  --debug /tmp/ru_lineart_debug_nurbs
```

Run the rect-like hard-edge fitting test only:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
.venv/bin/python -m pytest -q tests/test_nurbs_fit.py::test_auto_mode_detects_near_straight_as_polyline
```

Run full test suite:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
.venv/bin/python -m pytest -q
```

## Local API (Agent Discoverability)

Start the local API server:

```bash
.venv/bin/sketch2rhino serve --host 127.0.0.1 --port 8000
```

Core discovery endpoints:

- `GET /openapi.json`
- `GET /tool_manifest.json`
- `GET /.well-known/ai-tool.json`
- `GET /health`
- `POST /convert`

Convert request example:

```json
{
  "image_path": "/absolute/path/to/input.png",
  "output_path": "/absolute/path/to/output.3dm",
  "config_path": "/absolute/path/to/config.yaml",
  "debug_dir": "/absolute/path/to/debug_dir",
  "include_report": true
}
```

Generate discovery files for desktop release folders:

```bash
.venv/bin/sketch2rhino agent-files --out-dir ./release_bundle --endpoint http://127.0.0.1:8000
```

Generated files:

- `tool_manifest.json` (machine-readable API manifest)
- `README_AI.md` (agent usage instructions)

## Output Modes

The tool supports two path-count modes via `configs/default.yaml`:

- `output.mode: "single"`: extract and export one main open curve.
- `output.mode: "multi"`: extract and export multiple open curves.

Geometry export mode is independent from path count:

- `fit.geometry_mode: "mixed"` (default): auto classify each segment to Polyline or NURBS.
- `fit.geometry_mode: "polyline_only"`: force all exported geometry to Polyline.
- `fit.geometry_mode: "nurbs_only"`: force all exported geometry to NURBS.

You can set `fit.geometry_mode` in YAML, or override on CLI with `--geometry-mode`.

Multi-mode controls:

- `output.multi.min_length_px`: drop tiny short strokes.
- `output.multi.max_curves`: cap number of exported curves (`0` means no cap).
- `output.multi.include_loops`: if true, closed loops are cut open and exported.
- `output.multi.sort_by`: currently supports `length`.
- `output.multi.preserve_junctions`: keep detected junction anchors in simplification/fitting.
- `output.multi.junction_snap_radius_px`: snap nearby stroke points to the same junction anchor.

For multi-mode across disjoint parts, keep:

- `path_extract.choose_component: "all"`

To force legacy single-component behavior, set:

- `path_extract.choose_component: "largest"`

Crossing-focused extraction controls:

- `path_extract.cluster_radius_px`: groups nearby junction pixels into one stable node.
- `path_extract.junction_bridge_max_px`: merges split junction clusters connected by a short bridge.
- `path_extract.tangent_k`: tangent sampling distance used for direction pairing at junctions.

Dense/junction-heavy drawings:

- Set `path_extract.crossing_policy: "junction_split"` to break continuation at branch nodes and reduce wrong straight-through links.
- Use `configs/dense_topology.yaml` when central dense regions are messy and missing details:

```bash
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output_dense.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/dense_topology.yaml \
  --debug /tmp/ru_lineart_debug_dense
```

For tightly parallel double-lines + rectangle-heavy plans, prefer:

```bash
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output_parallel_detail.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/parallel_detail.yaml \
  --debug /tmp/ru_lineart_debug_parallel_detail
```

`configs/parallel_detail.yaml` keeps endpoint/node snap conservative for close parallel lines:
- `fit.segment.endpoint_snap_tolerance_px: 1.2`
- `output.multi.junction_snap_radius_px: 2.0`

Recommended smoothing-first defaults (already reflected in `configs/default.yaml`):

- `fit.geometry_mode: "mixed"`
- `fit.stabilize.enable: false` (set `true` for hand-drawn anti-jitter)
- `fit.stabilize.method: "savgol"`
- `fit.stabilize.window: 11`
- `fit.stabilize.passes: 2`
- `fit.stabilize.blend: 0.7`
- `fit.simplify.epsilon_px: 1.2`
- `fit.simplify.smooth_enable: false`
- `fit.simplify.smooth_window: 7`
- `fit.simplify.smooth_passes: 1`
- `fit.segment.enable: true` (split mixed straight/curved strokes into multiple segments)
- `fit.segment.corner_split_angle_deg: 45.0`
- `fit.segment.corner_scales_px: [6, 12, 24]` (multi-scale turning-angle detection)
- `fit.segment.fillet_enable: true` (detect rounded corner transitions and split at tangency-like boundaries)
- `fit.segment.endpoint_snap_tolerance_px: 2.5` (snap segment endpoints to same node for easy Join)
- `fit.spline.mode: "auto"` (`"smooth"` for always-spline, `"hard_edge"` for always-polyline-like)
- `fit.spline.smoothing: 6.0`
- `fit.spline.hard_edge_straight_max_deviation_px: 2.0`
- `fit.spline.hard_edge_straight_max_deviation_ratio: 0.02`
- `fit.spline.hard_edge_straight_max_turn_deg: 12.0`
- `output.multi.junction_snap_radius_px: 3.0`
- `fit.spline.anchor_correction_enable: true`
- `fit.spline.anchor_tolerance_px: 2.0`
- `fit.spline.anchor_snap_max_dist_px: 10.0`
- `fit.spline.anchor_weight: 20.0`

If curve shape becomes too loose, reduce `fit.spline.smoothing` first (e.g. `6 -> 3`).
If curves become too jagged, increase `fit.spline.smoothing` slightly (e.g. `6 -> 10`).
If rectangle/building edges are rounded too much, keep `fit.spline.mode: "auto"` (or force `"hard_edge"`).
If one long stroke contains both straight and curved regions, keep `fit.segment.enable: true`.
If segment endpoints are hard to Join in Rhino, increase `fit.segment.endpoint_snap_tolerance_px` (e.g. `2.5 -> 3.5`).
If straight lines are still treated as curves, relax straight detection a bit:
- raise `fit.spline.hard_edge_straight_max_deviation_px` (e.g. `2.0 -> 3.0`)
- or raise `fit.spline.hard_edge_straight_max_turn_deg` (e.g. `12 -> 16`)
If crossings drift apart, increase `fit.spline.anchor_weight` (e.g. `20 -> 28`).
If hand-drawn jitter remains, enable `fit.stabilize.enable` and raise `fit.stabilize.window` (e.g. `11 -> 15`).
If closely parallel double-lines get merged too early, try:
- reduce junction merge first:
  - lower `path_extract.cluster_radius_px` (e.g. `4.0 -> 2.0 -> 1.5`)
  - lower `path_extract.junction_bridge_max_px` (e.g. `12.0 -> 4.0 -> 3.0`)
- if skeleton still looks too thick, thin binarization slightly:
  - lower `preprocess.binarize.otsu_offset` (e.g. `0 -> -4 -> -8`)
- avoid aggressive erosion by default (`preprocess.morph.erode_iter: 0`), because it can drop weak strokes.

Hand-drawn preset (stabilization enabled):

```bash
.venv/bin/sketch2rhino run \
  --image data/samples/sample.png \
  --out   data/outputs/sample_smooth.3dm \
  --config configs/handdrawn_smooth.yaml \
  --debug data/outputs/debug_sample_smooth
```

## Safe Environment Workflow (Important)

To avoid accidentally mixing Conda `base` packages with this project:

- Always install with `.venv/bin/python -m pip ...`
- Always test with `.venv/bin/python -m pytest ...`
- Always run CLI with `.venv/bin/sketch2rhino ...`
- Do **not** use bare `pip`, bare `pytest`, or bare `python` in this project directory unless you verified they point to `.venv`.

Quick sanity checks:

```bash
which python
which pytest
.venv/bin/python -c "import sys; print(sys.executable)"
```

If `pytest` resolves to something like `/opt/anaconda3/bin/pytest`, you are not running inside the project venv.

## Pipeline overview

1. Preprocess:
- denoise / contrast
- binarize (black strokes on white)
- morphological cleanup

2. Skeletonize:
- convert thick strokes into 1-pixel-wide skeleton

3. Graph + main path:
- build a pixel graph from skeleton
- remove tiny spurs
- build a junction-clustered super-graph and extract stroke paths (single or multi mode)
- crossings are treated as overpasses using direction-based continuation pairing

4. Fit:
- optional stroke stabilization (Procreate-like anti-jitter)
- simplify polyline
- split each path into piecewise segments (corner/transition/anchor aware)
- snap segment endpoints to shared nodes (for post-edit Join)
- auto classify stroke geometry:
  - straight / hard-edge -> Polyline (PL)
  - smooth curvy -> open B-spline / NURBS (control points ≤ 50, best-effort)

5. Export:
- write `.3dm` using `rhino3dm`

## Brand Metadata Strategy

The project now ships with machine-readable brand identity in multiple layers:

- API schema metadata includes provider and author fields (`x-provider`, `x-author`, `x-service`).
- `tool_manifest.json` includes `service/provider/author/url`.
- `.3dm` export writes metadata into:
  - document user text (`behop.*`, `generated_by`, `generator`)
  - per-object user strings (`behop.*`, `generated_by`, `generator`)
- Conversion/API logs use `[BEhop AI]` prefix for observability and attribution.

This keeps outputs clean (no visual watermark) while preserving attribution for agents and automation systems.

## Debug outputs

If `--debug` is provided, the tool saves:

- `01_binarized.png`
- `02_skeleton.png`
- `03_path_overlay.png` (raw extracted paths before segment split/fitting)
- `03_segment_overlay.png` (post-split segments that are actually sent to fitting/export)
- `04_polyline.json`
- `05_nurbs.json`
- `report.json` (timings, counts, warnings)

## Known limitations

- Very dense tangles can still create small extra fragments; raise `output.multi.min_length_px` to filter them.
- Extremely close parallel strokes may still merge during skeletonization at low resolution.
- 3D perspective sketches are treated as 2D projections.

## Topology-Preserving Strategy (Current Default)

For crossing-heavy clean line art, extraction follows:

1. Skeleton -> pixel graph.
2. Junction clustering (`cluster_radius_px`) + short-bridge merge (`junction_bridge_max_px`).
3. Keep parallel super-edges (important: do not collapse distinct stroke branches).
4. At each junction, pair incident edges by best straight-through direction continuity.
5. Trace open strokes from endpoints.
6. Preserve/snap junction anchors during simplification and fitting.

This is the recommended baseline when you want results similar to red/green stroke overlays in debug images.

## Development

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Troubleshooting

### NumPy / SciPy binary incompatibility

If you see errors like:

- `numpy.dtype size changed`
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

then your environment has incompatible wheel versions (often caused by mixing Conda/base and venv, or upgrading NumPy to 2.x while SciPy/skimage were built for NumPy 1.x).

Use this recovery flow:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
rm -rf .venv
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Project dependency guardrails are pinned in `pyproject.toml` to avoid known incompatible combinations (notably NumPy 2.x with older SciPy/skimage wheels).

### Matplotlib cache warning (`/Users/.../.matplotlib is not a writable directory`)

This warning is non-fatal for this project. If you want to silence it:

```bash
mkdir -p .tmp/mplconfig
export MPLCONFIGDIR="$(pwd)/.tmp/mplconfig"
```

### Why control points look like "one above, one below"

In NURBS, control points define a control polygon and are not expected to lie on the curve centerline.
Alternating control-point positions can be normal.

If the final curve itself drifts locally (not just the control polygon), common causes are:

- over-constrained duplicate junction anchors around one crossing area
- over-smoothed fitting (`fit.spline.smoothing` too high)
- too-weak simplification causing pixel-zigzag to leak into fitting

Current pipeline mitigations:

- junction anchors come from the same super-graph used for stroke extraction
- near-duplicate anchors are merged before fitting
- `report.json` includes `anchor_error_px` for quantitative checks

## Roadmap ideas (optional)

- Better crossing disentanglement
- Robust main-path selection (closed-loop handling)
- Optional interactive correction
