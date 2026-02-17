# sketch2rhino

A local tool that converts a planar sketch image into one or more editable open NURBS curves in Rhino.

## What it does
Input:
- A sketch image containing one or more contour strokes (can be jittery, messy, varying thickness).
- Crossings are treated as overpasses (not connected), with topology-preserving pairing in multi mode.

Output:
- A `.3dm` file containing one or more open NURBS curves (mode-dependent), editable in Rhino.
- Optional debug artifacts (binarized image, skeleton, extracted path overlay).

## Design constraints (fixed for MVP)
- ✅ Planar only (outputs curve in XY plane).
- ✅ No scale calibration (you can scale in Rhino).
- ✅ No text/semantic hints.
- ✅ Fully automatic (no user interaction).
- ✅ Output can be single-curve or multi-curve (configurable), all open.
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
.venv/bin/sketch2rhino run --image <path> --out <path.3dm> [--config <yaml>] [--debug <dir>]
```

## Output Modes

The tool supports two modes via `configs/default.yaml`:

- `output.mode: "single"`: extract and export one main open curve.
- `output.mode: "multi"`: extract and export multiple open curves.

Multi-mode controls:

- `output.multi.min_length_px`: drop tiny short strokes.
- `output.multi.max_curves`: cap number of exported curves.
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

Recommended smoothing-first defaults (already reflected in `configs/default.yaml`):

- `fit.stabilize.enable: false` (set `true` for hand-drawn anti-jitter)
- `fit.stabilize.method: "savgol"`
- `fit.stabilize.window: 11`
- `fit.stabilize.passes: 2`
- `fit.stabilize.blend: 0.7`
- `fit.simplify.epsilon_px: 1.2`
- `fit.simplify.smooth_enable: false`
- `fit.simplify.smooth_window: 7`
- `fit.simplify.smooth_passes: 1`
- `fit.spline.smoothing: 6.0`
- `output.multi.junction_snap_radius_px: 3.0`
- `fit.spline.anchor_correction_enable: true`
- `fit.spline.anchor_tolerance_px: 2.0`
- `fit.spline.anchor_snap_max_dist_px: 10.0`
- `fit.spline.anchor_weight: 20.0`

If curve shape becomes too loose, reduce `fit.spline.smoothing` first (e.g. `6 -> 3`).
If curves become too jagged, increase `fit.spline.smoothing` slightly (e.g. `6 -> 10`).
If crossings drift apart, increase `fit.spline.anchor_weight` (e.g. `20 -> 28`).
If hand-drawn jitter remains, enable `fit.stabilize.enable` and raise `fit.stabilize.window` (e.g. `11 -> 15`).

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
- fit an open B-spline / NURBS with control points ≤ 50 (best-effort)

5. Export:
- write `.3dm` using `rhino3dm`

## Debug outputs

If `--debug` is provided, the tool saves:

- `01_binarized.png`
- `02_skeleton.png`
- `03_path_overlay.png`
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
