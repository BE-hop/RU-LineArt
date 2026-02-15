# sketch2rhino

A local tool that converts a **single-line planar sketch image** into **one editable open NURBS curve** in Rhino.

## What it does
Input:
- A sketch image containing **one main contour stroke** (can be jittery, messy, varying thickness).
- Crossings are treated as **overpasses** (not connected).

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

For multi-mode across disjoint parts, keep:

- `path_extract.choose_component: "all"`

To force legacy single-component behavior, set:

- `path_extract.choose_component: "largest"`

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
- extract **one main open path** (crossings are overpasses)

4. Fit:
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

- If the input has multiple disjoint strokes, the tool selects the largest/main component.
- If the sketch is extremely tangled, the extracted main path may deviate.
- 3D perspective sketches are treated as 2D projections.

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

## Roadmap ideas (optional)

- Better crossing disentanglement
- Robust main-path selection (closed-loop handling)
- Optional interactive correction
