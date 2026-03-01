#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_BIN="$ROOT_DIR/.venv/bin/sketch2rhino"
if [[ ! -x "$CLI_BIN" ]]; then
  CLI_BIN="sketch2rhino"
fi

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$ROOT_DIR/data/outputs/mode_baseline_${STAMP}"
mkdir -p "$OUT_DIR"

STRAIGHT_IMG="$ROOT_DIR/data/samples/直线测试图.jpg"
CURVE_IMG="$ROOT_DIR/data/samples/曲线测试图.jpg"
MIXED_IMG="$ROOT_DIR/data/samples/混合测试图.jpg"
CFG="$ROOT_DIR/configs/default.yaml"

for img in "$STRAIGHT_IMG" "$CURVE_IMG" "$MIXED_IMG"; do
  if [[ ! -f "$img" ]]; then
    echo "Missing baseline image: $img" >&2
    exit 1
  fi
done

echo "[baseline] output directory: $OUT_DIR"

"$CLI_BIN" run \
  --image "$STRAIGHT_IMG" \
  --out "$OUT_DIR/straight_polyline_only.3dm" \
  --config "$CFG" \
  --geometry-mode polyline_only \
  --debug "$OUT_DIR/straight_polyline_only_debug"

"$CLI_BIN" run \
  --image "$CURVE_IMG" \
  --out "$OUT_DIR/curve_nurbs_only.3dm" \
  --config "$CFG" \
  --geometry-mode nurbs_only \
  --debug "$OUT_DIR/curve_nurbs_only_debug"

"$CLI_BIN" run \
  --image "$MIXED_IMG" \
  --out "$OUT_DIR/mixed_mixed.3dm" \
  --config "$CFG" \
  --geometry-mode mixed \
  --debug "$OUT_DIR/mixed_mixed_debug"

echo "[baseline] done:"
echo "  $OUT_DIR/straight_polyline_only.3dm"
echo "  $OUT_DIR/curve_nurbs_only.3dm"
echo "  $OUT_DIR/mixed_mixed.3dm"
