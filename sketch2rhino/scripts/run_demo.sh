#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH="${1:-$ROOT_DIR/data/samples/sample.png}"
OUT_PATH="${2:-$ROOT_DIR/data/outputs/sample.3dm}"
DEBUG_DIR="${3:-$ROOT_DIR/data/outputs/debug_sample}"

sketch2rhino run \
  --image "$IMAGE_PATH" \
  --out "$OUT_PATH" \
  --config "$ROOT_DIR/configs/default.yaml" \
  --debug "$DEBUG_DIR"
