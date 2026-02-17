#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/desktop_app"

rm -rf "$APP_DIR/__pycache__" \
       "$APP_DIR/build" \
       "$APP_DIR/build_windows" \
       "$APP_DIR/dist" \
       "$APP_DIR/dist_windows"

find "$APP_DIR" -name "*.spec" -type f -delete
find "$APP_DIR" -name ".DS_Store" -type f -delete

echo "Cleaned generated build artifacts under:"
echo "  $APP_DIR"
