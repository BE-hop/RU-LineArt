#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/desktop_app"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
APP_NAME="RU-LineArt"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing virtualenv python: $PYTHON_BIN"
  exit 1
fi

"$APP_DIR/create_icon_assets.sh"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --icon "$APP_DIR/assets/app_icon.icns" \
  --paths "$ROOT_DIR/src" \
  --add-data "$ROOT_DIR/configs:configs" \
  --add-data "$APP_DIR/assets:assets" \
  --distpath "$APP_DIR/dist" \
  --workpath "$APP_DIR/build" \
  --specpath "$APP_DIR" \
  "$APP_DIR/main.py"

cp "$ROOT_DIR/tool_manifest.json" "$APP_DIR/dist/tool_manifest.json"
cp "$ROOT_DIR/README_AI.md" "$APP_DIR/dist/README_AI.md"

echo
echo "Build complete:"
echo "  $APP_DIR/dist/$APP_NAME.app"
echo "  $APP_DIR/dist/tool_manifest.json"
echo "  $APP_DIR/dist/README_AI.md"
