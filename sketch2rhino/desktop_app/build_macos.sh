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

APP_VERSION="$(PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -c 'from sketch2rhino import __version__; print(__version__)')"
RELEASE_DIR="$APP_DIR/release"
PKG_DIR="$RELEASE_DIR/${APP_NAME}-v${APP_VERSION}-macOS"
PKG_ZIP="$RELEASE_DIR/${APP_NAME}-v${APP_VERSION}-macOS.zip"
LATEST_ZIP="$RELEASE_DIR/${APP_NAME}-macOS.zip"

# Enforce version bump per packaging run unless explicitly overridden.
if [ "${RU_LINEART_ALLOW_SAME_VERSION:-0}" != "1" ]; then
  if [ -d "$PKG_DIR" ] || [ -f "$PKG_ZIP" ]; then
    echo "Release ${APP_VERSION} already exists in $RELEASE_DIR."
    echo "Please bump sketch2rhino version before packaging."
    echo "Override intentionally with RU_LINEART_ALLOW_SAME_VERSION=1."
    exit 1
  fi
fi

# Cleanup old generated content before each build.
rm -rf "$APP_DIR/build" "$APP_DIR/dist"
find "$APP_DIR" -name "*.spec" -type f -delete
find "$APP_DIR" -name ".DS_Store" -type f -delete

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

# Keep only the latest macOS release package.
mkdir -p "$RELEASE_DIR"
find "$RELEASE_DIR" -maxdepth 1 -mindepth 1 -type d -name "${APP_NAME}-v*-macOS" ! -name "${APP_NAME}-v${APP_VERSION}-macOS" -exec rm -rf {} +
find "$RELEASE_DIR" -maxdepth 1 -type f -name "${APP_NAME}-v*-macOS.zip" ! -name "${APP_NAME}-v${APP_VERSION}-macOS.zip" -delete
rm -rf "$PKG_DIR"
rm -f "$PKG_ZIP" "$LATEST_ZIP"

mkdir -p "$PKG_DIR"
cp -R "$APP_DIR/dist/$APP_NAME.app" "$PKG_DIR/"
cp "$APP_DIR/dist/tool_manifest.json" "$PKG_DIR/"
cp "$APP_DIR/dist/README_AI.md" "$PKG_DIR/"

(cd "$RELEASE_DIR" && zip -qry "$(basename "$PKG_ZIP")" "$(basename "$PKG_DIR")")
cp "$PKG_ZIP" "$LATEST_ZIP"

echo
echo "Build complete:"
echo "  $APP_DIR/dist/$APP_NAME.app"
echo "  $APP_DIR/dist/tool_manifest.json"
echo "  $APP_DIR/dist/README_AI.md"
echo "  $PKG_DIR"
echo "  $PKG_ZIP"
echo "  $LATEST_ZIP"
