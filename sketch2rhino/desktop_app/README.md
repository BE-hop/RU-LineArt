# RU-LineArt Desktop App

This folder contains a local desktop wrapper for `sketch2rhino`.

## Structure guide

- Full folder responsibilities: `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/FOLDER_GUIDE.md`

## What you get

- Bilingual interface (Chinese + English).
- Drag and drop image into the window.
- Choose image manually.
- Choose output `.3dm` path.
- Optional custom YAML config.
- Geometry output mode selector: `直线 / 曲线 / 混合` (`Straight / Curves / Mixed`), default is `曲线 / Curves`.
- App icon includes `BEhop` and `RU-LineArt`.

## Versioning

- Rule: each software update must bump version and update README notes.
- Current version: `0.2.8` (2026-02-27)

## Run locally (development)

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
chmod +x desktop_app/run_local.sh
./desktop_app/run_local.sh
```

## Build macOS app bundle

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
chmod +x desktop_app/build_macos.sh
./desktop_app/build_macos.sh
```

Expected output:

`/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/dist/RU-LineArt.app`

Build behavior (auto cleanup):

- Deletes old macOS release folders/zips (`RU-LineArt-v*-macOS*`) before packaging.
- Deletes stale build cache (`build/`, `dist/`, `.spec`, `.DS_Store`) before each build.
- Keeps only current version package + `RU-LineArt-macOS.zip`.

The macOS dist folder also includes:

- `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/dist/tool_manifest.json`
- `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/dist/README_AI.md`

These files help agents discover and call the local API quickly.

## Icon customization

- Selected icon source (A): `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/assets/icon_candidates/candidate_a_curve-first.png`
- Icon build script: `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/create_icon_assets.py`
- Regenerate icon assets:

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
chmod +x desktop_app/create_icon_assets.sh
./desktop_app/create_icon_assets.sh
```

If Finder still shows an old icon after rebuild:

1. Remove old app copy from `/Applications` (if exists).
2. Re-copy latest app from `desktop_app/release/RU-LineArt.app`.
3. Restart Finder:

```bash
killall Finder
```

## Build Windows app

Local Windows machine:

```powershell
cd C:\path\to\RU-LineArt\sketch2rhino
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyinstaller pyside6 pillow
.\desktop_app\build_windows.ps1 -PythonBin "python"
```

Output:

`desktop_app\release\RU-LineArt-windows.zip`

The packaged Windows output includes `tool_manifest.json` and `README_AI.md` next to `RU-LineArt.exe` before zipping.

GitHub Actions:

- Workflow file: `/Users/mac/Documents/RU-LineArt/.github/workflows/build-windows-desktop.yml`
- Trigger manually from Actions page: `Build RU-LineArt Windows App`
- Download artifact: `RU-LineArt-windows`
- Artifact content: `RU-LineArt-windows.zip` (contains `RU-LineArt.exe` and runtime files)
- Recommended final step: move downloaded zip to `desktop_app/release/` for unified release management

Detailed pipeline reference:

- `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/FOLDER_GUIDE.md`

## Cleanup generated build cache

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino
chmod +x desktop_app/clean_artifacts.sh
./desktop_app/clean_artifacts.sh
```

## Distribution notes

- Build on macOS for macOS users.
- Build on Windows for Windows users (cross-platform build is not recommended for desktop apps).
- If you want notarized/signable macOS distribution, add code signing in a separate release step.

## Semi-auto update check (launch-time)

The desktop app now checks a remote JSON feed every time it starts.

- Feed URL (default): `https://www.behop.cn/behop-ai-product/products/ru-lineart/version.json`
- Product page fallback: `https://www.behop.cn/behop-ai-product/products/ru-lineart/`
- Local test override: set env var `RU_LINEART_UPDATE_JSON_URL`

Behavior:

1. App requests JSON feed.
2. If `latest` is newer than current version, it shows an update dialog.
3. If `latest` is equal to current version, no popup is shown (this is expected), and a "already latest" log is written.
4. If network fails or feed JSON is invalid, no popup is shown, and concrete failure reason is written in the log.
5. If system certificate chain validation fails, app retries with bundled CA certificate file (`assets/cacert.pem`).
6. User clicks update, app opens the `page` URL in browser.
7. If `force: true`, app requires update and exits after prompt.

JSON format:

```json
{
  "latest": "0.2.8",
  "force": false,
  "page": "https://www.behop.cn/behop-ai-product/products/ru-lineart/",
  "notes": "Bug fixes and quality improvements."
}
```

Example file:

- `/Users/mac/Documents/RU-LineArt/sketch2rhino/desktop_app/update_feed.example.json`
