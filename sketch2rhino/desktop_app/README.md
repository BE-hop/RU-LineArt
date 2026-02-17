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
- App icon includes `BEhop` and `RU-LineArt`.

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
