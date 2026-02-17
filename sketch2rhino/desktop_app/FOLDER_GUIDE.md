# RU-LineArt Desktop Folder Guide

## Core source folders

- `desktop_app/main.py`
  - Desktop GUI entrypoint (Chinese/English UI, drag-drop, conversion trigger).
- `desktop_app/assets/icon_candidates/`
  - Source icon design candidates.
  - Current selected icon source: `candidate_a_curve-first.png`.
- `desktop_app/assets/`
  - Generated icon assets used by packagers:
  - `app_icon.png` (shared base)
  - `app_icon.icns` (macOS)
  - `app_icon.ico` (Windows)

## Build scripts

- `desktop_app/create_icon_assets.py`
  - Converts selected icon source into `.png/.icns/.ico`.
- `desktop_app/create_icon_assets.sh`
  - macOS wrapper for icon asset generation.
- `desktop_app/build_macos.sh`
  - Builds `RU-LineArt.app` (PyInstaller, macOS).
- `desktop_app/build_windows.ps1`
  - Builds `RU-LineArt.exe` bundle and zip on Windows.
- `desktop_app/clean_artifacts.sh`
  - Removes generated build cache (`build*/dist*`, `*.spec`, `.DS_Store`).

## CI / automation

- `.github/workflows/build-windows-desktop.yml`
  - GitHub Actions workflow for Windows packaging.
  - Trigger mode: manual (`workflow_dispatch`).
  - Output artifact: `RU-LineArt-windows`.

## Generated artifact folders (do not hand-edit)

- `desktop_app/build/` and `desktop_app/build_windows/`
  - Intermediate PyInstaller build cache.
- `desktop_app/dist/` and `desktop_app/dist_windows/`
  - Raw packaged outputs.
- `desktop_app/release/`
  - Final deliverables for sharing:
  - `RU-LineArt-macOS.zip`
  - `RU-LineArt-windows.zip` (after Windows build)

## Recommended operating procedure

1. Keep source edits in `main.py`, icon candidates, scripts, and workflow file only.
2. Regenerate icon assets before packaging (`create_icon_assets.py`).
3. Build macOS locally with `build_macos.sh`.
4. Build Windows via GitHub Actions (`build-windows-desktop.yml`) or on a Windows machine with `build_windows.ps1`.
5. Share only files inside `desktop_app/release/`.

## Windows packaging flow (GitHub Actions)

1. Push repository changes (including `.github/workflows/build-windows-desktop.yml`) to GitHub.
2. Open `Actions` tab and run `Build RU-LineArt Windows App`.
3. Wait for job success (`build-windows`).
4. Download artifact `RU-LineArt-windows`.
5. Move/downloaded zip into local `desktop_app/release/` for unified release archive management.
