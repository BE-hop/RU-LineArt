param(
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $RootDir "desktop_app"
$AppName = "RU-LineArt"

if (-not $PythonBin) {
    $VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonBin = $VenvPython
    } else {
        $PythonBin = "python"
    }
}

if ($PythonBin -ne "python" -and -not (Test-Path $PythonBin)) {
    throw "Python not found: $PythonBin"
}

& $PythonBin "$AppDir\create_icon_assets.py" --target windows

& $PythonBin -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "$AppName" `
    --icon "$AppDir\assets\app_icon.ico" `
    --paths "$RootDir\src" `
    "--add-data=$RootDir\configs;configs" `
    "--add-data=$AppDir\assets;assets" `
    --distpath "$AppDir\dist_windows" `
    --workpath "$AppDir\build_windows" `
    --specpath "$AppDir" `
    "$AppDir\main.py"

$ReleaseDir = Join-Path $AppDir "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

$ZipPath = Join-Path $ReleaseDir "$AppName-windows.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Compress-Archive -Path "$AppDir\dist_windows\$AppName" -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Build complete:"
Write-Host "  EXE folder: $AppDir\dist_windows\$AppName"
Write-Host "  ZIP: $ZipPath"
