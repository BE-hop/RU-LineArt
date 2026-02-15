from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from sketch2rhino.config import load_config
from sketch2rhino.pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """sketch2rhino command-line interface."""


@app.command()
def run(
    image: Path = typer.Option(..., "--image", exists=True, readable=True, help="Input sketch image path"),
    out: Path = typer.Option(..., "--out", help="Output Rhino .3dm path"),
    config: Path | None = typer.Option(None, "--config", exists=True, readable=True, help="YAML config path"),
    debug: Path | None = typer.Option(None, "--debug", help="Debug artifacts output directory"),
) -> None:
    """Run the sketch-to-Rhino pipeline."""
    try:
        cfg = load_config(config)
        result = run_pipeline(image_path=image, output_path=out, cfg=cfg, debug_dir=debug)
    except Exception as exc:  # pragma: no cover - CLI boundary
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Done.[/green] Output: {result.output_path}")
    if result.report:
        points = result.report.get("polyline_points")
        ctrl = result.report.get("control_points")
        console.print(f"[cyan]Polyline points:[/cyan] {points}")
        console.print(f"[cyan]NURBS control points:[/cyan] {ctrl}")


if __name__ == "__main__":  # pragma: no cover
    app()
