from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from sketch2rhino.brand import LOG_PREFIX, build_tool_manifest
from sketch2rhino.config import load_config
from sketch2rhino.discovery import write_agent_discovery_files

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
    geometry_mode: str | None = typer.Option(
        None,
        "--geometry-mode",
        help="Geometry mode override: mixed | polyline_only | nurbs_only",
    ),
) -> None:
    """Run the sketch-to-Rhino pipeline."""
    from sketch2rhino.pipeline import run_pipeline

    try:
        cfg = load_config(config)
        if geometry_mode is not None:
            mode = geometry_mode.strip().lower()
            if mode not in {"mixed", "polyline_only", "nurbs_only"}:
                raise ValueError("--geometry-mode must be one of: mixed, polyline_only, nurbs_only")
            cfg.fit.geometry_mode = mode
        result = run_pipeline(image_path=image, output_path=out, cfg=cfg, debug_dir=debug)
    except Exception as exc:  # pragma: no cover - CLI boundary
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Done.[/green] Output: {result.output_path}")
    if result.report:
        points = result.report.get("polyline_points")
        ctrl = result.report.get("control_points")
        geometry = result.report.get("geometry_type", result.report.get("geometry_types"))
        console.print(f"[cyan]Polyline points:[/cyan] {points}")
        console.print(f"[cyan]Export geometry:[/cyan] {geometry}")
        console.print(f"[cyan]Curve points/control points:[/cyan] {ctrl}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host for local API server"),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Port for local API server"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Run local RU-LineArt API server."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        console.print("[red]Missing dependency:[/red] uvicorn")
        raise typer.Exit(code=1) from exc

    endpoint = f"http://{host}:{port}"
    console.print(f"[green]{LOG_PREFIX} Serving on {endpoint}[/green]")
    console.print(f"[cyan]{LOG_PREFIX} OpenAPI:[/cyan] {endpoint}/openapi.json")
    console.print(f"[cyan]{LOG_PREFIX} Manifest:[/cyan] {endpoint}/tool_manifest.json")
    uvicorn.run("sketch2rhino.api_server:app", host=host, port=port, reload=reload)


@app.command("agent-files")
def agent_files(
    out_dir: Path = typer.Option(Path("."), "--out-dir", help="Output directory for agent-discovery files"),
    endpoint: str = typer.Option("http://127.0.0.1:8000", "--endpoint", help="API endpoint to write into manifest"),
) -> None:
    """Generate tool_manifest.json and README_AI.md for agent discovery."""
    generated = write_agent_discovery_files(out_dir, endpoint=endpoint)
    manifest = build_tool_manifest(endpoint=endpoint)
    console.print(f"[green]{LOG_PREFIX} Generated:[/green] {generated['manifest']}")
    console.print(f"[green]{LOG_PREFIX} Generated:[/green] {generated['readme_ai']}")
    console.print(f"[cyan]{LOG_PREFIX} Convert endpoint:[/cyan] {manifest['tools'][0]['url']}")


if __name__ == "__main__":  # pragma: no cover
    app()
