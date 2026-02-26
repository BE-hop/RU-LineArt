from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from sketch2rhino import __version__
from sketch2rhino.brand import LOG_PREFIX, BRAND_URL, build_tool_manifest, brand_signature

LOGGER = logging.getLogger("sketch2rhino.api")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ConvertByPathRequest(BaseModel):
    image_path: str = Field(..., description="Input sketch image path")
    output_path: str | None = Field(default=None, description="Output .3dm path. Default: image_path with .3dm suffix")
    config_path: str | None = Field(default=None, description="Optional YAML config path")
    debug_dir: str | None = Field(default=None, description="Optional debug directory")
    include_report: bool = Field(default=True, description="Return debug report in response")


class ConvertResponse(BaseModel):
    status: str
    trace_id: str
    output_path: str
    report: dict[str, object] | None = None
    brand: dict[str, str]


app = FastAPI(
    title="RU-LineArt API",
    version=__version__,
    description="Local API for sketch-to-Rhino conversion with BEhop brand identity metadata.",
    contact={"name": "BEhop", "url": BRAND_URL},
)


def _custom_openapi() -> dict[str, object]:
    if app.openapi_schema is not None:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    info = schema.setdefault("info", {})
    info["x-provider"] = "BEhop"
    info["x-author"] = "BEhop Design AI Lab"
    info["x-service"] = "RU-LineArt API"
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[assignment]


def _as_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser()


def _validate_image_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail=f"Input image does not exist: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(IMAGE_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported image extension: {path.suffix}. Allowed: {allowed}")


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    return {"status": "ok", "version": __version__, "brand": brand_signature()}


@app.get("/brand", tags=["discoverability"])
def brand() -> dict[str, str]:
    return brand_signature()


@app.get("/tool-manifest", tags=["discoverability"])
@app.get("/tool_manifest.json", tags=["discoverability"])
@app.get("/.well-known/ai-tool.json", tags=["discoverability"])
def tool_manifest(request: Request) -> dict[str, object]:
    endpoint = str(request.base_url).rstrip("/")
    return build_tool_manifest(endpoint=endpoint)


@app.post("/convert", response_model=ConvertResponse, tags=["convert"])
def convert(payload: ConvertByPathRequest) -> ConvertResponse:
    from sketch2rhino.config import load_config
    from sketch2rhino.pipeline import run_pipeline

    trace_id = uuid4().hex[:12]
    image_path = _as_path(payload.image_path)
    if image_path is None:
        raise HTTPException(status_code=400, detail="image_path is required")
    _validate_image_path(image_path)

    output_path = _as_path(payload.output_path) or image_path.with_suffix(".3dm")
    if output_path.suffix.lower() != ".3dm":
        output_path = output_path.with_suffix(".3dm")

    config_path = _as_path(payload.config_path)
    debug_dir = _as_path(payload.debug_dir)

    try:
        cfg = load_config(config_path)
        result = run_pipeline(
            image_path=image_path,
            output_path=output_path,
            cfg=cfg,
            debug_dir=debug_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("%s trace=%s conversion_failed", LOG_PREFIX, trace_id)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc}") from exc

    LOGGER.info(
        "%s trace=%s convert image=%s output=%s",
        LOG_PREFIX,
        trace_id,
        image_path,
        result.output_path,
    )
    return ConvertResponse(
        status="ok",
        trace_id=trace_id,
        output_path=str(result.output_path),
        report=result.report if payload.include_report else None,
        brand=brand_signature(),
    )
