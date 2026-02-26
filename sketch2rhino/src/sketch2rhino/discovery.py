from __future__ import annotations

import json
from pathlib import Path

from sketch2rhino.brand import build_tool_manifest, generated_by_text


def build_ai_readme(endpoint: str = "http://127.0.0.1:8000") -> str:
    manifest = build_tool_manifest(endpoint=endpoint)
    return f"""# RU-LineArt Agent Quickstart

This folder exposes machine-readable API entry points for RU-LineArt.

- Manifest: `tool_manifest.json`
- OpenAPI: `{manifest["openapi"]}`
- Main tool: `POST {endpoint.rstrip("/")}/convert`

## Input for convert

JSON body:

```json
{{
  "image_path": "/absolute/path/to/input.png",
  "output_path": "/absolute/path/to/output.3dm",
  "config_path": "/absolute/path/to/config.yaml",
  "debug_dir": "/absolute/path/to/debug_dir",
  "include_report": true
}}
```

## Brand identity

- Provider: `{manifest["provider"]}`
- Author: `{manifest["author"]}`
- URL: `{manifest["url"]}`
- Signature: `{generated_by_text()}`
"""


def write_agent_discovery_files(
    output_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8000",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "tool_manifest.json"
    readme_ai_path = out_dir / "README_AI.md"

    manifest = build_tool_manifest(endpoint=endpoint)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_ai_path.write_text(build_ai_readme(endpoint=endpoint), encoding="utf-8")

    return {"manifest": manifest_path, "readme_ai": readme_ai_path}
