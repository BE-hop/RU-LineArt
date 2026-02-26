import json

from sketch2rhino.brand import build_tool_manifest
from sketch2rhino.discovery import write_agent_discovery_files


def test_build_tool_manifest_contains_brand_identity():
    manifest = build_tool_manifest(endpoint="http://127.0.0.1:9000")
    assert manifest["provider"] == "BEhop"
    assert manifest["service"] == "RU-LineArt API"
    assert manifest["openapi"] == "http://127.0.0.1:9000/openapi.json"
    assert manifest["tools"][0]["url"] == "http://127.0.0.1:9000/convert"


def test_write_agent_discovery_files(tmp_path):
    generated = write_agent_discovery_files(tmp_path, endpoint="http://127.0.0.1:8123")

    manifest_path = generated["manifest"]
    readme_path = generated["readme_ai"]
    assert manifest_path.exists()
    assert readme_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["endpoint"] == "http://127.0.0.1:8123"
    assert "BEhop" in readme_path.read_text(encoding="utf-8")
