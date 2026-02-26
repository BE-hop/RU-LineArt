from starlette.requests import Request

from sketch2rhino.api_server import app, health, tool_manifest


def _request_for(base_url: str) -> Request:
    scheme, host_port = base_url.split("://", 1)
    if ":" in host_port:
        host, port_text = host_port.split(":", 1)
        port = int(port_text)
    else:
        host, port = host_port, 80

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/tool_manifest.json",
        "raw_path": b"/tool_manifest.json",
        "scheme": scheme,
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": (host, port),
        "root_path": "",
        "app": app,
    }
    return Request(scope)


def test_health_and_manifest_endpoints():
    health_payload = health()
    assert health_payload["status"] == "ok"
    assert health_payload["brand"]["provider"] == "BEhop"

    manifest_payload = tool_manifest(_request_for("http://127.0.0.1:8000"))
    assert manifest_payload["provider"] == "BEhop"
    assert manifest_payload["tools"][0]["path"] == "/convert"
    assert manifest_payload["openapi"] == "http://127.0.0.1:8000/openapi.json"


def test_openapi_contains_brand_extensions():
    schema = app.openapi()
    info = schema["info"]

    assert info["x-provider"] == "BEhop"
    assert info["x-author"] == "BEhop Design AI Lab"
    assert info["x-service"] == "RU-LineArt API"
