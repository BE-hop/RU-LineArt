# RU-LineArt

这个仓库的主要代码在子目录：`sketch2rhino/`。

`sketch2rhino` 是一个将**平面单线草图图片**转换为 Rhino 可编辑 `.3dm` 开放曲线（自动混合 NURBS 与 Polyline）的工具。

现在支持本地 API（FastAPI）与 Agent 可发现能力：

- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Manifest: `http://127.0.0.1:8000/tool_manifest.json`
- Convert API: `POST http://127.0.0.1:8000/convert`

## 快速开始

```bash
cd sketch2rhino
python -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .

.venv/bin/sketch2rhino run \
  --image data/samples/sample.png \
  --out   data/outputs/sample.3dm \
  --config configs/default.yaml \
  --debug data/outputs/debug_sample

# 启动本地 API
.venv/bin/sketch2rhino serve --host 127.0.0.1 --port 8000
```

## 常用测试命令

```bash
cd /Users/mac/Documents/RU-LineArt/sketch2rhino

# 输入图片 -> 输出 3dm（绝对路径）
.venv/bin/sketch2rhino run \
  --image /absolute/path/to/input.png \
  --out /absolute/path/to/output.3dm \
  --config /Users/mac/Documents/RU-LineArt/sketch2rhino/configs/default.yaml \
  --debug /tmp/ru_lineart_debug

# 仅跑直线/矩形硬边拟合相关测试
.venv/bin/python -m pytest -q tests/test_nurbs_fit.py::test_auto_mode_detects_near_straight_as_polyline

# 全量测试
.venv/bin/python -m pytest -q
```

## 详细文档

- 完整说明见：`sketch2rhino/README.md`
- 配置文件：`sketch2rhino/configs/default.yaml`
- 示例图片：`sketch2rhino/data/samples/`
