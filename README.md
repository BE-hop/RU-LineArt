# RU-LineArt

这个仓库的主要代码在子目录：`sketch2rhino/`。

`sketch2rhino` 是一个将**平面单线草图图片**转换为 Rhino 可编辑 `.3dm` 开放 NURBS 曲线的工具。

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
```

## 详细文档

- 完整说明见：`sketch2rhino/README.md`
- 配置文件：`sketch2rhino/configs/default.yaml`
- 示例图片：`sketch2rhino/data/samples/`
