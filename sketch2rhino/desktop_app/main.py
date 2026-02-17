from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


def _ensure_import_path() -> None:
    """Allow running from repo without requiring an installed wheel."""
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_import_path()

from sketch2rhino.config import load_config  # noqa: E402
from sketch2rhino.pipeline import run_pipeline  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


def default_config_path() -> Path | None:
    if hasattr(sys, "_MEIPASS"):
        bundled = Path(getattr(sys, "_MEIPASS")) / "configs" / "default.yaml"
        if bundled.exists():
            return bundled

    repo_default = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    if repo_default.exists():
        return repo_default
    return None


def default_icon_path() -> Path | None:
    if hasattr(sys, "_MEIPASS"):
        bundled = Path(getattr(sys, "_MEIPASS")) / "assets" / "app_icon.png"
        if bundled.exists():
            return bundled

    local_icon = Path(__file__).resolve().parent / "assets" / "app_icon.png"
    if local_icon.exists():
        return local_icon
    return None


class ConvertWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, image_path: Path, output_path: Path, config_path: Path | None) -> None:
        super().__init__()
        self.image_path = image_path
        self.output_path = output_path
        self.config_path = config_path

    def run(self) -> None:  # noqa: D401
        try:
            cfg = load_config(self.config_path)
            result = run_pipeline(
                image_path=self.image_path,
                output_path=self.output_path,
                cfg=cfg,
                debug_dir=None,
            )
            self.done.emit(str(result.output_path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BEhop RU-LineArt")
        self.resize(760, 520)
        self.setAcceptDrops(True)

        icon_path = default_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.worker: ConvertWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        brand = QLabel("BEhop")
        brand.setStyleSheet("font-size: 24px; font-weight: 700; color: #213547;")
        layout.addWidget(brand)

        product = QLabel("RU-LineArt")
        product.setStyleSheet("font-size: 30px; font-weight: 800; color: #0b5a65;")
        layout.addWidget(product)

        top_hint = QLabel(
            bi(
                "将图片拖拽到窗口，或点击“选择图片”按钮。",
                "Drag an image into this window, or click the browse button.",
            )
        )
        top_hint.setStyleSheet("font-size: 14px;")
        layout.addWidget(top_hint)

        file_group = QGroupBox(bi("文件", "Files"))
        grid = QGridLayout(file_group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText(bi("输入图片路径", "Input image path"))
        self.image_btn = QPushButton(bi("选择图片", "Browse"))
        self.image_btn.clicked.connect(self._pick_image)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(bi("输出 .3dm 路径", "Output .3dm path"))
        self.output_btn = QPushButton(bi("输出路径", "Save As"))
        self.output_btn.clicked.connect(self._pick_output)

        self.config_edit = QLineEdit()
        cfg = default_config_path()
        if cfg is not None:
            self.config_edit.setText(str(cfg))
        self.config_edit.setPlaceholderText(bi("可选：配置 yaml", "Optional: config yaml"))
        self.config_btn = QPushButton(bi("配置文件", "Config"))
        self.config_btn.clicked.connect(self._pick_config)

        grid.addWidget(QLabel(bi("图片", "Image")), 0, 0)
        grid.addWidget(self.image_edit, 0, 1)
        grid.addWidget(self.image_btn, 0, 2)

        grid.addWidget(QLabel(bi("输出", "Output")), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(self.output_btn, 1, 2)

        grid.addWidget(QLabel(bi("配置", "Config")), 2, 0)
        grid.addWidget(self.config_edit, 2, 1)
        grid.addWidget(self.config_btn, 2, 2)

        layout.addWidget(file_group)

        actions = QHBoxLayout()
        self.generate_btn = QPushButton(bi("生成 .3dm", "Generate .3dm"))
        self.generate_btn.clicked.connect(self._generate)
        self.clear_btn = QPushButton(bi("清空日志", "Clear Log"))
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        actions.addWidget(self.generate_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText(bi("运行日志", "Logs"))
        layout.addWidget(self.log_edit, 1)

        self.setCentralWidget(root)

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            bi("选择图片", "Choose image"),
            "",
            "Images 图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        self._set_image_path(Path(path))

    def _pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, bi("保存 3dm", "Save 3dm"), "", "Rhino (*.3dm)")
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".3dm":
            output = output.with_suffix(".3dm")
        self.output_edit.setText(str(output))

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, bi("选择配置 yaml", "Choose config yaml"), "", "YAML (*.yaml *.yml)")
        if path:
            self.config_edit.setText(path)

    def _set_image_path(self, image_path: Path) -> None:
        self.image_edit.setText(str(image_path))
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(image_path.with_suffix(".3dm")))
        self._log(f"{bi('已选择图片', 'Image selected')}: {image_path}")

    def _generate(self) -> None:
        image_text = self.image_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not image_text or not output_text:
            QMessageBox.warning(self, bi("输入不完整", "Incomplete input"), bi("请填写图片和输出路径。", "Please provide image and output path."))
            return

        try:
            image = Path(image_text)
            output = Path(output_text)
            cfg_text = self.config_edit.text().strip()
            config = Path(cfg_text) if cfg_text else None
        except Exception:
            QMessageBox.critical(self, bi("输入无效", "Invalid input"), bi("路径格式无效。", "Path is invalid."))
            return

        if not image.exists() or not image.is_file():
            QMessageBox.warning(self, bi("图片无效", "Image missing"), bi("请选择有效的图片文件。", "Please choose a valid image file."))
            return
        if not is_image_file(image):
            QMessageBox.warning(
                self,
                bi("不支持的图片", "Unsupported image"),
                f"{bi('允许格式', 'Allowed')}: {', '.join(sorted(IMAGE_SUFFIXES))}",
            )
            return
        if output.suffix.lower() != ".3dm":
            output = output.with_suffix(".3dm")
            self.output_edit.setText(str(output))
        output.parent.mkdir(parents=True, exist_ok=True)

        if config is not None and not config.exists():
            QMessageBox.warning(self, bi("配置文件不存在", "Config missing"), bi("配置文件不存在。", "Config file does not exist."))
            return

        self.generate_btn.setEnabled(False)
        self._log(bi("正在转换...", "Converting..."))

        self.worker = ConvertWorker(image_path=image, output_path=output, config_path=config)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self.generate_btn.setEnabled(True))
        self.worker.start()

    def _on_done(self, output_path: str) -> None:
        self._log(f"{bi('完成', 'Done')}: {output_path}")
        QMessageBox.information(
            self,
            bi("转换成功", "Success"),
            f"{bi('已生成文件', 'Generated file')}:\n{output_path}",
        )

    def _on_failed(self, error_text: str) -> None:
        self._log(f"{bi('转换失败', 'Failed')}:\n{error_text}")
        QMessageBox.critical(
            self,
            bi("转换失败", "Failed"),
            bi("转换失败，请查看日志详情。", "Conversion failed. See logs for details."),
        )

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_file() and is_image_file(p):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_file() and is_image_file(p):
                self._set_image_path(p)
                event.acceptProposedAction()
                return
        event.ignore()


def main() -> int:
    app = QApplication(sys.argv)
    icon_path = default_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
