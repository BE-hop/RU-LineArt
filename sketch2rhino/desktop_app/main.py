from __future__ import annotations

import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
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

from sketch2rhino import __version__  # noqa: E402
from sketch2rhino.config import load_config  # noqa: E402
from sketch2rhino.pipeline import run_pipeline  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_UPDATE_FEED_URL = "https://www.behop.cn/behop-ai-product/products/ru-lineart/version.json"
DEFAULT_UPDATE_PAGE_URL = "https://www.behop.cn/behop-ai-product/products/ru-lineart/"
UPDATE_CHECK_TIMEOUT_SEC = 3.0


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


@dataclass(frozen=True)
class UpdateInfo:
    latest: str
    page: str
    force: bool
    notes: str


def update_feed_url() -> str:
    custom = os.environ.get("RU_LINEART_UPDATE_JSON_URL", "").strip()
    return custom or DEFAULT_UPDATE_FEED_URL


def _version_parts(raw: str) -> tuple[int, ...]:
    nums = [int(v) for v in re.findall(r"\d+", raw)]
    return tuple(nums)


def is_newer_version(candidate: str, current: str) -> bool:
    c1 = list(_version_parts(candidate))
    c2 = list(_version_parts(current))
    if not c1 or not c2:
        return False
    size = max(len(c1), len(c2))
    c1.extend([0] * (size - len(c1)))
    c2.extend([0] * (size - len(c2)))
    return tuple(c1) > tuple(c2)


def _parse_update_info(data: dict[str, object]) -> UpdateInfo | None:
    latest = str(data.get("latest", "")).strip()
    if not latest:
        return None

    page = str(data.get("page", "")).strip() or DEFAULT_UPDATE_PAGE_URL
    force_raw = data.get("force", False)
    if isinstance(force_raw, bool):
        force = force_raw
    elif isinstance(force_raw, (int, float)):
        force = bool(force_raw)
    elif isinstance(force_raw, str):
        force = force_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        force = False

    notes = str(data.get("notes", "")).strip()
    if not notes:
        zh = str(data.get("notes_zh", "")).strip()
        en = str(data.get("notes_en", "")).strip()
        if zh or en:
            notes = bi(zh or "-", en or "-")

    return UpdateInfo(latest=latest, page=page, force=force, notes=notes)


def fetch_update_info(feed_url: str) -> UpdateInfo | None:
    request = urllib.request.Request(
        feed_url,
        headers={"User-Agent": f"RU-LineArt/{__version__}"},
    )
    with urllib.request.urlopen(request, timeout=UPDATE_CHECK_TIMEOUT_SEC) as resp:
        payload = resp.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        return None
    return _parse_update_info(parsed)


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


class UpdateCheckWorker(QThread):
    found = Signal(object)

    def __init__(self, current_version: str, feed_url: str) -> None:
        super().__init__()
        self.current_version = current_version
        self.feed_url = feed_url

    def run(self) -> None:  # noqa: D401
        try:
            info = fetch_update_info(self.feed_url)
            if info is None:
                return
            if is_newer_version(info.latest, self.current_version):
                self.found.emit(info)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            # Network or JSON errors should not block app startup.
            return


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
        self.update_worker: UpdateCheckWorker | None = None
        self.current_version = __version__
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

    def check_update_on_startup(self) -> None:
        feed_url = update_feed_url()
        self.update_worker = UpdateCheckWorker(current_version=self.current_version, feed_url=feed_url)
        self.update_worker.found.connect(self._on_update_found)
        self.update_worker.start()
        self._log(
            f"{bi('当前版本', 'Current version')}: {self.current_version} | "
            f"{bi('更新源', 'Update feed')}: {feed_url}"
        )

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

    def _on_update_found(self, payload: object) -> None:
        if not isinstance(payload, UpdateInfo):
            return

        if payload.force:
            self._show_force_update_dialog(payload)
        else:
            self._show_optional_update_dialog(payload)

    def _show_optional_update_dialog(self, info: UpdateInfo) -> None:
        text = (
            f"{bi('发现新版本', 'New version found')}: {info.latest}\n"
            f"{bi('当前版本', 'Current version')}: {self.current_version}"
        )
        if info.notes:
            text = f"{text}\n\n{bi('更新说明', 'Release notes')}:\n{info.notes}"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(bi("发现更新", "Update available"))
        box.setText(text)
        open_btn = box.addButton(bi("前往官网更新", "Open update page"), QMessageBox.AcceptRole)
        box.addButton(bi("稍后", "Later"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == open_btn:
            self._open_update_page(info.page)

    def _show_force_update_dialog(self, info: UpdateInfo) -> None:
        text = (
            f"{bi('发现必须更新版本', 'A required update is available')}: {info.latest}\n"
            f"{bi('当前版本', 'Current version')}: {self.current_version}\n\n"
            f"{bi('请先更新后再继续使用。', 'Please update before continuing.')}"
        )
        if info.notes:
            text = f"{text}\n\n{bi('更新说明', 'Release notes')}:\n{info.notes}"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(bi("必须更新", "Update required"))
        box.setText(text)
        open_btn = box.addButton(bi("前往官网更新", "Open update page"), QMessageBox.AcceptRole)
        box.addButton(bi("退出软件", "Quit"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == open_btn:
            self._open_update_page(info.page)
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _open_update_page(self, page_url: str) -> None:
        if QDesktopServices.openUrl(QUrl(page_url)):
            self._log(f"{bi('已打开更新页面', 'Opened update page')}: {page_url}")
            return
        QMessageBox.warning(
            self,
            bi("打开失败", "Open failed"),
            f"{bi('无法打开更新页面', 'Could not open update page')}: {page_url}",
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
    window.check_update_on_startup()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
