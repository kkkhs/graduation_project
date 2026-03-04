from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
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

from src.application.bootstrap import build_predict_service
from src.infrastructure.visualization import render_detections

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "models.yaml"
OUT_VIS_DIR = BASE_DIR / "outputs" / "ui_visualizations"
OUT_JSON_DIR = BASE_DIR / "outputs" / "predictions"

OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("遥感微小船舶检测系统 - Python Desktop UI (Qt)")
        self.resize(1220, 820)

        if not CONFIG_PATH.exists():
            raise FileNotFoundError("configs/models.yaml not found. Copy from configs/models.example.yaml first.")

        self.service = build_predict_service(str(CONFIG_PATH))
        self.models = self.service.available_models()
        if not self.models:
            raise ValueError("No models found in configs/models.yaml")
        self.mode_options = ["ensemble_all"] + self.models

        self.image_path: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("图片:"))
        self.image_input = QLineEdit()
        row1.addWidget(self.image_input, stretch=1)
        btn_choose = QPushButton("选择图片")
        btn_choose.clicked.connect(self.choose_image)
        row1.addWidget(btn_choose)
        main.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.mode_options)
        row2.addWidget(self.model_combo)

        row2.addWidget(QLabel("conf:"))
        self.conf_input = QLineEdit("0.25")
        self.conf_input.setFixedWidth(80)
        row2.addWidget(self.conf_input)

        row2.addWidget(QLabel("iou:"))
        self.iou_input = QLineEdit("0.50")
        self.iou_input.setFixedWidth(80)
        row2.addWidget(self.iou_input)

        btn_run = QPushButton("执行推理")
        btn_run.clicked.connect(self.run_predict)
        row2.addWidget(btn_run)

        row2.addStretch(1)
        main.addLayout(row2)

        row3 = QHBoxLayout()

        left_box = QVBoxLayout()
        left_box.addWidget(QLabel("原图"))
        self.original_view = QLabel("请选择图片")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setStyleSheet("border:1px solid #d0d0d0; min-height:340px;")
        left_box.addWidget(self.original_view)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("检测可视化"))
        self.vis_view = QLabel("等待推理")
        self.vis_view.setAlignment(Qt.AlignCenter)
        self.vis_view.setStyleSheet("border:1px solid #d0d0d0; min-height:340px;")
        right_box.addWidget(self.vis_view)

        row3.addLayout(left_box, stretch=1)
        row3.addLayout(right_box, stretch=1)
        main.addLayout(row3)

        main.addWidget(QLabel("结构化输出（JSON）"))
        self.json_box = QPlainTextEdit()
        self.json_box.setPlainText("[]")
        self.json_box.setMinimumHeight(220)
        main.addWidget(self.json_box)

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(BASE_DIR),
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image_input.setText(path)
        self._display(self.original_view, self.image_path)

    def run_predict(self) -> None:
        path_text = self.image_input.text().strip()
        if not path_text:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return

        image_path = Path(path_text)
        if not image_path.exists():
            QMessageBox.critical(self, "错误", f"图片不存在: {image_path}")
            return

        selected = self.model_combo.currentText().strip()
        try:
            conf = float(self.conf_input.text().strip())
            iou = float(self.iou_input.text().strip())
        except ValueError:
            QMessageBox.critical(self, "错误", "conf / iou 必须是数字")
            return

        if conf < 0 or conf > 1 or iou < 0 or iou > 1:
            QMessageBox.critical(self, "错误", "conf / iou 必须在 [0,1]")
            return

        try:
            if selected == "ensemble_all":
                records = self.service.predict_ensemble(
                    image_path=str(image_path),
                    model_names=self.models,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    fusion_iou_threshold=0.55,
                    min_votes=1,
                )
                result_tag = "ensemble_all"
            else:
                records = self.service.predict(
                    image_path=str(image_path),
                    model_name=selected,
                    conf_threshold=conf,
                    iou_threshold=iou,
                )
                result_tag = selected
        except Exception as exc:
            QMessageBox.critical(self, "推理失败", str(exc))
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = OUT_VIS_DIR / f"vis_{result_tag}_{stamp}.jpg"
        json_path = OUT_JSON_DIR / f"pred_{result_tag}_{stamp}.json"

        render_detections(str(image_path), records, str(vis_path))
        json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        self._display(self.vis_view, vis_path)
        self.json_box.setPlainText(json.dumps(records, ensure_ascii=False, indent=2))

        QMessageBox.information(
            self,
            "完成",
            f"推理完成\n检测数: {len(records)}\n可视化: {vis_path}\nJSON: {json_path}",
        )

    def _display(self, widget: QLabel, image_path: Path) -> None:
        # Ensure image exists and is loadable.
        with Image.open(image_path) as _:
            pass
        pix = QPixmap(str(image_path))
        if pix.isNull():
            widget.setText("图片加载失败")
            return
        scaled = pix.scaled(560, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        widget.setPixmap(scaled)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
