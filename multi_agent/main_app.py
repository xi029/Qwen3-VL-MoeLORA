"""PyQt5 图形界面：接入多智能体 + RAG 流水线。"""

import os
import sys
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from multi_agent import MultiAgentOrchestrator
from rag_pipeline import format_documents

APP_ICON_PATH = "logo.png"
KNOWLEDGE_DIR = "./knowledge_base"


class Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str, str)  # plan, answer, rag_text
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        orchestrator: MultiAgentOrchestrator,
        question: str,
        image_path: Optional[str],
        use_rag: bool,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._question = question
        self._image_path = image_path
        self._use_rag = use_rag

    def run(self) -> None:
        try:
            images = [self._image_path] if self._image_path else None
            result = self._orchestrator.run(
                question=self._question,
                image_paths=images,
                use_knowledge=self._use_rag,
            )
            rag_text = format_documents(result.supporting_documents)
            if not rag_text:
                rag_text = "未检索到相关文档或未启用 RAG。"
            self.finished.emit(result.plan, result.answer, rag_text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ChatBubble(QtWidgets.QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        name = QtWidgets.QLabel("你" if is_user else "Multi‑Agent")
        name.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: %s;"
            % ("#2563eb" if not is_user else "#6366f1")
        )
        layout.addWidget(name)

        bubble = QtWidgets.QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bubble.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 12px; padding: 12px; "
            "font-size: 15px; line-height: 1.5; color: #111827; }"
            % ("#dbeafe" if not is_user else "#e0e7ff")
        )
        layout.addWidget(bubble)
        layout.setAlignment(QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft)


class ChatArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.container = QtWidgets.QWidget()
        self.v_layout = QtWidgets.QVBoxLayout(self.container)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        self.v_layout.setSpacing(16)
        self.v_layout.addStretch(1)
        self.setWidget(self.container)

    def add_message(self, text: str, is_user: bool) -> None:
        bubble = ChatBubble(text, is_user)
        self.v_layout.insertWidget(self.v_layout.count() - 1, bubble)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, orchestrator: MultiAgentOrchestrator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._worker: Optional[Worker] = None
        self._image_path: Optional[str] = None

        self.setWindowTitle("多智能体 RAG 助手")
        self.resize(1150, 720)
        self.setMinimumSize(960, 620)
        if os.path.exists(APP_ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setFrameShape(QtWidgets.QFrame.NoFrame)
        header.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 14px; border: 1px solid #e5e7eb; }"
        )
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(16, 10, 16, 10)
        logo = QtWidgets.QLabel()
        logo.setFixedSize(36, 36)
        if os.path.exists(APP_ICON_PATH):
            pix = QtGui.QPixmap(APP_ICON_PATH)
            logo.setPixmap(
                pix.scaled(36, 36, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )
        h_layout.addWidget(logo)
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Multi‑Agent 小苔藓")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #111827;")
        subtitle = QtWidgets.QLabel("基于 LangChain + 多模态大模型 + RAG 的多智能体助手 · 支持文本和图片输入")
        subtitle.setStyleSheet("font-size: 11px; color: #6b7280;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        h_layout.addLayout(title_box, 1)
        main_layout.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_card = QtWidgets.QFrame()
        left_card.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 14px; border: 1px solid #e5e7eb; }"
        )
        left_layout = QtWidgets.QVBoxLayout(left_card)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        self.chat_area = ChatArea()
        left_layout.addWidget(self.chat_area, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("输入你的问题...")
        self.send_btn = QtWidgets.QPushButton("发送")
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.send_btn)
        left_layout.addLayout(input_row)

        preview_row = QtWidgets.QHBoxLayout()
        self.image_preview = QtWidgets.QLabel("无图片")
        self.image_preview.setFixedSize(70, 70)
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setStyleSheet(
            "QLabel { background-color: #f9fafb; border-radius: 8px; border: 1px dashed #d1d5db; }"
        )
        self.image_name_label = QtWidgets.QLabel("未选择图片")
        preview_row.addWidget(self.image_preview)
        preview_row.addWidget(self.image_name_label, 1)
        self.choose_img_btn = QtWidgets.QPushButton("选择图片")
        self.clear_img_btn = QtWidgets.QPushButton("清除")
        preview_row.addWidget(self.choose_img_btn)
        preview_row.addWidget(self.clear_img_btn)
        left_layout.addLayout(preview_row)
        splitter.addWidget(left_card)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(10)

        plan_group = QtWidgets.QGroupBox("调度 + 任务计划")
        plan_layout = QtWidgets.QVBoxLayout(plan_group)
        self.plan_text = QtWidgets.QTextEdit()
        self.plan_text.setReadOnly(True)
        plan_layout.addWidget(self.plan_text)
        right_layout.addWidget(plan_group, 2)

        rag_group = QtWidgets.QGroupBox("RAG 检索参考")
        rag_layout = QtWidgets.QVBoxLayout(rag_group)
        self.rag_text = QtWidgets.QTextEdit()
        self.rag_text.setReadOnly(True)
        rag_layout.addWidget(self.rag_text)
        right_layout.addWidget(rag_group, 3)

        option_group = QtWidgets.QGroupBox("选项")
        option_layout = QtWidgets.QVBoxLayout(option_group)
        self.use_rag_cb = QtWidgets.QCheckBox("启用知识库检索 (RAG)")
        self.use_rag_cb.setChecked(True)
        option_layout.addWidget(self.use_rag_cb)
        right_layout.addWidget(option_group)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.statusBar().showMessage("多智能体已就绪。")

        self.send_btn.clicked.connect(self._on_send)
        self.input_edit.returnPressed.connect(self._on_send)
        self.choose_img_btn.clicked.connect(self._on_choose_image)
        self.clear_img_btn.clicked.connect(self._on_clear_image)
        self.image_preview.mousePressEvent = self._preview_image

    # ----- UI 事件 -----

    def _on_choose_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self._image_path = path
        self.image_name_label.setText(os.path.basename(path))
        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            self.image_preview.setPixmap(
                pix.scaled(70, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def _on_clear_image(self) -> None:
        self._image_path = None
        self.image_name_label.setText("未选择图片")
        self.image_preview.setPixmap(QtGui.QPixmap())
        self.image_preview.setText("无图片")

    def _preview_image(self, event) -> None:  # noqa: ANN001
        if not self._image_path:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("图片预览")
        dlg.resize(640, 480)
        layout = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        pix = QtGui.QPixmap(self._image_path)
        if not pix.isNull():
            label.setPixmap(
                pix.scaled(dlg.size() * 0.95, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )
        layout.addWidget(label)
        dlg.exec_()

    def _on_send(self) -> None:
        query = self.input_edit.text().strip()
        if not query and not self._image_path:
            return

        self.chat_area.add_message(query or "(仅图片)", is_user=True)
        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self.statusBar().showMessage("多智能体正在思考...")

        self._worker = Worker(
            orchestrator=self._orchestrator,
            question=query or "请描述这张图片",
            image_path=self._image_path,
            use_rag=self.use_rag_cb.isChecked(),
        )
        self._worker.finished.connect(self._on_answer_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_answer_ready(self, plan: str, answer: str, rag_text: str) -> None:
        self.chat_area.add_message(answer, is_user=False)
        self.plan_text.setPlainText(plan)
        self.rag_text.setPlainText(rag_text)
        self.send_btn.setEnabled(True)
        self.statusBar().showMessage("就绪。")
        self._worker = None

    def _on_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "推理失败", message)
        self.send_btn.setEnabled(True)
        self.statusBar().showMessage("发生错误，请重试。")
        self._worker = None


def run_app() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    orchestrator = MultiAgentOrchestrator(knowledge_dir=KNOWLEDGE_DIR)
    window = MainWindow(orchestrator)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
