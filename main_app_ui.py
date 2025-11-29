import os
import sys
from typing import List, Tuple, Any, Dict

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

from sentence_transformers import SentenceTransformer
import faiss

from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets


# ===================== 1. 基础配置 =====================

BASE_MODEL_PATH = "./qwen3-vl-4b-instruct"      # 你的基础模型路径
LORA_CHECKPOINT = "output/Qwen3-VL-4Blora/checkpoint-310"
RAG_DIR = "rag_data"
USE_4BIT = True

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
APP_ICON_PATH = "./image/logo.png"                      # 图标文件，和本脚本同目录


# ===================== 2. 加载模型 =====================

def load_model():
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        quant_cfg = dict(quantization_config=bnb_config)
    else:
        quant_cfg = {}

    print("[Model] Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,** quant_cfg,
    )
    print("[Model] Loading LoRA...")
    model = PeftModel.from_pretrained(model, LORA_CHECKPOINT)
    model = model.to(model.device)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )
    print("[Model] Loaded.")
    return model, processor


model, processor = load_model()
DEVICE = model.device


# ===================== 3. RAG 构建 =====================

def load_corpus(rag_dir: str) -> Tuple[List[str], List[str]]:
    docs, meta = [], []
    if not os.path.isdir(rag_dir):
        print(f"[RAG] directory {rag_dir} not found.")
        return docs, meta
    for fname in os.listdir(rag_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(rag_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append(text)
                meta.append(fname)
        except Exception as e:
            print(f"[RAG] read {path} fail: {e}")
    print(f"[RAG] loaded {len(docs)} docs.")
    return docs, meta


corpus_docs, corpus_meta = load_corpus(RAG_DIR)
if corpus_docs:
    embed_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    embeddings = embed_model.encode(corpus_docs, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
else:
    embed_model = None
    index = None


def search_docs(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    if index is None or embed_model is None or not query.strip():
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((corpus_docs[idx], corpus_meta[idx], float(score)))
    return results


# ===================== 4. 推理逻辑 =====================

def build_messages(user_query: str,
                   pil_image: Image.Image,
                   rag_results: List[Tuple[str, str, float]]) -> List[Dict[str, Any]]:
    context_text = ""
    if rag_results:
        parts = []
        for doc, fname, score in rag_results:
            parts.append(f"【来自文档 {fname} 的参考内容（相似度 {score:.2f}）】\n{doc}")
        context_text = "\n\n".join(parts)

    system_prompt = (
        "你是“xiao taixian 生活助手”，擅长提供生活、学习和职场效率相关的建议。"
        "回答要逻辑清晰、结构化、简洁友好。"
    )
    if context_text:
        system_prompt += (
            "\n\n以下是与用户问题相关的参考资料，请理解后综合回答，不要逐字照搬：\n"
            f"{context_text}"
        )

    content_list = []
    if pil_image is not None:
        content_list.append({"type": "image", "image": pil_image})
    content_list.append({"type": "text", "text": user_query})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content_list},
    ]
    return messages


def generate_answer(user_query: str,
                    pil_image: Image.Image,
                    use_rag: bool = True,
                    max_new_tokens: int = 256,
                    temperature: float = 0.6) -> Tuple[str, str]:
    rag_results = search_docs(user_query) if use_rag else []
    if rag_results:
        rag_info = "\n\n".join(
            [f"【{fname} | 相似度 {score:.2f}】\n{doc[:400]}..."
             for doc, fname, score in rag_results]
        )
    else:
        rag_info = "未使用或未检索到相关文档。"

    messages = build_messages(user_query, pil_image, rag_results)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    gen_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
    ]
    output_text = processor.batch_decode(
        gen_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text, rag_info


# ===================== 5. UI 线程封装 =====================

class Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str)  # (answer, rag_info)

    def __init__(self, query: str, img_path: str,
                 use_rag: bool, max_tokens: int, temperature: float):
        super().__init__()
        self.query = query
        self.img_path = img_path
        self.use_rag = use_rag
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
        pil_img = None
        if self.img_path:
            try:
                pil_img = Image.open(self.img_path).convert("RGB")
            except Exception as e:
                print(f"[Image] open fail: {e}")
                pil_img = None
        answer, rag_info = generate_answer(
            user_query=self.query,
            pil_image=pil_img,
            use_rag=self.use_rag,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.finished.emit(answer, rag_info)


# ===================== 6. 自定义聊天气泡控件 =====================

class ChatBubble(QtWidgets.QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        # 关键：让气泡在水平方向扩展、垂直方向按内容自适应
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.setSizePolicy(sp)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # 用户名标签（简洁小字号）
        name = QtWidgets.QLabel("你" if is_user else "xiao taixian")
        name.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: %s;" %
            ("#4b5563" if is_user else "#2563eb")
        )
        layout.addWidget(name)

        # 消息内容
        self.bubble_label = QtWidgets.QLabel(text)
        self.bubble_label.setWordWrap(True)
        self.bubble_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        # 简洁气泡样式：无阴影、淡色背景
        self.bubble_label.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 10px; padding: 8px 12px; "
            "font-size: 13px; color: #111827; }" %
            ("#e0e7ff" if is_user else "#dbeafe")  # 用户 / 助手不同颜色
        )
        layout.addWidget(self.bubble_label)

        # 对齐方式
        align = QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft
        layout.setAlignment(align)

        # 关键：文本设置完后手动调整大小，保证 Layout 计算到正确高度
        self.bubble_label.adjustSize()
        self.adjustSize()

    def update_text(self, text: str):
        """如果以后做流式输出，可以用这个方法动态更新内容."""
        self.bubble_label.setText(text)
        self.bubble_label.adjustSize()
        self.adjustSize()



class ChatArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container = QtWidgets.QWidget()
        self.v_layout = QtWidgets.QVBoxLayout(self.container)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        self.v_layout.setSpacing(12)
        # 可以保留一个 stretch，使内容向上靠，但容易影响高度，这里去掉
        # self.v_layout.addStretch(1)

        self.setWidget(self.container)

    def add_message(self, text: str, is_user: bool) -> ChatBubble:
        bubble = ChatBubble(text, is_user)
        self.v_layout.addWidget(bubble)

        # 关键：强制重新布局，保证所有气泡按内容撑开高度
        self.v_layout.activate()
        self.container.adjustSize()
        bubble.adjustSize()

        QtCore.QTimer.singleShot(0, self.scroll_to_bottom)
        return bubble

    def scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

# ===================== 7. 主窗口 =====================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("xiao taixian")
        self.resize(1200, 800)  # 增大窗口默认尺寸
        self.setMinimumSize(1000, 700)

        # 设置窗口图标
        if os.path.exists(APP_ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))

        # 全局样式（优化版，移除不支持的CSS属性）
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #f8fafc; 
            }
            QLineEdit {
                background-color: #ffffff;
                border-radius: 20px;
                border: 1px solid #d1d5db;
                padding: 10px 16px;
                font-size: 15px;
                selection-background-color: #dbeafe;
            }
            QLineEdit:focus {
                border: 1px solid #2563eb;
                outline: none;
            }
            QPushButton {
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: 500;
                background-color: #2563eb;
                color: white;
            }
            QPushButton:hover { 
                background-color: #1d4ed8;
            }
            QPushButton:pressed { 
                background-color: #1e40af;
            }
            QPushButton:disabled { 
                background-color: #9ca3af;
            }
            QGroupBox {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                margin-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0px 6px;
                font-size: 14px;
                font-weight: 600;
                color: #1e293b;
            }
            QTextEdit {
                background-color: #f9fafb;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                padding: 10px;
                font-size: 14px;
                line-height: 1.5;
            }
            QLabel { 
                color: #334155; 
            }
            QSplitter::handle {
                background-color: #e2e8f0;
                width: 4px;
                margin: 0 2px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #cbd5e1;
            }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 顶部标题 + logo
        header = QtWidgets.QFrame()
        header.setFrameShape(QtWidgets.QFrame.NoFrame)
        header.setStyleSheet("""
            QFrame { 
                background-color: #ffffff; 
                border-radius: 16px; 
                border: 1px solid #e2e8f0;
            }
        """)
        # 给标题栏添加阴影
        header_shadow = QtWidgets.QGraphicsDropShadowEffect(
            blurRadius=8, xOffset=0, yOffset=2, color=QtGui.QColor(0, 0, 0, 20)
        )
        header.setGraphicsEffect(header_shadow)
        
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(20, 16, 20, 16)
        h_layout.setSpacing(12)

        # logo
        logo_label = QtWidgets.QLabel()
        logo_label.setFixedSize(40, 40)
        if os.path.exists(APP_ICON_PATH):
            pix = QtGui.QPixmap(APP_ICON_PATH)
            logo_label.setPixmap(pix.scaled(40, 40, QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.SmoothTransformation))
        h_layout.addWidget(logo_label)

        title_box = QtWidgets.QVBoxLayout()
        title_label = QtWidgets.QLabel("xiao taixian 生活助手")
        title_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #1e293b;")
        subtitle_label = QtWidgets.QLabel(
            "基于本地多模态大模型 + RAG 的生活/学习/职场效率助手 · 支持文本和图片输入。"
        )
        subtitle_label.setStyleSheet("font-size: 13px; color: #64748b;")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        h_layout.addLayout(title_box, 1)

        main_layout.addWidget(header)

        # 中间区域：左聊天，右 RAG + 设置
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setStyleSheet("QSplitter { background-color: transparent; }")
        main_layout.addWidget(split, 1)

        # 左侧区域
        left_card = QtWidgets.QFrame()
        left_card.setStyleSheet("""
            QFrame { 
                background-color: #ffffff; 
                border-radius: 16px; 
                border: 1px solid #e2e8f0;
            }
        """)
        # 给左侧卡片添加阴影
        left_shadow = QtWidgets.QGraphicsDropShadowEffect(
            blurRadius=8, xOffset=0, yOffset=2, color=QtGui.QColor(0, 0, 0, 20)
        )
        left_card.setGraphicsEffect(left_shadow)
        
        left_layout = QtWidgets.QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        self.chat_area = ChatArea()
        left_layout.addWidget(self.chat_area, 1)

        # 输入行
        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText(
            "输入你的问题，例如：帮我规划一周学习计划 / 看看这张桌面布置合不合理？"
        )
        self.send_btn = QtWidgets.QPushButton("发送")
        self.send_btn.setFixedWidth(100)

        input_row.addWidget(self.input_edit, 1)
        input_row.addSpacing(10)
        input_row.addWidget(self.send_btn)
        left_layout.addLayout(input_row)

        # 图片预览行
        preview_row = QtWidgets.QHBoxLayout()
        self.preview_frame = QtWidgets.QFrame()
        pf_layout = QtWidgets.QHBoxLayout(self.preview_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setSpacing(10)

        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setFixedSize(70, 70)
        self.image_preview.setStyleSheet(
            "QLabel { background-color: #f1f5f9; border-radius: 8px; border: 1px dashed #cbd5e1; }"
        )
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setText("无图片")

        self.image_name_label = QtWidgets.QLabel("未选择图片")
        self.image_name_label.setStyleSheet("font-size: 13px; color: #64748b;")
        self.image_name_label.setWordWrap(True)

        pf_layout.addWidget(self.image_preview)
        pf_layout.addWidget(self.image_name_label, 1)

        preview_row.addWidget(self.preview_frame, 1)

        self.choose_img_btn = QtWidgets.QPushButton("选择图片")
        self.clear_img_btn = QtWidgets.QPushButton("清除")
        self.clear_img_btn.setStyleSheet(
            "QPushButton { background-color: #64748b; color: white; }"
            "QPushButton:hover { background-color: #475569; }"
            "QPushButton:pressed { background-color: #334155; }"
        )
        preview_row.addWidget(self.choose_img_btn)
        preview_row.addSpacing(6)
        preview_row.addWidget(self.clear_img_btn)
        left_layout.addLayout(preview_row)

        split.addWidget(left_card)

        # 右侧区域
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(16)

        # RAG 结果卡片
        rag_group = QtWidgets.QGroupBox("RAG 检索到的参考内容（前 400 字）")
        # 给RAG卡片添加阴影
        rag_shadow = QtWidgets.QGraphicsDropShadowEffect(
            blurRadius=8, xOffset=0, yOffset=2, color=QtGui.QColor(0, 0, 0, 20)
        )
        rag_group.setGraphicsEffect(rag_shadow)
        
        rag_layout = QtWidgets.QVBoxLayout(rag_group)
        rag_layout.setContentsMargins(14, 12, 14, 14)
        self.rag_text = QtWidgets.QTextEdit()
        self.rag_text.setReadOnly(True)
        self.rag_text.setStyleSheet(
            "QTextEdit { font-family: 'Microsoft YaHei', 'Consolas','Menlo','Monaco',monospace; "
            "font-size: 13px; }"
        )
        rag_layout.addWidget(self.rag_text)
        right_layout.addWidget(rag_group, 3)

        # 设置卡片
        param_group = QtWidgets.QGroupBox("生成参数 · 使用说明")
        # 给参数卡片添加阴影
        param_shadow = QtWidgets.QGraphicsDropShadowEffect(
            blurRadius=8, xOffset=0, yOffset=2, color=QtGui.QColor(0, 0, 0, 20)
        )
        param_group.setGraphicsEffect(param_shadow)
        
        param_layout = QtWidgets.QFormLayout(param_group)
        param_layout.setContentsMargins(14, 12, 14, 14)
        param_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        param_layout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        param_layout.setSpacing(12)  # 修正：只传一个参数

        self.use_rag_cb = QtWidgets.QCheckBox("启用 RAG 文档检索")
        self.use_rag_cb.setChecked(True)
        self.use_rag_cb.setStyleSheet("font-size: 14px;")
        param_layout.addRow(self.use_rag_cb)

        # 最大长度设置
        max_tokens_label = QtWidgets.QLabel("最大生成长度")
        max_tokens_label.setStyleSheet("font-size: 14px;")
        self.max_tokens_spin = QtWidgets.QSpinBox()
        self.max_tokens_spin.setRange(32, 1024)
        self.max_tokens_spin.setSingleStep(32)
        self.max_tokens_spin.setValue(256)
        self.max_tokens_spin.setStyleSheet("font-size: 14px; padding: 4px;")
        param_layout.addRow(max_tokens_label, self.max_tokens_spin)

        # 温度设置
        temp_label = QtWidgets.QLabel("Temperature")
        temp_label.setStyleSheet("font-size: 14px;")
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 1.2)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(0.60)
        self.temp_spin.setStyleSheet("font-size: 14px; padding: 4px;")
        param_layout.addRow(temp_label, self.temp_spin)

        # 使用提示
        hint_label = QtWidgets.QLabel(
            "- 将生活/学习/工作笔记以 .txt 放入 rag_data/ 目录即可扩展知识库。\n"
            "- 推荐问题示例：\n"
            "  · 帮我设计适合上班族的一周运动计划；\n"
            "  · 帮我准备 XX 技术面试的复习大纲；\n"
            "  · 看看这张桌面照片，有什么提高专注度的建议？"
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("font-size: 13px; color: #64748b; margin-top: 8px;")
        param_layout.addRow(hint_label)
        right_layout.addWidget(param_group, 2)

        split.addWidget(right_widget)
        split.setStretchFactor(0, 3)  # 左侧聊天区域占比更大
        split.setStretchFactor(1, 2)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("font-size: 13px; padding: 6px;")
        self.status_bar.showMessage("模型已加载完成。")

        # 成员变量
        self.current_image_path: str = ""

        # 事件绑定
        self.send_btn.clicked.connect(self.on_send)
        self.input_edit.returnPressed.connect(self.on_send)
        self.choose_img_btn.clicked.connect(self.on_choose_image)
        self.clear_img_btn.clicked.connect(self.on_clear_image)
        self.image_preview.mousePressEvent = self.on_preview_clicked  # 点击预览放大

        self.worker: Worker | None = None

    # ---------- 事件 ----------

    def on_choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self.current_image_path = path
        name = os.path.basename(path)
        self.image_name_label.setText(name)

        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            self.image_preview.setPixmap(
                pix.scaled(70, 70, QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)
            )
        else:
            self.image_preview.setText("预览失败")

    def on_clear_image(self):
        self.current_image_path = ""
        self.image_name_label.setText("未选择图片")
        self.image_preview.setPixmap(QtGui.QPixmap())
        self.image_preview.setText("无图片")

    def on_preview_clicked(self, event):
        # 如果有图片，弹出大图预览
        if not self.current_image_path:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("图片预览")
        dlg.resize(800, 600)
        if os.path.exists(APP_ICON_PATH):
            dlg.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
        # 设置对话框样式
        dlg.setStyleSheet("""
            QDialog { background-color: #f8fafc; }
        """)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(20, 20, 20, 20)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        pix = QtGui.QPixmap(self.current_image_path)
        if not pix.isNull():
            # 自适应窗口大小
            label.setPixmap(
                pix.scaled(
                    dlg.size() * 0.95, 
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
            )
        layout.addWidget(label)
        dlg.exec_()

    def on_send(self):
        query = self.input_edit.text().strip()
        if not query and not self.current_image_path:
            QtWidgets.QMessageBox.information(self, "提示", "请输入问题，或选择一张图片。")
            return

        self.chat_area.add_message(query if query else "(仅图片)", is_user=True)
        self.input_edit.clear()

        self.send_btn.setEnabled(False)
        self.status_bar.showMessage("正在生成回答，请稍候...")

        use_rag = self.use_rag_cb.isChecked()
        max_tokens = self.max_tokens_spin.value()
        temperature = float(self.temp_spin.value())

        self.worker = Worker(query, self.current_image_path, use_rag, max_tokens, temperature)
        self.worker.finished.connect(self.on_answer_ready)
        self.worker.start()

    def on_answer_ready(self, answer: str, rag_info: str):
        self.chat_area.add_message(answer, is_user=False)
        self.rag_text.setPlainText(rag_info)
        self.send_btn.setEnabled(True)
        self.status_bar.showMessage("就绪。")


def main():
    app = QtWidgets.QApplication(sys.argv)
    # 全局应用图标
    if os.path.exists(APP_ICON_PATH):
        app.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
    # 设置全局字体
    font = QtGui.QFont("Microsoft YaHei", 10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()