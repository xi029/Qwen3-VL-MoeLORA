import os
import sys
from typing import List, Tuple, Any, Dict
from operator import itemgetter

import torch
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

# ===================== LangChain & Transformer Imports =====================
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

# ===================== 1. 基础配置 =====================

BASE_MODEL_PATH = "./qwen3-vl-4b-instruct"      # 你的基础模型路径
LORA_CHECKPOINT = "output/Qwen3-VL-4Blora/checkpoint-310"
RAG_DIR = "rag_data"
USE_4BIT = True

# 使用 LangChain 兼容的 embedding 模型
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
APP_ICON_PATH = "./image/logo.png"


# ===================== 2. 加载模型 (保持不变) =====================

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
        trust_remote_code=True,
        **quant_cfg,
    )
    print("[Model] Loading LoRA...")
    model = PeftModel.from_pretrained(model, LORA_CHECKPOINT)
    # Merge LoRA weights logic can be added here if needed, but usually not strictly required for inference
    
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )
    print("[Model] Loaded.")
    return model, processor

# 初始化模型
model, processor = load_model()
DEVICE = model.device


# ===================== 3. LangChain RAG 构建 =====================

def build_vectorstore(rag_dir: str):
    """使用 LangChain 构建向量数据库"""
    if not os.path.exists(rag_dir):
        os.makedirs(rag_dir, exist_ok=True)
        print(f"[RAG] Directory {rag_dir} created.")
        return None

    # 1. 加载文档
    print("[RAG] Loading documents via LangChain...")
    loader = DirectoryLoader(rag_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    
    if not docs:
        print("[RAG] No documents found.")
        return None

    # 2. 文本切分 (这是 LangChain 的优势，处理长文档更智能)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"[RAG] Split into {len(splits)} chunks.")

    # 3. 初始化 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # 4. 构建/索引向量库
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("[RAG] VectorStore built.")
    return vectorstore

# 初始化向量库和检索器
vectorstore = build_vectorstore(RAG_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None


# ===================== 4. 自定义 LangChain 推理逻辑 =====================

def format_docs(docs: List[Document]) -> str:
    """将检索到的 LangChain Document 对象格式化为字符串"""
    if not docs:
        return ""
    formatted = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        content = doc.page_content.strip()
        formatted.append(f"【参考资料 {i+1} | 来源: {source}】\n{content}")
    return "\n\n".join(formatted)

def qwen_multimodal_generate(inputs: dict) -> dict:
    """
    自定义的 Runnabel 函数，用于处理多模态输入并调用模型。
    inputs 包含: 'context' (str), 'query' (str), 'image' (PIL.Image or None), 'config' (dict)
    """
    context_text = inputs.get("context", "")
    user_query = inputs.get("query", "")
    pil_image = inputs.get("image", None)
    
    # 从 chain 的 config 中获取生成参数 (如果传入的话)，否则用默认
    gen_config = inputs.get("gen_params", {"max_new_tokens": 256, "temperature": 0.6})

    # 构建 System Prompt
    system_prompt = (
        "你是“xiao taixian 生活助手”，擅长提供生活、学习和职场效率相关的建议。"
        "回答要逻辑清晰、结构化、简洁友好。"
    )
    if context_text:
        system_prompt += (
            "\n\n以下是与用户问题相关的参考资料，请理解后综合回答，不要逐字照搬：\n"
            f"{context_text}"
        )

    # 构建 Content 列表
    content_list = []
    if pil_image is not None:
        content_list.append({"type": "image", "image": pil_image})
    content_list.append({"type": "text", "text": user_query})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content_list},
    ]

    # 模型推理
    text_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    text_inputs = text_inputs.to(DEVICE)

    with torch.no_grad():
        gen_ids = model.generate(
            **text_inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 256),
            temperature=gen_config.get("temperature", 0.6),
            do_sample=True,
        )
    
    # 解码
    gen_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(text_inputs.input_ids, gen_ids)
    ]
    output_text = processor.batch_decode(
        gen_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {"answer": output_text, "context_str": context_text}

# ===================== 5. 构建 LCEL Chain =====================

def get_rag_chain(use_rag: bool):
    """
    构建 LangChain 流水线:
    Inputs -> (Retrieval) -> Combine -> Inference
    """
    
    if use_rag and retriever:
        # 启用 RAG 的分支
        # RunnablePassthrough.assign 用于在不覆盖原始 query/image 的情况下增加 context 字段
        retrieval_step = RunnablePassthrough.assign(
            context=itemgetter("query") | retriever | format_docs
        )
    else:
        # 不启用 RAG，context 为空字符串
        retrieval_step = RunnablePassthrough.assign(context=lambda x: "")

    # 最终链：检索/透传 -> 调用自定义推理函数
    chain = (
        retrieval_step 
        | RunnableLambda(qwen_multimodal_generate)
    )
    return chain


# ===================== 6. UI 线程封装 (已更新) =====================

class Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str)  # (answer, rag_info)

    def __init__(self, query: str, img_path: str,
                 use_rag: bool, max_tokens: int, temperature: float):
        super().__init__()
        self.query = query
        self.img_path = img_path
        self.use_rag = use_rag
        self.gen_params = {"max_new_tokens": max_tokens, "temperature": temperature}

    def run(self):
        pil_img = None
        if self.img_path:
            try:
                pil_img = Image.open(self.img_path).convert("RGB")
            except Exception as e:
                print(f"[Image] open fail: {e}")
                pil_img = None
        
        # 获取链对象
        chain = get_rag_chain(self.use_rag)
        
        try:
            # 调用 LangChain invoke
            result = chain.invoke({
                "query": self.query, 
                "image": pil_img,
                "gen_params": self.gen_params
            })
            
            answer = result["answer"]
            rag_context = result["context_str"]
            if not rag_context:
                rag_context = "未使用或未检索到相关文档。"
                
        except Exception as e:
            answer = f"生成出错: {str(e)}"
            rag_context = "Error"
            print(f"[Error] Chain invoke failed: {e}")

        self.finished.emit(answer, rag_context)


# ===================== 6. 自定义聊天气泡控件 =====================

class ChatBubble(QtWidgets.QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumWidth(100)
        
        # 设置气泡阴影效果（使用Qt原生阴影）
        shadow = QtWidgets.QGraphicsDropShadowEffect(
            blurRadius=6, xOffset=2, yOffset=2, color=QtGui.QColor(0, 0, 0, 30)
        )
        self.setGraphicsEffect(shadow)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        # 用户名标签
        name = QtWidgets.QLabel("你" if is_user else "xiao taixian")
        name.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: %s;" %
            ("#2563eb" if not is_user else "#6366f1")
        )
        layout.addWidget(name)

        # 消息内容
        bubble = QtWidgets.QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bubble.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 12px; padding: 12px; "
            "font-size: 15px; line-height: 1.5; color: #111827; }" %
            ("#dbeafe" if not is_user else "#e0e7ff")  # 区分用户和助手的气泡颜色
        )
        layout.addWidget(bubble)

        # 设置对齐方式
        align = QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft
        layout.setAlignment(align)


class ChatArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border: none; }")

        self.container = QtWidgets.QWidget()
        self.v_layout = QtWidgets.QVBoxLayout(self.container)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        self.v_layout.setSpacing(16)  # 增加消息间距
        self.v_layout.addStretch(1)

        self.setWidget(self.container)

    def add_message(self, text: str, is_user: bool):
        # 在倒数第二个（stretch 之前）插入
        bubble = ChatBubble(text, is_user)
        self.v_layout.insertWidget(self.v_layout.count() - 1, bubble)
        QtCore.QTimer.singleShot(0, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ===================== 7. 主窗口 =====================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("xiao taixian 生活助手 (LangChain Edition)")
        self.resize(1150, 720)
        self.setMinimumSize(950, 620)

        # 设置窗口图标
        if os.path.exists(APP_ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))

        # 样式表
        self.setStyleSheet("""
            QMainWindow { background-color: #f3f4f6; }
            QLineEdit {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #d1d5db;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #2563eb; }
            QPushButton {
                border-radius: 16px;
                padding: 6px 14px;
                font-size: 13px;
                background-color: #2563eb;
                color: white;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #9ca3af; }
            QGroupBox {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                margin-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0px 4px;
                font-size: 12px;
                font-weight: 600;
                color: #111827;
            }
            QTextEdit {
                background-color: #f9fafb;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
                padding: 8px;
                font-size: 12px;
            }
            QLabel { color: #374151; }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header
        header = QtWidgets.QFrame()
        header.setFrameShape(QtWidgets.QFrame.NoFrame)
        header.setStyleSheet("QFrame { background-color: #ffffff; border-radius: 14px; border: 1px solid #e5e7eb; }")
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(16, 10, 16, 10)
        h_layout.setSpacing(10)

        logo_label = QtWidgets.QLabel()
        logo_label.setFixedSize(32, 32)
        if os.path.exists(APP_ICON_PATH):
            pix = QtGui.QPixmap(APP_ICON_PATH)
            logo_label.setPixmap(pix.scaled(32, 32, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        h_layout.addWidget(logo_label)

        title_box = QtWidgets.QVBoxLayout()
        title_label = QtWidgets.QLabel("xiao taixian")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #111827;")
        subtitle_label = QtWidgets.QLabel("基于 LangChain + 多模态大模型 + RAG 的助手 · 支持文本和图片输入")
        subtitle_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        h_layout.addLayout(title_box, 1)
        main_layout.addWidget(header)

        # Split
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(split, 1)

        # Left
        left_card = QtWidgets.QFrame()
        left_card.setStyleSheet("QFrame { background-color: #ffffff; border-radius: 14px; border: 1px solid #e5e7eb; }")
        left_layout = QtWidgets.QVBoxLayout(left_card)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        self.chat_area = ChatArea()
        left_layout.addWidget(self.chat_area, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("输入你的问题...")
        self.send_btn = QtWidgets.QPushButton("发送")
        self.send_btn.setFixedWidth(80)
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.send_btn)
        left_layout.addLayout(input_row)

        preview_row = QtWidgets.QHBoxLayout()
        self.preview_frame = QtWidgets.QFrame()
        pf_layout = QtWidgets.QHBoxLayout(self.preview_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setSpacing(6)
        self.image_preview = QtWidgets.QLabel("无图片")
        self.image_preview.setFixedSize(60, 60)
        self.image_preview.setStyleSheet("QLabel { background-color: #f9fafb; border-radius: 8px; border: 1px dashed #d1d5db; }")
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_name_label = QtWidgets.QLabel("未选择图片")
        pf_layout.addWidget(self.image_preview)
        pf_layout.addWidget(self.image_name_label, 1)
        preview_row.addWidget(self.preview_frame, 1)
        
        self.choose_img_btn = QtWidgets.QPushButton("选择图片")
        self.clear_img_btn = QtWidgets.QPushButton("清除")
        self.clear_img_btn.setStyleSheet("QPushButton { background-color: #6b7280; color: white; } QPushButton:hover { background-color: #4b5563; }")
        preview_row.addWidget(self.choose_img_btn)
        preview_row.addWidget(self.clear_img_btn)
        left_layout.addLayout(preview_row)
        split.addWidget(left_card)

        # Right
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(10)

        rag_group = QtWidgets.QGroupBox("RAG 检索参考内容")
        rag_layout = QtWidgets.QVBoxLayout(rag_group)
        rag_layout.setContentsMargins(10, 8, 10, 10)
        self.rag_text = QtWidgets.QTextEdit()
        self.rag_text.setReadOnly(True)
        self.rag_text.setStyleSheet("QTextEdit { font-family: 'Consolas', monospace; font-size: 11px; }")
        rag_layout.addWidget(self.rag_text)
        right_layout.addWidget(rag_group, 3)

        param_group = QtWidgets.QGroupBox("生成参数")
        param_layout = QtWidgets.QFormLayout(param_group)
        param_layout.setContentsMargins(10, 8, 10, 10)
        self.use_rag_cb = QtWidgets.QCheckBox("启用 RAG")
        self.use_rag_cb.setChecked(True)
        param_layout.addRow(self.use_rag_cb)
        
        self.max_tokens_spin = QtWidgets.QSpinBox()
        self.max_tokens_spin.setRange(32, 1024)
        self.max_tokens_spin.setValue(256)
        param_layout.addRow("最大长度", self.max_tokens_spin)
        
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 1.2)
        self.temp_spin.setValue(0.60)
        self.temp_spin.setSingleStep(0.05)
        param_layout.addRow("Temp", self.temp_spin)
        right_layout.addWidget(param_group, 2)

        split.addWidget(right_widget)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("LangChain + RAG 模型已就绪。")

        self.current_image_path = ""
        self.worker = None

        self.send_btn.clicked.connect(self.on_send)
        self.input_edit.returnPressed.connect(self.on_send)
        self.choose_img_btn.clicked.connect(self.on_choose_image)
        self.clear_img_btn.clicked.connect(self.on_clear_image)
        self.image_preview.mousePressEvent = self.on_preview_clicked

    # ---------- 事件 ---------- (逻辑保持不变)
    
    def on_choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path: return
        self.current_image_path = path
        self.image_name_label.setText(os.path.basename(path))
        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            self.image_preview.setPixmap(pix.scaled(60, 60, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_clear_image(self):
        self.current_image_path = ""
        self.image_name_label.setText("未选择图片")
        self.image_preview.setPixmap(QtGui.QPixmap())
        self.image_preview.setText("无图片")

    def on_preview_clicked(self, event):
        if not self.current_image_path: return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("图片预览")
        dlg.resize(640, 480)
        layout = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        pix = QtGui.QPixmap(self.current_image_path)
        if not pix.isNull():
            label.setPixmap(pix.scaled(dlg.size()*0.95, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        layout.addWidget(label)
        dlg.exec_()

    def on_send(self):
        query = self.input_edit.text().strip()
        if not query and not self.current_image_path: return
        
        self.chat_area.add_message(query if query else "(仅图片)", is_user=True)
        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self.status_bar.showMessage("LangChain 正在思考...")
        
        self.worker = Worker(
            query, 
            self.current_image_path, 
            self.use_rag_cb.isChecked(),
            self.max_tokens_spin.value(),
            float(self.temp_spin.value())
        )
        self.worker.finished.connect(self.on_answer_ready)
        self.worker.start()

    def on_answer_ready(self, answer: str, rag_info: str):
        self.chat_area.add_message(answer, is_user=False)
        self.rag_text.setPlainText(rag_info)
        self.send_btn.setEnabled(True)
        self.status_bar.showMessage("就绪。")


def main():
    app = QtWidgets.QApplication(sys.argv)
    if os.path.exists(APP_ICON_PATH):
        app.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()