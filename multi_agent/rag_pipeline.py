"""
RAG（检索增强生成）流水线：
- 加载本地知识库目录下的文本/Markdown文件；
- 进行分块（避免长文丢失召回）；
- 使用句向量模型嵌入并构建 FAISS 向量索引；
- 暴露检索器以便 Agent 组件查询相关片段；
- 提供格式化函数将检索结果转为可阅读字符串。
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
try:  # LangChain 0.1+ 拆分
    from langchain_core.documents import Document  # type: ignore
except ImportError:  # 回退旧版本
    try:
        from langchain.schema import Document  # type: ignore
    except ImportError:
        from typing import Any as Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


DEFAULT_KNOWLEDGE_BASE = "./knowledge_base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


TEXT_ENCODINGS = ("utf-8", "utf-16", "gbk", "gb2312", "latin-1")
TEXT_EXTENSIONS = {".txt", ".md", ".mdx", ".csv", ".log"}
PDF_EXTENSIONS = {".pdf"}


@lru_cache(maxsize=1)
def build_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def _load_text_file(path: Path) -> List[Document]:
    """尝试多种编码打开文本文件，失败则忽略异常继续。"""
    for encoding in TEXT_ENCODINGS:
        try:
            loader = TextLoader(str(path), encoding=encoding)
            return loader.load()
        except UnicodeDecodeError:
            continue
    # 最后兜底，以忽略模式读取，避免完全丢失内容
    try:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=False)
        return loader.load()
    except Exception:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
        return [Document(page_content=content, metadata={"source": str(path)})]


def _load_pdf_file(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def _iter_knowledge_files(knowledge_dir: str) -> Iterable[Path]:
    base = Path(knowledge_dir)
    if not base.exists():
        return []
    return (file for file in base.rglob("*") if file.is_file())


def load_documents(knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE) -> List[Document]:
    if not os.path.isdir(knowledge_dir):
        return []

    documents: List[Document] = []
    for file_path in _iter_knowledge_files(knowledge_dir):
        suffix = file_path.suffix.lower()
        try:
            if suffix in TEXT_EXTENSIONS:
                documents.extend(_load_text_file(file_path))
            elif suffix in PDF_EXTENSIONS:
                documents.extend(_load_pdf_file(file_path))
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - 打印即可
            print(f"[RAG] 无法解析文件 {file_path}: {exc}")
            continue
    return documents


@lru_cache(maxsize=1)
def build_retriever(
    knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE,
    k: int = 3,
):
    """构建并缓存检索器（首次调用会进行向量化与索引构建）。

    参数：
    - knowledge_dir: 知识库目录路径。
    - k: 每次检索返回的文档片段数量。
    返回：
    - langchain 的检索器对象（vectorstore.as_retriever），或 None 当目录为空。
    """
    docs = load_documents(knowledge_dir)
    if not docs:
        return None
    # 将长文按字符递归分块，兼顾语义完整性与检索粒度
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = build_embeddings()
    # 用 FAISS 构建高效相似度搜索索引
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


def format_documents(docs: List[Document]) -> str:
    if not docs:
        return ""
    formatted_lines = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted_lines.append(f"[Doc {idx} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_lines)
