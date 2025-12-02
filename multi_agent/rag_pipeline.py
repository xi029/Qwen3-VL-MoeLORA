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
from typing import Any, Iterable, List, Optional

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

try:
    from sentence_transformers import CrossEncoder  # type: ignore

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:  # sentence-transformers 缺失时降级
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False

try:  # BM25 依赖 rank_bm25，若缺失则降级为纯向量检索
    from langchain_community.retrievers import BM25Retriever  # type: ignore

    _BM25_AVAILABLE = True
except ImportError:
    BM25Retriever = None  # type: ignore
    _BM25_AVAILABLE = False


DEFAULT_KNOWLEDGE_BASE = "./knowledge_base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
BM25_TOP_K = 8
VECTOR_TOP_K = 8
RERANK_TOP_N = 4


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
@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> Any:
    if CrossEncoder is None:
        raise RuntimeError("sentence-transformers 未安装，无法使用 CrossEncoder reranker。")
    return CrossEncoder(model_name, max_length=512)


class SimpleCrossEncoderReranker:
    def __init__(self, model_name: str, top_n: int) -> None:
        self._model_name = model_name
        self._top_n = top_n
        self._model = _load_cross_encoder(model_name)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[: self._top_n]]


class HybridRetriever:
    """轻量混合检索器，不依赖 langchain.retrievers 子模块。"""

    def __init__(
        self,
        keyword_retriever: Optional[Any],
        semantic_retriever: Any,
        *,
        k: int,
        bm25_weight: float = BM25_WEIGHT,
        semantic_weight: float = VECTOR_WEIGHT,
    reranker: Optional[Any] = None,
    ) -> None:
        self._keyword = keyword_retriever
        self._semantic = semantic_retriever
        self._k = k
        self._weights = (bm25_weight, semantic_weight)
        self._reranker = reranker

    def _run_retriever(self, retriever: Any, weight: float, query: str) -> List[tuple]:
        if retriever is None or weight <= 0:
            return []
        documents: List[Document] = []
        if hasattr(retriever, "get_relevant_documents"):
            documents = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "invoke"):
            result = retriever.invoke(query)
            if isinstance(result, list):
                documents = result
            elif result:
                documents = [result]
        elif callable(retriever):
            result = retriever(query)
            if isinstance(result, list):
                documents = result
            elif result:
                documents = [result]
        scored = []
        for idx, doc in enumerate(documents):
            key = (doc.metadata.get("source"), doc.page_content[:200])
            score = weight / (idx + 1)
            scored.append((key, score, doc))
        return scored

    def get_relevant_documents(self, query: str) -> List[Document]:
        scored_docs: dict = {}
        for key, score, doc in self._run_retriever(self._keyword, self._weights[0], query):
            current = scored_docs.get(key)
            if current is None or score > current[0]:
                scored_docs[key] = (score, doc)
        for key, score, doc in self._run_retriever(self._semantic, self._weights[1], query):
            current = scored_docs.get(key)
            if current is None or score > current[0]:
                scored_docs[key] = (score, doc)
        docs_sorted = [pair[1] for pair in sorted(scored_docs.values(), key=lambda item: item[0], reverse=True)]
        if self._reranker:
            try:
                docs_sorted = self._reranker.compress_documents(docs_sorted, query)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                print("[RAG] Cross-Encoder rerank 失败，退回混合检索排序：", exc)
        return docs_sorted[: self._k]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


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

    # ---- 语义检索：FAISS ----
    vectorstore = FAISS.from_documents(chunks, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_TOP_K})

    # ---- 关键词检索：BM25 ----
    keyword_retriever: Optional[Any] = None
    if _BM25_AVAILABLE:
        try:
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = BM25_TOP_K
        except Exception as exc:  # noqa: BLE001
            print(
                "[RAG] BM25Retriever 构建失败，已降级为纯向量检索。"
                "请确认已安装 rank_bm25 依赖。",
                exc,
            )

    # ---- 混合检索 ----
    reranker: Optional[Any] = None
    if _CROSS_ENCODER_AVAILABLE:
        try:
            reranker = SimpleCrossEncoderReranker(
                model_name=DEFAULT_RERANK_MODEL,
                top_n=max(k, RERANK_TOP_N),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[RAG] 重排序模型加载失败，已回退至混合检索排序。"
                "请确认已安装 sentence-transformers 并可访问 BAAI/bge-reranker-base。",
                exc,
            )
    else:
        print(
            "[RAG] sentence-transformers 未安装，无法启用交叉编码重排序，将直接返回混合检索结果。"
        )

    return HybridRetriever(
        keyword_retriever=keyword_retriever,
        semantic_retriever=semantic_retriever,
        k=k,
        reranker=reranker,
    )


def format_documents(docs: List[Document]) -> str:
    if not docs:
        return ""
    formatted_lines = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted_lines.append(f"[Doc {idx} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_lines)
