"""MCP 风格工具集合：用于多智能体调用的本地工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain.tools import tool  # type: ignore

REPORT_DIR = Path("./output/reports")


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "-", text.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "session"


@tool("save_session_summary")
def save_session_summary(content: str, title: str = "对话总结") -> str:
    """保存会话纪要到 Markdown，模仿 MCP server 的归档指令。"""

    _ensure_report_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{_slugify(title)}.md"
    file_path = REPORT_DIR / filename
    header = f"# {title}\n\n"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(content.strip())
        f.write("\n")
    return str(file_path.resolve())


def save_session_summary_sync(content: str, title: str = "对话总结") -> str:
    """便捷函数：直接调用 LangChain 工具的同步实现。"""

    return save_session_summary.invoke({"content": content, "title": title})
