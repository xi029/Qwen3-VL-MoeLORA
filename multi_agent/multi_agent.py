"""多智能体 + RAG 编排，去除对 LangChain Agent API 的依赖。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

try:  # LangChain >= 0.1 拆分
    from langchain_core.documents import Document as _LCDocument
except ImportError:
    try:
        from langchain.schema import Document as _LCDocument  # type: ignore
    except ImportError:  # 最后回退为 Any，避免类型问题
        _LCDocument = Any

Document = _LCDocument

from langchain_wrappers import LocalQwenTextLLM
from model_client import LocalMultimodalModel
from rag_pipeline import build_retriever, format_documents
from tools import save_session_summary_sync


@dataclass
class AgentOutput:
    plan: str
    answer: str
    supporting_documents: List[Document]
    summary_path: Optional[str] = None

#1、规划智能体（PlannerAgent）：将用户问题拆解为 3-5 个关键步骤，制定任务计划
#   工作方式：接收用户问题，通过 LLM 生成解决问题的步骤规划
class PlannerAgent:
    def __init__(self, llm: LocalQwenTextLLM) -> None:
        self._llm = llm

    def run(self, question: str) -> str:
        prompt = (
            "你是一名任务规划助手。请阅读用户问题，拆解需要完成的3-5个关键步骤。"
            "必须使用中文编号列出步骤，并在必要时提醒是否需要查阅知识库。"
            f"\n\n用户问题：{question}"
        )
        return self._llm.generate(prompt)

#2、知识智能体（KnowledgeAgent）：负责从知识库中检索与问题相关的文档
#工作方式：使用 RAG 流水线中构建的检索器（retriever），根据问题查找相关知识
class KnowledgeAgent:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    def search(self, question: str) -> List[Document]:
       # 从知识库中检索相关文档
        if self._retriever is None:
            return []
        if hasattr(self._retriever, "get_relevant_documents"):
            return self._retriever.get_relevant_documents(question)
        if hasattr(self._retriever, "invoke"):
            result = self._retriever.invoke(question)
            return list(result) if isinstance(result, list) else [result]
        if callable(self._retriever):
            result = self._retriever(question)
            return list(result) if isinstance(result, list) else [result]
        return []

#3、管理智能体（ManagerAgent）：根据问题和检索结果，决定是否需要使用知识库，并给出额外提示
#   工作方式：分析问题和检索到的信息，判断是否需要进一步检索或直接回答
class ManagerAgent:
    def __init__(self, llm: LocalQwenTextLLM) -> None:
        self._llm = llm

    def run(self, question: str, context_preview: str) -> str:
        prompt = (
            "你是一名调度智能体，需要根据问题与检索结果决定行动。"
            "请总结以下信息：\n"
            f"- 用户问题：{question}\n"
            f"- 检索摘要：{context_preview or '无'}\n\n"
            "输出格式：\n1. 工具调用建议（例如：需要/不需要使用知识库，并说明原因）\n"
            "2. 额外提示（若无可写“无”）。"
        )
        return self._llm.generate(prompt)

#4、响应智能体（ResponseAgent）：综合所有信息（任务计划、知识库内容、图片输入）生成最终回答
#   工作方式：接收规划、上下文和图片信息，生成自然语言回答
class ResponseAgent:
    def __init__(self, llm_client: LocalMultimodalModel) -> None:
        self._client = llm_client

    def respond(
        self,
        question: str,
        plan: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        review_feedback: Optional[str] = None,
    ) -> str:
        prompt = (
            "你是一名图文多模态专家。请综合任务计划与知识库内容回答用户问题，"
            "输出结构化中文答案，必要时引用 [Doc 序号]。"
            "\n\n【任务计划】\n"
            f"{plan}\n\n"
            "【知识库检索结果】\n"
            f"{context or '（未检索到相关文档）'}\n\n"
            "【用户问题】\n"
            f"{question}"
        )
        if review_feedback:
            prompt += (
                "\n\n【Reviewer 改进建议】\n"
                f"{review_feedback}\n"
                "请务必根据上述建议修订答案，确保完整回答用户。"
            )
        return self._client.generate(prompt=prompt, image_inputs=image_paths)


class ReviewerAgent:
    def __init__(self, llm: LocalQwenTextLLM) -> None:
        self._llm = llm

    def review(self, question: str, answer: str) -> Tuple[bool, str]:
        prompt = (
            "你是一名答案质检员，请判断助手的回复是否充分解决用户问题。"
            "请输出以下格式：\nVerdict: PASS 或 RETRY\nFeedback: 具体建议。"
            "\n\n【用户问题】\n"
            f"{question}\n\n"
            "【助手回答】\n"
            f"{answer}\n"
        )
        feedback = self._llm.generate(prompt, max_new_tokens=200)
        verdict_line = next(
            (line for line in feedback.splitlines() if "verdict" in line.lower()),
            "",
        )
        verdict = verdict_line.upper()
        is_pass = "PASS" in verdict and "RETRY" not in verdict
        return is_pass, feedback.strip()

#多智能体的协作流程
class MultiAgentOrchestrator:
    def __init__(self, knowledge_dir: str = "./knowledge_base") -> None:
        base_llm = LocalQwenTextLLM()
        retriever = build_retriever(knowledge_dir=knowledge_dir)
        #四个agent
        self._planner = PlannerAgent(base_llm)
        self._knowledge = KnowledgeAgent(retriever)
        self._manager = ManagerAgent(base_llm)
        self._responder = ResponseAgent(LocalMultimodalModel.get_shared())
        self._reviewer = ReviewerAgent(base_llm)
        self._summary_keywords = (
            "总结对话",
            "总结记录",
            "保存记录",
            "保存对话",
            "归档",
            "保存摘要",
            "save record",
            "save log",
            "archive",
            "summarize",
            "summary",
        )

    def run(
        self,
        question: str,
        image_paths: Optional[List[str]] = None,
        use_knowledge: bool = True,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> AgentOutput:
        history = chat_history or []
        is_summary_request = self._needs_session_summary(question)
        history_text = self._format_chat_history(history) if history else ""
        effective_question = question
        if is_summary_request and history_text:
            effective_question = (
                question.strip()
                + "\n\n【对话记录】\n"
                + history_text
                + "\n请基于上述对话内容生成总结或归档。"
            )
        # 1. 知识智能体检索相关文档
        enable_knowledge = use_knowledge and not is_summary_request
        docs = self._knowledge.search(question) if enable_knowledge else []
        context = format_documents(docs)
        # 2. 规划智能体制定任务计划
        planner_plan = self._planner.run(effective_question)
        # 3. 管理智能体给出调度建议
        manager_feedback = self._manager.run(
            question=effective_question,
            context_preview=context[:1200],
        )
        # 4. 响应智能体生成最终回答
        answer = self._responder.respond(
            question=effective_question,
            plan=planner_plan,
            context=context,
            image_paths=image_paths,
        )
        review_passed, review_notes = self._reviewer.review(effective_question, answer)
        if not review_passed:
            revised_answer = self._responder.respond(
                question=effective_question,
                plan=planner_plan,
                context=context,
                image_paths=image_paths,
                review_feedback=review_notes,
            )
            answer = revised_answer
            review_passed, review_notes = self._reviewer.review(effective_question, answer)

        review_summary = (
            "Reviewer Verdict: "
            + ("PASS" if review_passed else "RETRY")
            + "\n"
            + review_notes
        )

        combined_plan = (
            manager_feedback.strip()
            + "\n\n"
            + planner_plan.strip()
            + "\n\n"
            + review_summary.strip()
        )

        summary_path: Optional[str] = None
        if is_summary_request:
            summary_title = question[:24] or "对话总结"
            summary_payload = (
                "## 用户问题\n"
                f"{question}\n\n"
                "## 最终回答\n"
                f"{answer}\n\n"
                "## 调度计划\n"
                f"{combined_plan}\n\n"
                "## 对话记录\n"
                f"{history_text or '（无历史记录，无法总结上下文）'}\n\n"
                "## 检索证据\n"
                f"{context or '无'}\n"
            )
            summary_path = save_session_summary_sync(
                content=summary_payload,
                title=summary_title,
            )
            answer = (
                answer
                + "\n\n（已根据指令调用 save_session_summary 工具，文件路径："
                + summary_path
                + ")"
            )

        return AgentOutput(
            plan=combined_plan,
            answer=answer,
            supporting_documents=docs,
            summary_path=summary_path,
        )

    def _needs_session_summary(self, question: str) -> bool:
        normalized = question.strip()
        if not normalized:
            return False
        lower = normalized.lower()
        for keyword in self._summary_keywords:
            key_lower = keyword.lower()
            if keyword in normalized or key_lower in lower:
                return True
        return False

    def _format_chat_history(self, history: List[Tuple[str, str]]) -> str:
        if not history:
            return ""
        speaker_map = {
            "user": "用户",
            "assistant": "Multi-Agent",
        }
        lines = []
        for idx, (role, content) in enumerate(history, start=1):
            speaker = speaker_map.get(role.lower(), role)
            lines.append(f"{idx}. **{speaker}**：{content}")
        return "\n".join(lines)
