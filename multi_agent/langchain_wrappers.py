"""面向多智能体的简易文本 LLM 封装。"""

from __future__ import annotations

from typing import Optional

from model_client import LocalMultimodalModel


class LocalQwenTextLLM:
    """对本地多模态模型进行再封装，仅返回纯文本结果。"""

    def __init__(
        self,
        default_max_new_tokens: int = 256,
        default_temperature: float = 0.7,
    ) -> None:
        self._model = LocalMultimodalModel.get_shared()
        self._default_max_tokens = default_max_new_tokens
        self._default_temperature = default_temperature

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """输出文本，供 Planner/Manager 等文本智能体使用。"""

        tokens = max_new_tokens or self._default_max_tokens
        temp = temperature or self._default_temperature
        return self._model.generate(
            prompt=prompt,
            image_inputs=None,
            max_new_tokens=tokens,
            temperature=temp,
        )
