"""Real LLM wiring (OpenAI-compatible endpoint). Env: LLM_API_KEY, LLM_BASE_URL (optional), LLM_MODEL, JUDGE_MODEL (optional)."""
from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM


def have_llm() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def make_chat_model(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL") or None,
        temperature=temperature,
    )


class OpenAICompatJudge(DeepEvalBaseLLM):
    """DeepEval judge on any OpenAI-compatible endpoint (e.g. in-house vLLM)."""

    def __init__(self, model: str | None = None):
        self._name = model or os.getenv("JUDGE_MODEL") or os.environ["LLM_MODEL"]
        super().__init__(self._name)

    def load_model(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._name,
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL") or None,
            temperature=0,
        )

    def generate(self, prompt: str, schema=None):
        m = self.load_model()
        if schema is not None:
            return m.with_structured_output(schema).invoke(prompt)
        return m.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        m = self.load_model()
        if schema is not None:
            return await m.with_structured_output(schema).ainvoke(prompt)
        return (await m.ainvoke(prompt)).content

    def get_model_name(self):
        return self._name
