"""Stub judge so deterministic DeepEval metrics can be constructed without any API key.

ToolCorrectnessMetric builds a model object in __init__ even when it never calls it.
Without a key that raises. This stub makes any accidental LLM call a hard error.
"""
from deepeval.models import DeepEvalBaseLLM


class NoLLM(DeepEvalBaseLLM):
    def __init__(self):
        super().__init__("no-llm")

    def load_model(self):
        return None

    def generate(self, *a, **k):
        raise RuntimeError("LLM must not be called in deterministic checks")

    async def a_generate(self, *a, **k):
        raise RuntimeError("LLM must not be called in deterministic checks")

    def get_model_name(self):
        return "no-llm"
