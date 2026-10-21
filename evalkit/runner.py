"""Run the graph once and return what tests need: answer, tool calls, node path."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class RunResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    steps: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    def to_tools_called(self) -> list[ToolCall]:
        return [ToolCall(name=c["name"], input_parameters=c["args"]) for c in self.tool_calls]

    def to_test_case(self, question: str, expected_tools: list[ToolCall] | None = None, **kw) -> LLMTestCase:
        return LLMTestCase(
            input=question,
            actual_output=self.answer,
            tools_called=self.to_tools_called(),
            expected_tools=expected_tools,
            **kw,
        )


def run_agent(graph, question: str, thread_id: str | None = None) -> RunResult:
    cfg = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    res = RunResult(answer="")
    for chunk in graph.stream({"messages": [HumanMessage(content=question)]}, cfg, stream_mode="updates"):
        for node, upd in chunk.items():
            res.path.append(node)
            for m in (upd or {}).get("messages", []) or []:
                if isinstance(m, AIMessage):
                    res.tool_calls.extend({"name": c["name"], "args": c["args"]} for c in m.tool_calls)
                    if not m.tool_calls:
                        res.answer = m.content
    res.steps = res.path.count("agent")
    return res
