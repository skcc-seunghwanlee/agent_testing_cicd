from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import prompts

MAX_STEPS = 4

# Deterministic input guard. Cheap, runs before any LLM call.
BLOCKED = re.compile(r"(삭제|지워|drop\s+table|rm\s+-rf|비밀번호\s*알려)", re.I)


def guard(state: dict[str, Any]) -> dict[str, Any]:
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else ""
    if BLOCKED.search(text):
        return {"blocked": True, "messages": [AIMessage(content=prompts.load("refusal"))]}
    return {"blocked": False}


def make_agent_node(model_with_tools):
    def agent(state: dict[str, Any]) -> dict[str, Any]:
        msgs = [SystemMessage(content=prompts.load("system"))] + list(state["messages"])
        reply = model_with_tools.invoke(msgs)
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    return agent


def make_tool_node(tools):
    by_name = {t.name: t for t in tools}

    def run_tools(state: dict[str, Any]) -> dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                out.append(ToolMessage(content=f"unknown tool: {call['name']}", tool_call_id=call["id"]))
                continue
            try:
                result = tool.invoke(call["args"])
            except Exception as e:  # tool errors go back to the model, not up the stack
                result = f"tool error: {type(e).__name__}: {e}"
            out.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    return run_tools


# --- routing functions: pure, unit-testable without a graph -----------------

def route_after_guard(state: dict[str, Any]) -> str:
    return "refuse" if state.get("blocked") else "agent"


def route_after_agent(state: dict[str, Any]) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state.get("steps", 0) < MAX_STEPS:
        return "tools"
    return "end"
