from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .backends import InMemoryBackend, SearchBackend
from .nodes import (
    guard,
    make_agent_node,
    make_tool_node,
    route_after_agent,
    route_after_guard,
)
from .tools import make_tools


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    blocked: bool
    steps: int


def build_graph(model, backend: SearchBackend | None = None, checkpointer: Any = None):
    """model: any LangChain chat model supporting bind_tools (or a fake in tests)."""
    tools = make_tools(backend or InMemoryBackend())
    try:
        bound = model.bind_tools(tools)
    except NotImplementedError:
        bound = model  # fake models used in tests already carry scripted tool_calls

    g = StateGraph(State)
    g.add_node("guard", guard)
    g.add_node("agent", make_agent_node(bound))
    g.add_node("tools", make_tool_node(tools))
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", route_after_guard, {"agent": "agent", "refuse": END})
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())
