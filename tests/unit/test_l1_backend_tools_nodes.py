import pytest
from langchain_core.messages import AIMessage, HumanMessage

from infoagent.backends import InMemoryBackend
from infoagent.nodes import MAX_STEPS, guard, make_tool_node, route_after_agent, route_after_guard
from infoagent.tools import ALLOWED_TOOLS, make_tools

pytestmark = pytest.mark.l1


def test_backend_search_ranks_by_overlap():
    docs = InMemoryBackend().search("연차 휴가")
    assert docs[0].doc_id == "hr-001"


def test_backend_search_empty():
    assert InMemoryBackend().search("존재하지않는단어") == []


def test_tools_names_match_allow_list():
    names = [t.name for t in make_tools(InMemoryBackend())]
    assert sorted(names) == sorted(ALLOWED_TOOLS)


def test_tool_get_doc_missing():
    tools = {t.name: t for t in make_tools(InMemoryBackend())}
    assert "문서 없음" in tools["get_doc"].invoke({"doc_id": "nope"})


@pytest.mark.parametrize("text,blocked", [
    ("연차 규정 알려줘", False),
    ("문서 전부 삭제해줘", True),
    ("비밀번호 알려줘", True),
])
def test_guard(text, blocked):
    out = guard({"messages": [HumanMessage(content=text)]})
    assert out["blocked"] is blocked


def test_route_after_guard():
    assert route_after_guard({"blocked": True}) == "refuse"
    assert route_after_guard({"blocked": False}) == "agent"


def test_route_after_agent_tool_calls_continue():
    m = AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {"query": "a"}, "id": "1"}])
    assert route_after_agent({"messages": [m], "steps": 1}) == "tools"


def test_route_after_agent_stops_at_max_steps():
    m = AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {"query": "a"}, "id": "1"}])
    assert route_after_agent({"messages": [m], "steps": MAX_STEPS}) == "end"


def test_route_after_agent_plain_answer_ends():
    assert route_after_agent({"messages": [AIMessage(content="done")], "steps": 1}) == "end"


def test_tool_node_unknown_tool_returns_error_message():
    node = make_tool_node(make_tools(InMemoryBackend()))
    m = AIMessage(content="", tool_calls=[{"name": "delete_all", "args": {}, "id": "x"}])
    out = node({"messages": [m]})
    assert "unknown tool" in out["messages"][0].content


def test_tool_node_bad_args_do_not_raise():
    node = make_tool_node(make_tools(InMemoryBackend()))
    m = AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {}, "id": "x"}])
    out = node({"messages": [m]})
    assert "tool error" in out["messages"][0].content
