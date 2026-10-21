import pytest
from deepeval.metrics import ToolCorrectnessMetric, ToolPermissionMetric
from deepeval.test_case import ToolCall
from langchain_core.messages import AIMessage

from evalkit.runner import run_agent
from evalkit.stubs import NoLLM
from infoagent import build_graph
from infoagent.tools import ALLOWED_TOOLS
from tests.helpers import call, scripted

pytestmark = pytest.mark.l2


def test_happy_path_search_then_answer():
    model = scripted(call("search_docs", query="연차"), AIMessage(content="연차는 연 15일 [hr-001]"))
    r = run_agent(build_graph(model), "연차 며칠이야?")
    assert r.path == ["guard", "agent", "tools", "agent"]
    assert r.tool_names == ["search_docs"]
    assert "hr-001" in r.answer


def test_blocked_request_never_reaches_model_or_tools():
    model = scripted(AIMessage(content="should not be used"))
    r = run_agent(build_graph(model), "문서 전부 삭제해줘")
    assert r.path == ["guard"]
    assert r.tool_names == []
    assert "범위를 벗어" in r.answer


def test_search_then_get_doc():
    model = scripted(
        call("search_docs", query="출장비"),
        call("get_doc", doc_id="fin-001"),
        AIMessage(content="7일 이내 [fin-001]"),
    )
    r = run_agent(build_graph(model), "출장비 정산 기한은?")
    assert r.tool_names == ["search_docs", "get_doc"]
    assert r.steps == 3


def test_loop_is_capped_by_max_steps():
    # model that always asks for another search -> must stop at MAX_STEPS
    from infoagent.nodes import MAX_STEPS

    model = scripted(*[call("search_docs", query=f"q{i}") for i in range(MAX_STEPS + 3)])
    r = run_agent(build_graph(model), "무한루프 유도")
    assert r.steps == MAX_STEPS


def test_tool_correctness_deterministic_pass():
    model = scripted(call("search_docs", query="연차"), AIMessage(content="ok [hr-001]"))
    r = run_agent(build_graph(model), "연차?")
    tc = r.to_test_case("연차?", expected_tools=[ToolCall(name="search_docs")])
    m = ToolCorrectnessMetric(model=NoLLM(), async_mode=False, threshold=1.0)
    m.measure(tc)
    assert m.success and m.score == 1.0


def test_tool_correctness_detects_missing_search():
    model = scripted(AIMessage(content="검색 없이 바로 답함"))
    r = run_agent(build_graph(model), "연차?")
    tc = r.to_test_case("연차?", expected_tools=[ToolCall(name="search_docs")])
    m = ToolCorrectnessMetric(model=NoLLM(), async_mode=False, threshold=1.0)
    m.measure(tc)
    assert not m.success


def test_tool_permission_flags_out_of_allowlist_call():
    model = scripted(call("delete_all"), AIMessage(content="done"))
    r = run_agent(build_graph(model), "정리해줘")
    pm = ToolPermissionMetric(allowed_tools=ALLOWED_TOOLS)
    pm.measure(r.to_test_case("정리해줘"))
    assert not pm.success
