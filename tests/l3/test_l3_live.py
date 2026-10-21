"""L3: real LLM, full trajectory. Skipped when LLM_API_KEY is unset.

SMOKE=1 -> only golden cases flagged smoke. Results are appended to results/l3_latest.jsonl
so scripts/compare.py can gate on regression vs baseline instead of absolute judge scores.
"""
import json
import os
from pathlib import Path

import pytest

from evalkit.live import OpenAICompatJudge, have_llm, make_chat_model
from evalkit.runner import run_agent
from evalkit.scoring import deterministic_checks, judge_check, load_golden
from infoagent import build_graph

pytestmark = [pytest.mark.l3, pytest.mark.skipif(not have_llm(), reason="LLM_API_KEY not set")]

CASES = load_golden(smoke_only=os.getenv("SMOKE") == "1")
OUT = Path(__file__).resolve().parents[2] / "results" / "l3_latest.jsonl"


@pytest.fixture(scope="module")
def graph():
    return build_graph(make_chat_model())


@pytest.fixture(scope="module")
def judge():
    return OpenAICompatJudge()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case(case, graph, judge):
    r = run_agent(graph, case["question"])
    det = deterministic_checks(case, r)
    score = judge_check(case, r, judge)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": case["id"], "det": det, "judge": score, "steps": r.steps,
                            "tools": r.tool_names}, ensure_ascii=False) + "\n")

    # Deterministic parts are hard failures; the judge score is recorded and gated relatively.
    assert all(det.values()), det
