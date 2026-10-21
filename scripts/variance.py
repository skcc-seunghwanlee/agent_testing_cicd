"""Run the golden set N times and report per-case pass-rate / judge-score spread.

Use it once to learn how noisy YOUR model + judge are before choosing a regression margin.
  python scripts/variance.py --runs 5 --smoke            # real LLM (needs LLM_API_KEY)
  python scripts/variance.py --runs 3 --fake             # offline wiring check, no LLM
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]

from langchain_core.messages import AIMessage  # noqa: E402

from evalkit.runner import run_agent  # noqa: E402
from evalkit.scoring import deterministic_checks, judge_check, load_golden  # noqa: E402
from infoagent import build_graph  # noqa: E402


def fake_model_for(case):
    from tests.helpers import call, scripted

    if case["blocked"]:
        return scripted(AIMessage(content="unused"))
    doc = case["expected_doc"]
    return scripted(call("search_docs", query=case["question"]), AIMessage(content=f"답변 [{doc}]" if doc else "문서에서 확인되지 않음"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    cases = load_golden(a.smoke)
    judge = None
    live_graph = None
    if not a.fake:
        from evalkit.live import OpenAICompatJudge, make_chat_model

        judge = OpenAICompatJudge()
        live_graph = build_graph(make_chat_model())

    print(f"{'id':5} {'det_pass':>9} {'judge_mean':>11} {'judge_sd':>9}")
    all_sd = []
    for c in cases:
        det_pass, js = 0, []
        for _ in range(a.runs):
            g = build_graph(fake_model_for(c)) if a.fake else live_graph
            r = run_agent(g, c["question"])
            det_pass += all(deterministic_checks(c, r).values())
            if judge:
                js.append(judge_check(c, r, judge))
        sd = st.pstdev(js) if len(js) > 1 else 0.0
        all_sd.append(sd)
        print(f"{c['id']:5} {det_pass}/{a.runs:>7} {(st.mean(js) if js else float('nan')):11.3f} {sd:9.3f}")
    if judge:
        print(f"\nmean per-case judge sd = {st.mean(all_sd):.3f}  -> use >= 2x this as the regression margin")


if __name__ == "__main__":
    main()
