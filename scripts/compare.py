"""Gate on relative regression: compare results/l3_latest.jsonl against a baseline file.

  python scripts/compare.py --baseline baseline/l3.jsonl --margin 0.10
Exit 1 if (a) any deterministic check that passed in baseline now fails, or
(b) mean judge score dropped by more than --margin.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path


def load(p):
    return {j["id"]: j for j in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines() if Path(p).exists() else [])}


ap = argparse.ArgumentParser()
ap.add_argument("--baseline", default="baseline/l3.jsonl")
ap.add_argument("--latest", default="results/l3_latest.jsonl")
ap.add_argument("--margin", type=float, default=0.10)
a = ap.parse_args()

base, cur = load(a.baseline), load(a.latest)
if not base:
    print("no baseline yet -> nothing to compare (copy latest to baseline once you trust it)")
    sys.exit(0)

common = sorted(set(base) & set(cur))
bad = []
for i in common:
    for k, v in base[i]["det"].items():
        if v and not cur[i]["det"].get(k, False):
            bad.append(f"{i}: deterministic '{k}' regressed")
drop = st.mean(base[i]["judge"] for i in common) - st.mean(cur[i]["judge"] for i in common) if common else 0
print(f"compared {len(common)} cases, judge mean drop = {drop:+.3f} (margin {a.margin})")
if drop > a.margin:
    bad.append(f"judge mean dropped {drop:.3f} > {a.margin}")
print("\n".join(bad) or "OK")
sys.exit(1 if bad else 0)
