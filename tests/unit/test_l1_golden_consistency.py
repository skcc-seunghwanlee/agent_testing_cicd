"""Golden set must stay consistent with the corpus and the guard, otherwise L3 results are meaningless."""
import json
from pathlib import Path

import pytest

from infoagent.backends import CORPUS
from infoagent.nodes import BLOCKED
from infoagent.tools import ALLOWED_TOOLS

pytestmark = pytest.mark.l1
GOLDEN = json.loads((Path(__file__).resolve().parents[2] / "data" / "golden.json").read_text(encoding="utf-8"))


def test_ids_unique():
    ids = [g["id"] for g in GOLDEN]
    assert len(ids) == len(set(ids))


def test_expected_docs_exist_in_corpus():
    ids = {d.doc_id for d in CORPUS}
    for g in GOLDEN:
        assert g["expected_doc"] is None or g["expected_doc"] in ids, g["id"]


def test_expected_tools_are_allowed():
    for g in GOLDEN:
        assert set(g["expected_tools"]) <= set(ALLOWED_TOOLS), g["id"]


def test_blocked_flag_matches_guard():
    for g in GOLDEN:
        assert bool(BLOCKED.search(g["question"])) == g["blocked"], g["id"]


def test_smoke_subset_covers_every_tag():
    tags = {t for g in GOLDEN for t in g["tags"]}
    smoke_tags = {t for g in GOLDEN if g["smoke"] for t in g["tags"]}
    assert tags - smoke_tags == {"multi"}
