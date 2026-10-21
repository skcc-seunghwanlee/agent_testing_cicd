"""Mock search backend. In a real project this is replaced by the internal search API.

Keeping it behind a small interface lets tests swap it without touching the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    body: str


class SearchBackend(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[Doc]: ...
    def get(self, doc_id: str) -> Doc | None: ...


CORPUS: list[Doc] = [
    Doc("hr-001", "연차 휴가 규정", "입사 1년 미만은 월 1일, 1년 이상은 연 15일의 연차가 부여된다. 연차는 다음 해 3월까지 사용할 수 있다."),
    Doc("hr-002", "재택근무 규정", "재택근무는 주 2회까지 가능하며 전날까지 팀장 승인이 필요하다."),
    Doc("it-001", "VPN 접속 방법", "사내망 외부에서는 VPN 클라이언트로 접속한다. 계정은 사번이며 OTP 인증이 필요하다."),
    Doc("it-002", "노트북 반납 절차", "퇴사 또는 교체 시 자산관리 시스템에서 반납 신청 후 IT 지원팀에 방문 반납한다."),
    Doc("fin-001", "출장비 정산", "출장 종료 후 7일 이내에 영수증을 첨부하여 정산을 신청한다. 7일을 넘기면 팀장 사유서가 필요하다."),
    Doc("sec-001", "문서 보안 등급", "문서는 공개, 내부, 기밀 3등급으로 분류한다. 기밀 문서는 외부 반출이 금지된다."),
]


class InMemoryBackend:
    """Naive keyword-overlap search. Deterministic, no network."""

    def __init__(self, corpus: list[Doc] | None = None):
        self._docs = list(corpus if corpus is not None else CORPUS)

    def search(self, query: str, top_k: int = 3) -> list[Doc]:
        terms = [t for t in query.replace(",", " ").split() if t]
        scored = []
        for d in self._docs:
            text = d.title + " " + d.body
            score = sum(1 for t in terms if t in text)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id))
        return [d for _, d in scored[:top_k]]

    def get(self, doc_id: str) -> Doc | None:
        return next((d for d in self._docs if d.doc_id == doc_id), None)
