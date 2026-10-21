from __future__ import annotations

from langchain_core.tools import StructuredTool

from .backends import SearchBackend

# Single source of truth for the allow-list used by tests (ToolPermissionMetric).
ALLOWED_TOOLS = ["search_docs", "get_doc"]


def make_tools(backend: SearchBackend) -> list[StructuredTool]:
    def search_docs(query: str) -> str:
        """사내 문서를 키워드로 검색한다. 결과는 doc_id, 제목, 본문 일부."""
        docs = backend.search(query)
        if not docs:
            return "검색 결과 없음"
        return "\n".join(f"[{d.doc_id}] {d.title}: {d.body[:80]}" for d in docs)

    def get_doc(doc_id: str) -> str:
        """doc_id로 문서 전문을 가져온다."""
        d = backend.get(doc_id)
        return f"[{d.doc_id}] {d.title}\n{d.body}" if d else f"문서 없음: {doc_id}"

    return [
        StructuredTool.from_function(search_docs),
        StructuredTool.from_function(get_doc),
    ]
