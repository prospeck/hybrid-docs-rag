from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8080").rstrip("/")

st.set_page_config(page_title="Hybrid Docs RAG", page_icon="🔎", layout="wide")
st.title("Hybrid Docs RAG")
st.caption("Dense vectors + BM25 → weighted RRF → reranking → grounded answer verification")


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"API request failed: {exc}")
        return None


with st.sidebar:
    st.header("Document index")
    upload = st.file_uploader("Add an internal document", type=["md", "txt", "html", "htm", "pdf"])
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "semantic"])
    if upload and st.button("Ingest document", type="primary", use_container_width=True):
        result = api_request(
            "POST",
            "/v1/ingest",
            params={"strategy": strategy},
            files={"file": (upload.name, upload.getvalue(), upload.type)},
        )
        if result:
            st.success(
                f"Indexed {result['chunks_created']} chunks; skipped "
                f"{result['duplicates_skipped']} duplicates."
            )
    documents = api_request("GET", "/v1/documents") or []
    st.metric("Indexed documents", len(documents))
    for document in documents:
        st.caption(f"{document['source']} · {document['chunk_count']} chunks")

question = st.text_area(
    "Ask a question",
    placeholder="Example: How long are production access approvals valid?",
    height=90,
)
compare = st.toggle("Compare hybrid retrieval with dense-only", value=False)


def render_result(result: dict[str, Any], heading: str) -> None:
    st.subheader(heading)
    if result.get("degraded_mode"):
        st.warning(
            "The configured LLM was unavailable; a grounded extractive fallback "
            "answered this query."
        )
    st.markdown(result["answer"])
    scores = result["confidence"]
    cols = st.columns(4)
    cols[0].metric("Composite", f"{scores['composite']:.0%}")
    cols[1].metric("Retrieval", f"{scores['retrieval']:.0%}")
    cols[2].metric("Citation coverage", f"{scores['citation_coverage']:.0%}")
    cols[3].metric("Completeness", f"{scores['completeness']:.0%}")
    st.caption(f"Provider: {result['provider_used']} · {result['latency_ms']:.0f} ms")

    if result["citations"]:
        st.markdown("#### Verified sources")
        for citation in result["citations"]:
            icon = "✅" if citation["verified"] else "⚠️"
            location = citation["source"]
            if citation.get("section_heading"):
                location += f" — {citation['section_heading']}"
            if citation.get("page_number"):
                location += f" · page {citation['page_number']}"
            with st.expander(
                f"{icon} [{citation['reference']}] {location} "
                f"({citation['verification_score']:.0%})"
            ):
                st.write(citation["quote"])

    with st.expander("Retrieval trace", expanded=False):
        for rank, hit in enumerate(result["retrieved_chunks"], start=1):
            st.markdown(f"**{rank}. {hit['chunk']['source']}** — score `{hit['score']:.3f}`")
            st.caption(
                f"dense={hit.get('dense_score')} · sparse={hit.get('sparse_score')} · "
                f"fusion={hit.get('fusion_score')} · rerank={hit.get('rerank_score')}"
            )
            st.write(hit["chunk"]["text"][:900])


if st.button("Ask", type="primary", disabled=not question.strip()):
    if compare:
        left, right = st.columns(2)
        hybrid = api_request(
            "POST", "/v1/ask", json={"question": question, "retrieval_mode": "hybrid"}
        )
        dense = api_request(
            "POST", "/v1/ask", json={"question": question, "retrieval_mode": "dense"}
        )
        if hybrid:
            with left:
                render_result(hybrid, "Hybrid")
        if dense:
            with right:
                render_result(dense, "Dense only")
    else:
        result = api_request(
            "POST", "/v1/ask", json={"question": question, "retrieval_mode": "hybrid"}
        )
        if result:
            render_result(result, "Answer")
