from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from hybrid_rag.config import Settings
from hybrid_rag.generation import (
    GROUNDING_SYSTEM_PROMPT,
    ExtractiveGenerator,
    TextGenerator,
    citation_support_score,
    context_prompt,
    llm_verify_citation,
    parse_claims,
    repair_missing_citations,
)
from hybrid_rag.loaders import DocumentLoader
from hybrid_rag.models import (
    AnswerResult,
    Citation,
    ConfidenceScores,
    DocumentSummary,
    SearchHit,
)
from hybrid_rag.retrieval import HybridRetriever
from hybrid_rag.text import clamp, tokenize
from hybrid_rag.vector_store import VectorStore


class RAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: TextGenerator,
        vector_store: VectorStore,
        loader: DocumentLoader,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.vector_store = vector_store
        self.loader = loader
        self.extractive = ExtractiveGenerator()

    def ask(self, question: str, *, mode: str = "hybrid") -> AnswerResult:
        started = time.perf_counter()
        hits = self.retriever.retrieve(question, mode=mode)
        retrieval_confidence = self.retriever.retrieval_confidence(hits)
        if retrieval_confidence < self.settings.minimum_retrieval_confidence:
            sources = list(dict.fromkeys(hit.chunk.source for hit in hits[:3]))
            suffix = (
                f" Relevant files to inspect manually: {', '.join(sources)}." if sources else ""
            )
            answer = (
                "I couldn't find sufficiently relevant evidence in the indexed documentation "
                "to answer "
                f"this reliably.{suffix}"
            )
            return AnswerResult(
                question=question,
                answer=answer,
                citations=[],
                confidence=ConfidenceScores(
                    retrieval=retrieval_confidence,
                    citation_coverage=0.0,
                    completeness=0.0,
                    composite=0.5 * retrieval_confidence,
                ),
                retrieved_chunks=hits,
                status="insufficient_context",
                provider_used="none",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        degraded = False
        if isinstance(self.generator, ExtractiveGenerator):
            answer = self.extractive.generate_from_hits(question, hits)
        else:
            try:
                answer = self.generator.generate(
                    GROUNDING_SYSTEM_PROMPT, context_prompt(question, hits)
                )
            except (httpx.HTTPError, RuntimeError, KeyError, ValueError):
                answer = self.extractive.generate_from_hits(question, hits)
                degraded = True

        answer = repair_missing_citations(answer, hits)
        citations, citation_coverage = self._verify_citations(answer, hits, degraded)
        completeness = self._completeness(question, answer)
        composite = clamp(
            0.45 * retrieval_confidence + 0.35 * citation_coverage + 0.20 * completeness
        )
        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            confidence=ConfidenceScores(
                retrieval=retrieval_confidence,
                citation_coverage=citation_coverage,
                completeness=completeness,
                composite=composite,
            ),
            retrieved_chunks=hits,
            status="answered",
            provider_used=self.extractive.name if degraded else self.generator.name,
            degraded_mode=degraded,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _verify_citations(
        self, answer: str, hits: Sequence[SearchHit], degraded: bool
    ) -> tuple[list[Citation], float]:
        claims = parse_claims(answer)
        claim_count = len(claims)
        supported_claims = 0
        best_by_reference: dict[int, tuple[float, str]] = {}
        for claim, references in claims:
            claim_supported = False
            for reference in references:
                if not 1 <= reference <= len(hits):
                    continue
                evidence = hits[reference - 1].chunk.text
                lexical_score = citation_support_score(claim, evidence)
                # Directly copied or lightly edited evidence is more reliably verified locally;
                # reserve the slower model judge for lower-overlap paraphrases.
                score: float
                if lexical_score >= 0.8:
                    score = lexical_score
                else:
                    judge_score = (
                        None
                        if degraded or isinstance(self.generator, ExtractiveGenerator)
                        else (llm_verify_citation(self.generator, claim, evidence))
                    )
                    score = lexical_score if judge_score is None else judge_score
                if score >= 0.45:
                    claim_supported = True
                previous = best_by_reference.get(reference)
                if previous is None or score > previous[0]:
                    best_by_reference[reference] = (score, claim)
            supported_claims += int(claim_supported)
        citations = []
        for reference, (score, claim) in sorted(best_by_reference.items()):
            chunk = hits[reference - 1].chunk
            citations.append(
                Citation(
                    reference=reference,
                    chunk_id=chunk.id,
                    source=chunk.source,
                    section_heading=chunk.section_heading,
                    page_number=chunk.page_number,
                    quote=self._best_quote(claim, chunk.text),
                    verified=score >= 0.45,
                    verification_score=clamp(score),
                )
            )
        return citations, supported_claims / claim_count if claim_count else 0.0

    @staticmethod
    def _best_quote(claim: str, evidence: str, limit: int = 280) -> str:
        claim_tokens = set(tokenize(claim))
        sentences = [
            item.strip() for item in evidence.replace("\n", " ").split(".") if item.strip()
        ]
        best = max(
            sentences or [evidence],
            key=lambda sentence: len(claim_tokens & set(tokenize(sentence))),
        )
        return (best[: limit - 1] + "…") if len(best) > limit else best

    @staticmethod
    def _completeness(question: str, answer: str) -> float:
        question_terms = {token for token in tokenize(question) if len(token) > 3}
        answer_terms = set(tokenize(answer))
        term_coverage = len(question_terms & answer_terms) / max(1, len(question_terms))
        uncertainty_penalty = 0.4 if "does not contain enough" in answer.lower() else 0.0
        return clamp(0.35 + 0.65 * term_coverage - uncertainty_penalty)

    def list_documents(self) -> list[DocumentSummary]:
        chunks, _ = self.vector_store.all()
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
        return [
            DocumentSummary(
                id=document.id,
                source=document.source,
                media_type=document.media_type,
                checksum=document.checksum,
                chunk_count=counts.get(document.id, 0),
                created_at=document.created_at,
            )
            for document in self.loader.list_documents()
            if document.id in counts
        ]
