# Architecture and production decisions

## Ingestion transaction boundary

Each file is checksum-addressed and stored in `data/raw`; normalized `Document` JSON is written to `data/processed`. Chunks contain source, document ID, index, strategy, heading, page, character count, and content hash. A source filename is treated as one logical document, so ingesting a changed revision replaces its earlier indexed chunks.

Embedding occurs before mutation. Exact hashes and cosine similarity against both stored and newly accepted chunks enforce deduplication. Chroma is updated first, then BM25 is rebuilt from Chroma truth. If either update fails, the newly inserted document is removed and BM25 is rebuilt from the surviving vector corpus.

For multi-process production ingestion, this in-process lock should be replaced with a distributed job queue and versioned index aliases. One API worker is intentionally configured in Docker so the documented consistency boundary is honest.

## Retrieval

Dense and BM25 searches independently return top-k candidates. Weighted Reciprocal Rank Fusion computes:

`score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))`

Scores are normalized for observability, not compared to original cosine or BM25 scores. The fused top 20 pass through `cross-encoder/ms-marco-MiniLM-L-6-v2`; the top five become context. The lexical reranker exists for tests and for explicit degraded operation.

## Generation and verification

Context blocks are numbered and include source location. The prompt requires a bracketed citation on every factual claim and explicitly rejects instructions embedded inside context. Claim parsing validates reference bounds and scores each claim/evidence pair. When the configured LLM is available it serves as the verifier; any judge failure falls back to a conservative token-support check.

Confidence is deliberately decomposed:

- Retrieval: rank-weighted final relevance.
- Citation coverage: fraction of answer claims with a supporting valid reference.
- Completeness: coverage of the question's meaningful terms, with uncertainty penalty.
- Composite: `0.45 retrieval + 0.35 citations + 0.20 completeness`.

These are operational signals, not calibrated probabilities. Production calibration would fit thresholds on human labels and monitor drift by question class.

## Threat model

The pipeline treats document text as untrusted. Relevant controls include extension/size allow-lists, basename normalization, no pickle deserialization, no raw provider responses in errors, environment-only secrets, prompt-injection instructions, and citation validation. Missing from this portfolio deployment are tenant isolation, malware scanning, document-level ACL filters, authentication, rate limiting, and encrypted external storage; those are called out rather than implied.

## Scaling path

1. Put uploads on object storage and ingestion jobs on a durable queue.
2. Create immutable, versioned vector/BM25 indexes and atomically switch aliases.
3. Add document ACLs to both retrieval paths before fusion.
4. Batch embeddings in the 50–250 range and add retry/idempotency records.
5. Cache query embeddings and reranker results with corpus-version keys.
6. Add OpenTelemetry traces for loader, embedding, retrieval, reranking, generation, and verification spans.

