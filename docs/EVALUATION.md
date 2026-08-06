# Evaluation framework

`eval_dataset.json` contains 56 curated questions tied to the synthetic Northstar documentation corpus. It covers direct lookups, exact identifiers/config keys, multi-hop synthesis, ambiguous wording, and questions that have no answer in the corpus.

## Metrics

- **Answer correctness:** token recall against the expected answer. Unanswerable cases pass only when the service returns `insufficient_context`.
- **Faithfulness:** the runtime fraction of claims backed by a verified citation.
- **Retrieval relevance:** mean reciprocal rank of the expected source documents in the final set. This rewards putting the right evidence first instead of treating every top-five position as equivalent.
- **Citation accuracy:** fraction of returned citations that pass claim/evidence verification.
- **Latency:** end-to-end service latency, including local generation and verification when enabled.

The metric implementations are deterministic so the suite can run without paid keys. They are intentionally separated from optional LLM judging: lexical correctness is reproducible but does not capture all paraphrases. For a real deployment, add blinded human labels, inter-rater agreement, semantic-answer scoring, and slice-level confidence intervals.

## Experiment protocol

`scripts/compare_chunking.py` uses an isolated index per strategy and fixes all other variables. It runs hybrid and dense-only modes for each corpus. The no-key experiment uses deterministic hash embeddings, lexical reranking, and extractive generation to make CI reproducible; it measures pipeline behavior, not the quality ceiling of the default local neural models.

Do not compare reports created with different corpora, provider models, thresholds, or dataset revisions without recording those changes.
