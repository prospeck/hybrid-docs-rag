# Case study: reliable answers over internal documentation

## Problem

Internal documentation mixes natural-language policies with exact error codes, headers, and configuration keys. Dense retrieval is strong at paraphrases but can miss identifiers; keyword retrieval finds identifiers but often ranks semantically irrelevant mentions. A useful system must also show where an answer came from and refuse unsupported questions.

## Approach

I built one ingestion model shared by Chroma and BM25, weighted RRF to combine independent rankings, and a cross-encoder precision pass. Grounded generation uses numbered evidence blocks. A second verification pass evaluates each claim/reference pair, and the API exposes the retrieval trace and confidence components instead of returning an opaque answer string.

## Evidence

The repository includes a reproducible 56-case golden suite and a controlled comparison across three chunking strategies and dense-only versus hybrid retrieval. Run `uv run python scripts/compare_chunking.py` to populate the measured results table in `eval-results/chunking-comparison.md`.

The committed [comparison report](../eval-results/chunking-comparison.md) records a reproducible no-key baseline. Across 56 cases it measured 100% faithfulness and 100% citation accuracy for every strategy. Fixed-size hybrid retrieval produced the best source-ranking score at 82.2% MRR. On semantic chunks, hybrid retrieval improved source MRR from 78.8% to 80.1% over dense-only. The deterministic extractive generator reached 62.6% token-recall correctness in the best run; this is a regression floor, not a claim about the neural stack's quality ceiling.

The report records configuration and case count alongside every result so model or dependency changes cannot silently leave stale portfolio claims.

## What I would do next

The next production step is ACL-aware retrieval: every chunk would carry tenant and group principals, and both dense and BM25 candidates would be filtered before fusion. I would then calibrate the refusal threshold on human-reviewed traffic, add a durable ingestion queue, and ship OpenTelemetry traces plus slice-based drift monitoring.
