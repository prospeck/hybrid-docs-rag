# Chunking and retrieval comparison

Reproducible no-key regression baseline. These figures measure the deterministic test stack, not the quality ceiling of the default neural stack.

| Strategy | Retrieval | Cases | Correctness | Faithfulness | Retrieval relevance | Citation accuracy | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | hybrid | 56 | 62.6% | 100.0% | 82.2% | 100.0% | 3.2 ms |
| fixed | dense | 56 | 62.6% | 100.0% | 82.2% | 100.0% | 1.7 ms |
| recursive | hybrid | 56 | 62.4% | 100.0% | 80.7% | 100.0% | 1.8 ms |
| recursive | dense | 56 | 62.4% | 100.0% | 80.7% | 100.0% | 1.4 ms |
| semantic | hybrid | 56 | 58.0% | 100.0% | 80.1% | 100.0% | 3.7 ms |
| semantic | dense | 56 | 58.0% | 100.0% | 78.8% | 100.0% | 3.9 ms |
