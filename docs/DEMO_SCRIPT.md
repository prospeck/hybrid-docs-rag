# Four-minute demo walkthrough

Target length: 3 minutes 35 seconds. Record the dashboard and terminal side by side at 1080p.

## 0:00–0:25 — Frame the problem

“Internal docs mix natural-language policies with exact error codes and configuration keys. This system combines dense retrieval with BM25, fuses the rankings, reranks the candidates, and verifies every citation before returning an answer.”

Show the architecture diagram in the README, then open the dashboard.

## 0:25–0:55 — Ingest documents

Upload `02-incident-response.md` with recursive chunking. Point out the created chunk count and duplicate count. Upload it a second time to show the idempotent duplicate path.

Say: “Raw and normalized copies are retained for re-indexing. Chroma and BM25 share one chunk model, and BM25 is rebuilt from vector-store truth so the indexes cannot silently drift.”

## 0:55–1:35 — Exact-term hybrid retrieval

Ask: “What does OBS-1042 mean and how do I fix it?”

Expand the verified citation and retrieval trace. Highlight the dense rank, sparse rank, fused score, rerank score, source filename, and quote.

Say: “BM25 protects exact identifiers such as `OBS-1042` and `METRICS_PUSH_TOKEN`; the dense path handles paraphrases. Weighted reciprocal-rank fusion avoids pretending their raw scores use the same scale.”

## 1:35–2:10 — Multi-hop answer

Ingest `03-platform-deployments.md`, then ask: “For a SEV-1 caused by a release, when should we roll back and how often update customers?”

Show that the answer combines the incident and deployment documents with separate references. Point out retrieval, citation coverage, completeness, and composite confidence.

## 2:10–2:35 — Refusal behavior

Ask: “What is Northstar's parental leave policy?”

Show the structured insufficient-context response. Say: “A low retrieval score is a product state, not permission to invent an answer. The response reports what was found and which documents are worth checking manually.”

## 2:35–3:05 — Hybrid versus dense-only

Enable the comparison toggle and rerun the OBS-1042 question. Compare ranked chunks and latency side by side.

Open `eval-results/chunking-comparison.md`. State only the committed baseline figures: 56 cases, 100% faithfulness, 100% citation accuracy, and semantic-chunk source MRR improving from 78.8% dense-only to 80.1% hybrid.

## 3:05–3:35 — Engineering close

Briefly show `/docs`, the passing GitHub Actions run, and the Docker Compose services.

Close with: “The system runs without paid keys using quantized ONNX models and Ollama, while provider interfaces keep OpenAI-compatible embeddings and generation optional. The next production step would be tenant-aware document ACL filtering before fusion.”

## Recording checklist

- Pull `llama3.2:3b` before recording so the first answer does not wait for a model download.
- Seed only the documents needed for each scene so retrieval traces remain readable.
- Keep browser zoom at 90–100% and terminal font at least 16 px.
- Hide notifications, local paths containing personal information, and all environment values.
- Do one silent rehearsal and keep the final take below four minutes.
