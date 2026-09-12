# ADR 004: Semantic Caching via pgvector and Strictly Grounded Gemini Synthesis

## Status
Accepted

## Context
Financial queries exhibit high semantic repetition (e.g., analysts querying identical metrics with varied phrasing). Hitting full-text search, HNSW retrieval, FlashRank re-ranking, and LLM inference for redundant requests creates unnecessary latency (1.5s–3.5s) and accumulates token costs. Furthermore, financial analysis demands zero tolerance for hallucination; answers must be strictly grounded with traceable citations to filing sections.

## Decision
1. **Semantic Cache Storage:** We utilize a PostgreSQL table (`semantic_cache`) indexed with HNSW (`vector_cosine_ops`, 768 dimensions) rather than provisioning a separate Redis cache.
2. **Cache Lookup Threshold:** Incoming queries are embedded via `gemini-embedding-001` (768 dimensions) and probed against `semantic_cache`. If cosine similarity $\ge 0.95$, the cached answer is served directly.
3. **Synthesis Model:** Google `gemini-2.5-flash` via the official `google-genai` SDK using `generate_content_stream`. It offers low-latency generation and structured context adherence.
4. **Strict Grounding Protocol:** The system prompt restricts synthesis exclusively to retrieved parent contexts. Unmentioned metrics must trigger an explicit refusal rather than speculative generation. Chunks require attribution through bracketed section citations.

## Consequences
### Positive
* Semantic cache hits reduce end-to-end latency from ~2,500ms to <15ms.
* Zero external cache infrastructure: PostgreSQL handles relational data, vectors, full-text indexes, and semantic caching in a single instance.
* Verifiable responses: Citations reference physical sections of the filing.

### Negative & Trade-offs
* Queries probing rapidly updating data risk serving stale answers unless a cache TTL or invalidation strategy is enforced. (Mitigated via `created_at` timestamping).