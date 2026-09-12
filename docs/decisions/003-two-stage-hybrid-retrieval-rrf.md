# ADR 003: Two-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF) and FlashRank Re-ranking

## Status
Accepted

## Context
Dense semantic search alone struggles with precision queries containing fiscal dates, numerical indicators, and acronyms in financial filings. Conversely, sparse keyword search misses contextual synonyms. Furthermore, bi-encoder embeddings evaluate queries and passages independently, missing fine-grained cross-token interactions.

## Decision
We implement a **Two-Stage Hybrid Retrieval & Fusion Pipeline**:
1. **Stage 1A (Dense Retrieval):** Retrieve top $K_1=20$ child chunks using Gemini `gemini-embedding-001` (768-dim) via PostgreSQL HNSW cosine distance (`<=>`).
2. **Stage 1B (Sparse Retrieval):** Retrieve top $K_2=20$ child chunks using PostgreSQL full-text search (`tsvector @@ plainto_tsquery`) ranked by `ts_rank_cd`.
3. **Score Fusion (RRF):** Combine dense and sparse candidate pools using Reciprocal Rank Fusion with smoothing factor $k=60$. Deduplicate and take the top 15 fused candidates.
4. **Stage 2 (Cross-Encoder Re-ranking):** Pass the fused candidates through `FlashRank` (`ms-marco-MiniLM-L-12-v2`). This performs full cross-attention over the `(query, passage)` pair on CPU, ranking candidates by deep contextual relevance.
5. **Parent Resolution:** Extract top $N=3$ re-ranked child chunks and resolve their `parent_id` foreign keys to return complete, unbroken parent contexts (tables/sections) for generation.

## Consequences
### Positive
* Robust against both vocabulary mismatch and semantic drift.
* Deterministic scoring normalization across vector and lexical results via RRF.
* High precision: Cross-encoder removes false-positive semantic hits before prompt synthesis.
* CPU-friendly: FlashRank uses ONNX runtime without heavy PyTorch dependencies.

### Negative & Trade-offs
* Retrieval involves two database queries plus an in-memory cross-encoder pass, adding ~25-40ms to total retrieval latency compared to single vector search.