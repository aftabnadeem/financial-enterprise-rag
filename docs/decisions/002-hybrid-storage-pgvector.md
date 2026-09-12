# ADR 002: Hybrid Storage Engine using PostgreSQL, pgvector (HNSW), and Gemini Embeddings

## Status
Accepted

## Context
We require dense semantic vector search coupled with exact lexical keyword search (full-text `tsvector`). Rather than maintaining an isolated vector database, we maintain data locality by storing child vectors, parent documents, and text search indexes within PostgreSQL using the `pgvector` extension.

For the embedding model, we utilize Google's `gemini-embedding-001`.

## Decision
1. **Embedding Model:** Google `gemini-embedding-001` via the official `google-genai` SDK.
2. **Matryoshka Truncation (768 Dimensions):** Default Gemini embeddings produce 3,072 dimensions, exceeding `pgvector`'s HNSW index limit of 2,000 dimensions. We configure `output_dimensionality=768` using Matryoshka Representation Learning (MRL). This stays well within HNSW limits and drastically cuts index memory footprint while preserving semantic fidelity.
3. **Vector Index:** HNSW on cosine distance (`vector_cosine_ops`) with parameters `m = 16` and `ef_construction = 64`.
4. **Lexical Index:** Native PostgreSQL `tsvector` with a `GIN` index on `child_chunks.content`.
5. **Relational Integrity:** Foreign key with `ON DELETE CASCADE` binding `child_chunks` to `parent_documents`.

## Consequences
* **Positive:** Enterprise-grade multilingual semantic embeddings via Gemini API.
* **Positive:** Single PostgreSQL instance handles vectors, relational joins, and BM25-style lexical search.
* **Trade-off:** Ingestion requires an active Google Gemini API key and network connectivity.