# Enterprise Financial 10-K RAG Engine

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%7C%20pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash%20%7C%20Embeddings-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A production-grade, layout-aware Retrieval-Augmented Generation (RAG) pipeline designed for SEC Form 10-K filings, balance sheets, and narrative financial disclosures. 

Engineered with **Docling** local structural parsing, **hierarchical parent-child chunking**, **PostgreSQL (`pgvector`) hybrid search** with Reciprocal Rank Fusion (RRF), **FlashRank cross-encoder re-ranking**, and **semantic response caching**.

---

## Architecture Overview

```
[ PDF Ingestion: SEC 10-K Filings ]
                │
                ▼ (Layout-Aware Structural Parsing)
        [ IBM Docling ]
        ├── Extracted Markdown Tables (Unbroken Financial Statements)
        └── Section Header Trees (#, ##, ###)
                │
                ▼ (Hierarchical Ingestion Layer)
        [ HierarchicalChunker ]
        ├── Parent Documents: Full Sections / Tables (~800–1200 words)
        └── Child Chunks: Sliding Overlap Windows (~200 words, points to parent_id)
                │
                ▼ (Embedding Generation: Matryoshka 768-dim)
      [ Google Gemini gemini-embedding-001 ]
                │
                ▼ (Unified Data & Vector Store)
     ┌────────────────────────────────────────────────────────┐
     │                PostgreSQL 16 + pgvector                │
     │  - parent_documents: Relational contexts & tables      │
     │  - child_chunks: 768-dim HNSW Cosine Index (<=>)       │
     │  - tsv_content: Full-Text GIN Index (@@)               │
     │  - semantic_cache: Sub-15ms cached query responses     │
     └──────────────────────────┬─────────────────────────────┘
                                │
                        User Search Query
                                │
                ┌───────────────┴───────────────┐
                ▼ (Stage 1A: Dense)             ▼ (Stage 1B: Sparse)
        Top 20 via HNSW Cosine          Top 20 via ts_rank_cd
                │                               │
                └───────────────┬───────────────┘
                                ▼
                [ Reciprocal Rank Fusion (RRF, k=60) ]
                        Merged Top 15 Candidates
                                │
                                ▼ (Stage 2: Cross-Attention)
             [ FlashRank Re-ranker (ms-marco-MiniLM-L-12-v2) ]
                        Top 3 High-Precision Chunks
                                │
                                ▼ (Relational Join via parent_id)
                [ Intact Parent Documents Resolved ]
                                │
                                ▼ (Strict Context Grounding + Citations)
                [ Gemini 2.5 Flash Streaming Synthesis ]
```

---

## Key Engineering Decisions

### 1. Layout-Aware Table Preservation (Docling vs. Fixed-Size Splitters)
* **Problem:** Standard naive character or token chunkers slice financial tables across row boundaries, severing numbers from column headers and rendering financial calculations useless.
* **Solution:** Used Docling to detect document layout boundaries locally. Financial statements and footnotes remain unified as discrete Markdown tables inside dedicated `ParentDocument` records.
* **Reference:** [ADR 001: Hierarchical Chunking & Layout-Aware Parsing](docs/decisions/001-hierarchical-chunking-strategy.md)

### 2. Matryoshka 768-dim Embeddings (`gemini-embedding-001` + `pgvector`)
* **Problem:** Gemini embeddings output 3,072 dimensions by default, exceeding PostgreSQL's `pgvector` HNSW hard indexing ceiling of 2,000 dimensions.
* **Solution:** Configured Matryoshka Representation Learning (MRL) truncation to `output_dimensionality=768`. This fits within `pgvector` HNSW index limits, preserves retrieval fidelity, and saves ~75% memory footprint.
* **Reference:** [ADR 002: Hybrid Storage Engine using PostgreSQL & pgvector](docs/decisions/002-hybrid-storage-pgvector.md)

### 3. Two-Stage Hybrid Retrieval (Dense + Sparse + RRF + FlashRank)
* **Stage 1 (Candidate Generation):** Executes parallel searches:
  * Dense semantic search via HNSW cosine distance (`<=>`).
  * Sparse keyword search via PostgreSQL `tsvector` with GIN indexing (`@@`).
* **Score Fusion (RRF):** Merges disparate rank lists using Reciprocal Rank Fusion ($k=60$) without requiring score normalization.
* **Stage 2 (Re-ranking):** Applies FlashRank (`ms-marco-MiniLM-L-12-v2`) cross-encoder over the top candidates on CPU using ONNX runtime (~34MB model, <25ms latency), capturing query-document token interactions before prompt construction.
* **Reference:** [ADR 003: Two-Stage Hybrid Retrieval with RRF & Re-ranking](docs/decisions/003-two-stage-hybrid-retrieval-rrf.md)

### 4. Semantic Caching Layer
* Repeated or semantically identical queries (e.g., *"Alphabet 2024 operating income"* vs. *"What was Google's operating income for 2024?"*) probe the `semantic_cache` table via cosine similarity.
* Queries matching with similarity $\ge 0.95$ return in **<15ms** with zero LLM inference cost.
* **Reference:** [ADR 004: Semantic Caching via pgvector & Grounded Generation](docs/decisions/004-semantic-caching-grounded-generation.md)

---

## Benchmark & Evaluation

Evaluated against the **RAG Triad** using a native schema-enforced LLM-as-a-Judge benchmark on Gemini 2.5 Flash, running across domain golden test cases from SEC 10-K filings.

| Metric | Score | Target | Interpretation |
| :--- | :---: | :---: | :--- |
| **Faithfulness (Groundedness)** | **1.0000** | $\ge 0.90$ | **Zero hallucination.** 100% of factual assertions generated are directly entailed by retrieved parent contexts. |
| **Answer Relevancy** | **0.8000** | $\ge 0.85$ | Answers directly address questions with minimal extraneous text. |
| **Context Recall** | **0.7310** | $\ge 0.70$ | Successfully retrieves ~73% of all reference ground-truth statements into the top 3 parent contexts. |

*Detailed per-query scoring and audit logs are documented in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).*

---

## Repository Structure

```text
├── data/
│   └── raw/                       # Uploaded and parsed PDF documents
├── docs/
│   ├── decisions/                 # Architecture Decision Records (ADRs)
│   │   ├── 001-hierarchical-chunking-strategy.md
│   │   ├── 002-hybrid-storage-pgvector.md
│   │   ├── 003-two-stage-hybrid-retrieval-rrf.md
│   │   ├── 004-semantic-caching-grounded-generation.md
│   │   └── 005-offline-evaluation-benchmarking.md
│   └── BENCHMARK_RESULTS.md       # RAG Triad evaluation scorecard
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI endpoints (SSE streaming & ingestion)
│   ├── evaluation/
│   │   └── evaluator.py           # Native LLM-as-a-Judge test runner
│   ├── generation/
│   │   └── generator.py           # Grounded Gemini 2.5 Flash streaming
│   ├── ingestion/
│   │   ├── parser.py              # Docling layout & table parser
│   │   └── chunker.py             # Hierarchical parent-child splitter
│   ├── models/
│   │   ├── schemas.py             # Pydantic schemas (Parents, Children)
│   │   └── retrieval.py           # Retrieval data models
│   ├── storage/
│   │   ├── schema.py              # PostgreSQL DDL, HNSW & GIN indexes
│   │   ├── vector_store.py        # Gemini MRL embeddings & DB transactions
│   │   └── cache.py               # Semantic cache lookups via pgvector
│   └── ui/
│       └── app.py                 # Streamlit chat interface with citations
├── tests/
│   ├── data/
│   │   └── golden_dataset.json    # Verified evaluation benchmark set
│   ├── test_ingest.py             # Ingestion smoke tests
│   ├── test_storage.py            # PostgreSQL vector storage tests
│   ├── test_retrieval.py          # Hybrid search & re-ranking verification
│   └── test_generation.py         # End-to-end generation & cache tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites
* Python 3.11+
* PostgreSQL 15+ with the `pgvector` extension enabled
* Google Gemini API Key

### 1. Environment Setup

Clone the repository and create a virtual environment:

```bash
git clone [https://github.com/] https://github.com/aftabnadeem/financial-enterprise-rag.git
cd hierarchical-rag-engine

python3 -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
DATABASE_URL="postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Initialize Database Schema

Ensure `pgvector` is enabled on your PostgreSQL instance, then run the schema migration to create the tables and indexes (HNSW, GIN):

```bash
python -m src.storage.schema
```

### 4. Run the Pipeline Services

#### Start the FastAPI Server (Backend)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive OpenAPI documentation: `http://localhost:8000/docs`

#### Start the Streamlit Dashboard (Frontend)
In a separate terminal:
```bash
streamlit run src/ui/app.py
```
* Web interface: `http://localhost:8501`

---

## API Endpoints

### 1. Ingest SEC 10-K Filing
* **Endpoint:** `POST /api/ingest`
* **Form Data:**
  * `file`: Multipart PDF upload
  * `company_name`: e.g. `"Alphabet Inc."`
  * `fiscal_year`: e.g. `"2024"`
* **Response:**
  ```json
  {
    "status": "success",
    "filename": "goog-10k-2024.pdf",
    "company_name": "Alphabet Inc.",
    "parent_documents_created": 48,
    "child_chunks_indexed": 215
  }
  ```

### 2. Stream Grounded Query (Server-Sent Events)
* **Endpoint:** `POST /api/chat/stream`
* **Payload:**
  ```json
  {
    "query": "What were Alphabet's cloud revenues and operating margins?",
    "top_k": 3
  }
  ```
* **Response:** Stream of SSE events yielding incremental response tokens and section-level attribution.

---

## Running Offline Evaluation

Execute the automated LLM-as-a-Judge benchmark:

```bash
python -m src.evaluation.evaluator
```

This runs test queries from `tests/data/golden_dataset.json`, scores Faithfulness, Relevancy, and Context Recall using Gemini 2.5 Flash, and regenerates `docs/BENCHMARK_RESULTS.md`.

---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.