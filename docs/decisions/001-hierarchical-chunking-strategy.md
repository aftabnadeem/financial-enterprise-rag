# ADR 001: Hierarchical Chunking & Layout-Aware Parsing via Docling

## Status
Accepted

## Context
Financial documents (such as SEC 10-K filings and annual reports) combine dense narrative disclosures with multi-column financial statements and balance sheets. Processing these documents via naive fixed-window token splitters breaks table row-column bindings and decouples numbers from their headers.

Furthermore, relying on remote SaaS document partition APIs introduces external failure modes: authentication schema divergence, endpoint deprecation, payload file-size limits, and network latency.

## Decision
We implement a **Local, Layout-Aware Hierarchical Ingestion Pipeline** using **Docling**:

1. **Parser Layer (Docling):**
   * Use IBM's Docling document conversion engine locally.
   * Preserve document structure natively: headings, section hierarchies, and financial tables are converted directly into clean Markdown format.
   * Zero dependence on remote external parsing APIs or cloud credentials.

2. **Hierarchical (Parent-Child) Chunking Strategy:**
   * **Parent Documents:** Group consecutive blocks under logical section headers into contextual units (~800–1,200 tokens). Tables are extracted as intact, discrete Parent units in Markdown format.
   * **Child Chunks:** Subdivide Parents into smaller, overlapping windows (~200–250 tokens) referencing the parent via `parent_id`.
   * **Retrieval Protocol:** Vector indexing (`pgvector`) indexes Child Chunks for high-precision semantic matching. At query time, retrieved children are mapped back to their `ParentDocument` so the generator LLM receives complete, unbroken context.

## Consequences
### Positive
* **Deterministic & Offline:** No API keys, rate limits, or network timeouts during ingestion.
* **Preserved Table Semantics:** Multi-year balance sheets and income statements remain intact as structured Markdown tables.
* **High Retrieval Precision:** Eliminates vector dilution by indexing compact child chunks while supplying complete parent contexts for LLM generation.

### Negative & Trade-offs
* Initial run requires downloading local model weights for document layout analysis.
* Join overhead between Child and Parent records during retrieval (resolved in PostgreSQL via foreign keys).