# ADR 005: Deterministic LLM-as-a-Judge Offline Evaluation Suite

## Status
Accepted

## Context
RAG pipelines can fail independently across retrieval (missing passages, low rank precision) and generation (hallucination, unfaithful extrapolation). 

While open-source aggregation wrappers (like Ragas) popularize the RAG Triad, in practice their deep transitive dependency trees (coupling langchain, instructor, and specific packaging versions) create frequent runtime breakage. In production, we require reproducible, deterministic metrics with zero dependency friction.

## Decision
We implement a **Native, Schema-Enforced LLM-as-a-Judge Evaluator** using Gemini 3.5 Flash and Pydantic:
1. **RAG Triad Metrics:**
   * **Faithfulness (Groundedness):** Extracts atomic factual claims from the synthesized response and validates whether each claim is logically entailed by the retrieved parent context. Returns a normalized score: $\frac{\text{Supported Claims}}{\text{Total Claims}}$.
   * **Answer Relevancy:** Measures whether the generated answer directly addresses the user query without conversational drift or fluff.
   * **Context Recall:** Evaluates whether critical domain facts in the ground-truth benchmark were successfully retrieved in the top-$K$ candidate passages.
2. **Deterministic Output:** Uses Gemini's native `response_schema` with Pydantic models to guarantee typed JSON responses containing both numeric scores and audit-trail reasoning.
3. **Automated Reporting:** Generates a structured Markdown scorecard (`docs/BENCHMARK_RESULTS.md`) suitable for CI/CD checks and documentation.

## Consequences
### Positive
* **Zero Dependency Overhead:** Relies entirely on `google-genai` and `pydantic`—no external evaluation frameworks or LangChain wrappers required.
* **Explainability:** Returns explicit lists of supported vs. unsupported claims for debugging hallucination sources.
* **Cost & Speed:** Evaluation runs concurrently on Gemini 2.5 Flash in sub-minute execution times.



### Validation Results
* **Date Evaluated:** 12 September 2026
* **Judge:** Gemini 3.5 Flash (Schema-enforced JSON structured output)
* **Average Faithfulness:** 1.0000 (0% hallucination rate across test suite)
* **Average Answer Relevancy:** 0.8000
* **Average Context Recall:** 0.7310
* **Artifact:** Full audit trail documented in `docs/BENCHMARK_RESULTS.md`