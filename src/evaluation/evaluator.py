import os
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import GroundedGenerator

load_dotenv()


# =====================================================================
# Pydantic Schemas for Structured LLM-as-a-Judge Evaluations
# =====================================================================

class FaithfulnessReport(BaseModel):
    claims: List[str] = Field(
        description="All individual factual assertions made in the generated answer."
    )
    supported_claims: List[str] = Field(
        description="Claims that are directly supported and verifiable by the retrieved context."
    )
    unsupported_claims: List[str] = Field(
        description="Claims that cannot be verified from the context (hallucinations or assumptions)."
    )
    score: float = Field(
        description="Ratio of supported_claims over total claims (0.0 to 1.0). 1.0 = zero hallucination."
    )
    reasoning: str = Field(
        description="Concise justification of the faithfulness score."
    )


class RelevancyReport(BaseModel):
    score: float = Field(
        description="Score between 0.0 and 1.0 evaluating how directly and comprehensively the answer addresses the user prompt."
    )
    reasoning: str = Field(
        description="Concise justification explaining the score."
    )


class ContextRecallReport(BaseModel):
    ground_truth_statements: List[str] = Field(
        description="Key factual claims required to constitute a complete answer based on ground truth."
    )
    retrieved_statements: List[str] = Field(
        description="Ground truth statements successfully found within the retrieved context."
    )
    score: float = Field(
        description="Ratio of retrieved_statements over ground_truth_statements (0.0 to 1.0)."
    )
    reasoning: str = Field(
        description="Explanation of whether the retrieved contexts captured the ground truth."
    )


# =====================================================================
# Native LLM-as-a-Judge Pipeline
# =====================================================================

class NativePipelineEvaluator:
    """
    Evaluates the RAG pipeline using Gemini 3.5 Flash with structured Pydantic outputs.
    Measures the RAG Triad: Faithfulness, Answer Relevancy, and Context Recall.
    """

    def __init__(self, golden_dataset_path: str = "tests/data/golden_dataset.json"):
        self.dataset_path = Path(golden_dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")

        self.client = genai.Client(api_key=api_key)
        self.judge_model = "gemini-3.5-flash"

        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()

    def evaluate_faithfulness(self, context: str, answer: str) -> FaithfulnessReport:
        prompt = f"""
You are an expert auditor evaluating factual hallucination in a financial question-answering system.
Given the following retrieved context and generated answer:
1. Break down the generated answer into its atomic factual claims.
2. Check whether each claim is strictly entailed and supported by the retrieved context.
3. Compute the score as: len(supported_claims) / len(total_claims). If no factual claims exist, score = 1.0.

Retrieved Context:
\"\"\"{context}\"\"\"

Generated Answer:
\"\"\"{answer}\"\"\"
"""
        response = self.client.models.generate_content(
            model=self.judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FaithfulnessReport,
                temperature=0.0
            )
        )
        return FaithfulnessReport.model_validate_json(response.text)

    def evaluate_relevancy(self, question: str, answer: str) -> RelevancyReport:
        prompt = f"""
You are an expert evaluator evaluating answer relevancy.
Assess how directly, specifically, and concisely the generated answer addresses the user's question without digression.

User Question:
\"\"\"{question}\"\"\"

Generated Answer:
\"\"\"{answer}\"\"\"
"""
        response = self.client.models.generate_content(
            model=self.judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RelevancyReport,
                temperature=0.0
            )
        )
        return RelevancyReport.model_validate_json(response.text)

    def evaluate_recall(self, ground_truth: str, context: str) -> ContextRecallReport:
        prompt = f"""
You are an expert retrieval evaluator.
Assess whether the retrieved context contains all key facts required by the ground-truth reference answer.
1. Extract key facts from the ground truth answer.
2. Determine which of those facts are present in the retrieved context.
3. Compute the recall score as: len(retrieved_statements) / len(ground_truth_statements).

Ground Truth Reference:
\"\"\"{ground_truth}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"
"""
        response = self.client.models.generate_content(
            model=self.judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContextRecallReport,
                temperature=0.0
            )
        )
        return ContextRecallReport.model_validate_json(response.text)

    def run_benchmark(self, output_markdown: str = "docs/BENCHMARK_RESULTS.md"):
        print(f"[Benchmark] Loading test cases from {self.dataset_path}...")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        reports = []
        print(f"[Benchmark] Evaluating {len(test_cases)} cases via Gemini structured judge...")

        for i, case in enumerate(test_cases, start=1):
            q = case["question"]
            gt = case["ground_truth"]
            print(f"\n[{i}/{len(test_cases)}] Evaluating Query: '{q[:60]}...'")

            # 1. Retrieve
            chunks = self.retriever.retrieve(query=q, top_k=3)
            context = "\n\n".join([c.parent_content or c.content for c in chunks])

            # 2. Fresh generation (bypassing semantic cache)
            prompt = self.generator._build_context_prompt(q, chunks)
            resp = self.generator.client.models.generate_content(
                model=self.generator.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            answer = resp.text or ""

            # 3. Judge Triad Metrics
            print("  -> Scoring Faithfulness...")
            faith_rep = self.evaluate_faithfulness(context=context, answer=answer)
            print(f"     Faithfulness Score: {faith_rep.score:.2f} ({len(faith_rep.supported_claims)}/{len(faith_rep.claims)} claims supported)")

            print("  -> Scoring Answer Relevancy...")
            rel_rep = self.evaluate_relevancy(question=q, answer=answer)
            print(f"     Relevancy Score:    {rel_rep.score:.2f}")

            print("  -> Scoring Context Recall...")
            recall_rep = self.evaluate_recall(ground_truth=gt, context=context)
            print(f"     Context Recall:     {recall_rep.score:.2f}")

            reports.append({
                "question": q,
                "ground_truth": gt,
                "answer": answer,
                "faithfulness": faith_rep.score,
                "relevancy": rel_rep.score,
                "context_recall": recall_rep.score,
                "faith_reason": faith_rep.reasoning,
                "rel_reason": rel_rep.reasoning,
                "recall_reason": recall_rep.reasoning,
            })

        # Summary averages
        avg_faith = sum(r["faithfulness"] for r in reports) / len(reports)
        avg_rel = sum(r["relevancy"] for r in reports) / len(reports)
        avg_rec = sum(r["context_recall"] for r in reports) / len(reports)

        print("\n" + "=" * 55)
        print("          RAG PIPELINE BENCHMARK SUMMARY")
        print("=" * 55)
        print(f" Average Faithfulness (Groundedness) : {avg_faith:.4f}")
        print(f" Average Answer Relevancy            : {avg_rel:.4f}")
        print(f" Average Context Recall              : {avg_rec:.4f}")
        print("=" * 55)

        # Generate docs/BENCHMARK_RESULTS.md
        Path("docs").mkdir(exist_ok=True)
        with open(output_markdown, "w", encoding="utf-8") as f:
            f.write("# RAG Pipeline Benchmark Results\n\n")
            f.write("Automated LLM-as-a-Judge benchmark evaluated across the **RAG Triad** using **Gemini 3.5 Flash** with schema-enforced Pydantic contracts.\n\n")
            f.write("### Aggregate Benchmark Summary\n\n")
            f.write("| Evaluation Metric | Score | Target | Status |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **Faithfulness (Zero-Hallucination)** | **{avg_faith:.4f}** | $\\ge 0.90$ | {'✅ Pass' if avg_faith >= 0.90 else '⚠️ Review'} |\n")
            f.write(f"| **Answer Relevancy** | **{avg_rel:.4f}** | $\\ge 0.85$ | {'✅ Pass' if avg_rel >= 0.85 else '⚠️ Review'} |\n")
            f.write(f"| **Context Recall** | **{avg_rec:.4f}** | $\\ge 0.80$ | {'✅ Pass' if avg_rec >= 0.80 else '⚠️ Review'} |\n\n")
            f.write("### Per-Query Diagnostic Breakdown\n\n")
            f.write("| Query | Faithfulness | Relevancy | Context Recall |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for r in reports:
                f.write(f"| {r['question']} | `{r['faithfulness']:.2f}` | `{r['relevancy']:.2f}` | `{r['context_recall']:.2f}` |\n")
            f.write("\n### Audit Trail & Reasoning\n\n")
            for i, r in enumerate(reports, start=1):
                f.write(f"#### Query {i}: {r['question']}\n\n")
                f.write(f"- **Generated Answer:** {r['answer']}\n")
                f.write(f"- **Faithfulness Rationale:** {r['faith_reason']}\n")
                f.write(f"- **Relevancy Rationale:** {r['rel_reason']}\n")
                f.write(f"- **Recall Rationale:** {r['recall_reason']}\n\n")

        print(f"\n[Benchmark] Detailed report saved to {output_markdown}")


if __name__ == "__main__":
    NativePipelineEvaluator().run_benchmark()