from src.retrieval.hybrid_retriever import HybridRetriever


def test_retrieval_pipeline():
    retriever = HybridRetriever()

    # Test with a specific financial query (both keywords and conceptual intent)
    test_query = "Consolidated revenues, net income, and operating results"

    print(f"\n================ EXECUTING HYBRID RETRIEVAL ================")
    print(f"User Query: '{test_query}'")
    print("============================================================\n")

    results = retriever.retrieve(query=test_query, stage1_limit=20, top_k=3)

    for i, res in enumerate(results, start=1):
        print(f"--- [Result #{i}] ---")
        print(f"Child ID            : {res.child_id}")
        print(f"Dense Rank          : {res.dense_rank}")
        print(f"Sparse (BM25) Rank  : {res.sparse_rank}")
        print(f"RRF Score           : {res.rrf_score:.5f}")
        print(f"Cross-Encoder Score : {res.rerank_score:.5f}")
        print(f"Parent Section      : {res.parent_section_title}")
        print(f"Child Snippet       :\n{res.content[:150]}...")
        if res.parent_content:
            print(f"Resolved Parent Preview:\n{res.parent_content[:200]}...\n")


if __name__ == "__main__":
    test_retrieval_pipeline()