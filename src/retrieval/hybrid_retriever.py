import os
from typing import List, Dict
import psycopg
from pgvector.psycopg import register_vector
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

from src.storage.vector_store import VectorStore
from src.models.retrieval import RetrievedChunk

load_dotenv()


class HybridRetriever:
    """
    Two-stage retrieval engine:
    1. Dense Vector Search (Gemini MRL 768-dim) + Sparse Full-Text (tsvector)
    2. Reciprocal Rank Fusion (RRF)
    3. Cross-Encoder Re-ranking via FlashRank
    4. Parent Document Context Resolution
    """

    def __init__(self, rrf_k: int = 60):
        self.db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
        )
        self.rrf_k = rrf_k
        self.vector_store = VectorStore()
        
        # Lightweight ONNX cross-encoder model (~34MB)
        print("[Retriever] Initializing FlashRank Cross-Encoder (ms-marco-MiniLM-L-12-v2)...")
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def _dense_search(self, conn, query_vector: List[float], limit: int = 20) -> List[RetrievedChunk]:
        """Performs approximate nearest neighbor search via HNSW cosine distance."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id, 
                    parent_id, 
                    content, 
                    chunk_type, 
                    metadata
                FROM child_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, limit)
            )
            rows = cur.fetchall()

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                RetrievedChunk(
                    child_id=str(row[0]),
                    parent_id=str(row[1]),
                    content=row[2],
                    chunk_type=row[3],
                    metadata=row[4],
                    dense_rank=rank
                )
            )
        return results

    def _sparse_search(self, conn, query: str, limit: int = 20) -> List[RetrievedChunk]:
        """Performs lexical full-text search using PostgreSQL tsvector and ts_rank_cd."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id, 
                    parent_id, 
                    content, 
                    chunk_type, 
                    metadata,
                    ts_rank_cd(tsv_content, plainto_tsquery('english', %s)) AS rank_score
                FROM child_chunks
                WHERE tsv_content @@ plainto_tsquery('english', %s)
                ORDER BY rank_score DESC
                LIMIT %s;
                """,
                (query, query, limit)
            )
            rows = cur.fetchall()

        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                RetrievedChunk(
                    child_id=str(row[0]),
                    parent_id=str(row[1]),
                    content=row[2],
                    chunk_type=row[3],
                    metadata=row[4],
                    sparse_rank=rank
                )
            )
        return results

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[RetrievedChunk], 
        sparse_results: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Fuses rankings using RRF: score = 1 / (k + rank)."""
        fused: Dict[str, RetrievedChunk] = {}

        # Process dense ranks
        for item in dense_results:
            fused[item.child_id] = item
            fused[item.child_id].rrf_score += 1.0 / (self.rrf_k + item.dense_rank)

        # Process sparse ranks
        for item in sparse_results:
            if item.child_id in fused:
                fused[item.child_id].sparse_rank = item.sparse_rank
                fused[item.child_id].rrf_score += 1.0 / (self.rrf_k + item.sparse_rank)
            else:
                fused[item.child_id] = item
                fused[item.child_id].rrf_score += 1.0 / (self.rrf_k + item.sparse_rank)

        # Sort descending by RRF score
        sorted_fused = sorted(fused.values(), key=lambda x: x.rrf_score, reverse=True)
        return sorted_fused

    def _resolve_parent_contexts(self, conn, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Hydrates top child chunks with their complete parent documents."""
        if not chunks:
            return []

        parent_ids = list({c.parent_id for c in chunks})
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, section_title
                FROM parent_documents
                WHERE id = ANY(%s::uuid[]);
                """,
                (parent_ids,)
            )
            parent_lookup = {str(row[0]): (row[1], row[2]) for row in cur.fetchall()}

        for c in chunks:
            if c.parent_id in parent_lookup:
                c.parent_content, c.parent_section_title = parent_lookup[c.parent_id]

        return chunks

    def retrieve(
        self, 
        query: str, 
        stage1_limit: int = 20, 
        top_k: int = 3
    ) -> List[RetrievedChunk]:
        """
        Executes end-to-end two-stage hybrid retrieval:
        Vector + Lexical -> RRF -> Cross-Encoder -> Parent Document Resolution
        """
        # 1. Generate query vector using Gemini API
        query_vector = self.vector_store.embed_query(query)

        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)

            # 2. Stage 1: Parallel candidate generation (Dense + Sparse)
            dense_candidates = self._dense_search(conn, query_vector, limit=stage1_limit)
            sparse_candidates = self._sparse_search(conn, query, limit=stage1_limit)

            # 3. Reciprocal Rank Fusion
            fused_candidates = self._reciprocal_rank_fusion(dense_candidates, sparse_candidates)

            # Take top candidates for re-ranking
            candidates_to_rerank = fused_candidates[:15]
            if not candidates_to_rerank:
                return []

            # 4. Stage 2: Cross-Encoder Re-ranking via FlashRank
            passages = [
                {"id": c.child_id, "text": c.content, "metadata": c.metadata}
                for c in candidates_to_rerank
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.reranker.rerank(rerank_request)

            # Map FlashRank scores back to candidate chunks
            rerank_score_map = {item["id"]: item["score"] for item in reranked_results}
            for c in candidates_to_rerank:
                c.rerank_score = rerank_score_map.get(c.child_id, 0.0)

            # Sort by cross-encoder score and take top_k
            top_ranked = sorted(candidates_to_rerank, key=lambda x: x.rerank_score, reverse=True)[:top_k]

            # 5. Resolve full parent documents
            final_results = self._resolve_parent_contexts(conn, top_ranked)

        return final_results