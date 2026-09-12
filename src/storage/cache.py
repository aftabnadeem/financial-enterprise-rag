import os
import uuid
from typing import Optional, Tuple
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()


class SemanticCache:
    """
    Embedding-based semantic cache using PostgreSQL pgvector.
    Caches LLM responses and retrieves them for semantically identical queries.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
        )
        self.threshold = similarity_threshold

    def get(self, query_vector: list[float]) -> Optional[Tuple[str, float]]:
        """
        Probes the cache for queries with cosine similarity >= threshold.
        Returns: (cached_response, similarity_score) or None.
        """
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        response_text,
                        1 - (query_vector <=> %s::vector) AS similarity
                    FROM semantic_cache
                    ORDER BY query_vector <=> %s::vector
                    LIMIT 1;
                    """,
                    (query_vector, query_vector)
                )
                row = cur.fetchone()

                if row and row[1] is not None and row[1] >= self.threshold:
                    return row[0], float(row[1])
        return None

    def set(self, query_text: str, query_vector: list[float], response_text: str):
        """Stores a generated response and its query vector into the cache."""
        cache_id = str(uuid.uuid4())
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_cache (id, query_text, query_vector, response_text)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (cache_id, query_text, query_vector, response_text)
                )
            conn.commit()