import os
import time
from typing import List
import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models.schemas import ParentDocument, ChildChunk

load_dotenv()


class VectorStore:
    def __init__(self):
        self.db_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-embedding-001"
        self.output_dim = 768  # Matryoshka dimension compatible with pgvector HNSW

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Embeds texts in batches using gemini-embedding-001 with 768 output dimensions.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        batch_size = 32  # Batch size to manage API payload and rate limits

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dim,
                    task_type=task_type
                )
            )
            for emb in response.embeddings:
                embeddings.append(emb.values)

            if len(texts) > batch_size:
                time.sleep(0.2)  # Short pause between batches for rate limits

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query with RETRIEVAL_QUERY task type."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=[query],
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dim,
                task_type="RETRIEVAL_QUERY"
            )
        )
        return response.embeddings[0].values

    def insert_documents(
        self, 
        parents: List[ParentDocument], 
        children: List[ChildChunk]
    ):
        """Transactionally inserts parents and embedded child chunks into PostgreSQL."""
        if not parents:
            return

        print(f"[Gemini] Generating embeddings for {len(children)} child chunks...")
        child_texts = [child.content for child in children]
        embeddings = self.embed_texts(child_texts, task_type="RETRIEVAL_DOCUMENT")

        for child, emb in zip(children, embeddings):
            child.embedding = emb

        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # 1. Insert Parent Documents
                parent_records = [
                    (
                        p.id,
                        p.content,
                        p.section_title,
                        p.chunk_type.value,
                        Jsonb(p.metadata.model_dump())
                    )
                    for p in parents
                ]
                cur.executemany(
                    """
                    INSERT INTO parent_documents (id, content, section_title, chunk_type, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    parent_records
                )

                # 2. Insert Child Chunks with 768-dim Embeddings
                child_records = [
                    (
                        c.id,
                        c.parent_id,
                        c.content,
                        c.chunk_type.value,
                        Jsonb(c.metadata),
                        c.token_count,
                        c.embedding
                    )
                    for c in children
                ]
                cur.executemany(
                    """
                    INSERT INTO child_chunks (id, parent_id, content, chunk_type, metadata, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    child_records
                )

            conn.commit()
        print(f"[Storage] Transaction committed: {len(parents)} parents & {len(children)} children saved.")