import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag")

INIT_SQL = """
-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Parent Documents Table
CREATE TABLE IF NOT EXISTS parent_documents (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    section_title VARCHAR(500),
    chunk_type VARCHAR(50) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create Child Chunks Table (768 dimensions for Gemini MRL Embeddings)
CREATE TABLE IF NOT EXISTS child_chunks (
    id UUID PRIMARY KEY,
    parent_id UUID NOT NULL REFERENCES parent_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    token_count INT,
    embedding vector(768),
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create HNSW Vector Index for Dense Cosine Search
CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_hnsw 
ON child_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. Create GIN Index for Sparse Full-Text Search
CREATE INDEX IF NOT EXISTS idx_child_chunks_tsv 
ON child_chunks 
USING gin (tsv_content);

-- 6. Index Foreign Key for fast joins
CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id 
ON child_chunks (parent_id);
"""


def initialize_database():
    print("[DB] Initializing database schema, pgvector, and indexes...")
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(INIT_SQL)
    print("[DB] Schema with vector(768) and indexes successfully initialized.")


if __name__ == "__main__":
    initialize_database()