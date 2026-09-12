import os
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector

from src.ingestion.parser import DoclingDocumentParser
from src.ingestion.chunker import HierarchicalChunker
from src.storage.schema import initialize_database
from src.storage.vector_store import VectorStore


def test_gemini_storage():
    # 1. Initialize DB schema and vector(768) column
    initialize_database()

    # 2. Parse & Chunk with Docling
    pdf_path = Path("data/raw/sample.pdf")
    parser = DoclingDocumentParser()
    chunker = HierarchicalChunker()

    parsed = parser.parse_pdf(pdf_path)
    parents, children = chunker.chunk_document(
        parsed_data=parsed,
        company_name="Alphabet Inc.",
        fiscal_year="2024"
    )

    # 3. Embed with Gemini & persist to pgvector
    store = VectorStore()
    store.insert_documents(parents, children)

    # 4. Perform a sample semantic vector similarity query
    query = "Total revenues and net income performance"
    print(f"\n[Gemini] Embedding search query: '{query}'...")
    query_vector = store.embed_query(query)

    db_url = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag")
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    c.id AS child_id,
                    c.content AS child_content,
                    1 - (c.embedding <=> %s::vector) AS cosine_similarity,
                    p.section_title AS parent_section,
                    p.content AS parent_content
                FROM child_chunks c
                JOIN parent_documents p ON c.parent_id = p.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT 1;
                """,
                (query_vector, query_vector)
            )
            row = cur.fetchone()

            print("\n================ VECTOR SEARCH RESULT ================")
            print(f"Top Matched Child ID : {row[0]}")
            print(f"Cosine Similarity    : {row[2]:.4f}")
            print(f"Parent Section Title : {row[3]}")
            print(f"\n[Child Snippet]:\n{row[1][:150]}...")
            print(f"\n[Resolved Parent Context Preview]:\n{row[4][:250]}...")
            print("======================================================")


if __name__ == "__main__":
    test_gemini_storage()