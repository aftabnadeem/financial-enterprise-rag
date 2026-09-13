import os
import json
import shutil
from pathlib import Path
from typing import Generator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.ingestion.parser import DoclingDocumentParser
from src.ingestion.chunker import HierarchicalChunker
from src.storage.vector_store import VectorStore
from src.generation.generator import GroundedGenerator

app = FastAPI(
    title="Enterprise Financial RAG API",
    version="1.0.0",
    description="High-precision financial 10-K RAG pipeline with hierarchical chunking and hybrid search."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared instances
parser = DoclingDocumentParser()
chunker = HierarchicalChunker()
vector_store = VectorStore()
generator = GroundedGenerator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "enterprise-rag-api"}


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    fiscal_year: str = Form(None)
):
    """
    Ingests an SEC 10-K PDF: parses via Docling, chunks hierarchically,
    generates 768-dim Gemini embeddings, and persists to PostgreSQL.
    """
    temp_dir = Path("data/raw")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parsed_data = parser.parse_pdf(temp_path)
        parents, children = chunker.chunk_document(
            parsed_data=parsed_data,
            company_name=company_name,
            fiscal_year=fiscal_year
        )
        vector_store.insert_documents(parents, children)

        return {
            "status": "success",
            "filename": file.filename,
            "company_name": company_name,
            "parent_documents_created": len(parents),
            "child_chunks_indexed": len(children),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/chat/stream")
def stream_query(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint:
    Checks semantic cache -> Hybrid Retrieval (RRF + FlashRank) -> Streams Gemini generation.
    """
    def event_generator() -> Generator[str, None, None]:
        try:
            for token in generator.generate_stream(query=request.query, top_k=request.top_k):
                # Yield in SSE format
                data_payload = json.dumps({"token": token})
                yield f"data: {data_payload}\n\n"
        except Exception as err:
            err_payload = json.dumps({"error": str(err)})
            yield f"data: {err_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")