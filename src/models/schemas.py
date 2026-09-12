from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

class ChunkType(str, Enum):
    PROSE = "prose"
    TABLE = "table"

class DocumentMetadata(BaseModel):
    source_filename: str
    company_name: str
    fiscal_year: Optional[str] = None
    total_pages: Optional[int] = None

class ParentDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata
    token_count: Optional[int] = None
    chunk_type: ChunkType = ChunkType.PROSE
    section_title: Optional[str] = None

class ChildChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str
    content: str
    chunk_type: ChunkType = ChunkType.PROSE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    embedding: Optional[list[float]] = None  # Populated in Phase 2