from typing import Optional, Dict, Any
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    child_id: str
    parent_id: str
    content: str
    chunk_type: str
    metadata: Dict[str, Any] = {}
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    parent_content: Optional[str] = None
    parent_section_title: Optional[str] = None