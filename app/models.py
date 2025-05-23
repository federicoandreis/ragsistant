from typing import List, Optional, Dict
from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    filename: str
    filetype: str
    extra: Optional[Dict] = None

class DocumentChunk(BaseModel):
    text: str
    # summary: Optional[str] = None  # Removed for community-level summaries
    claims: Optional[list] = None  # New: claims extracted per chunk

class IngestedDocument(BaseModel):
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]

class ChunkRetrievalResult(BaseModel):
    text: str
    doc_id: str
    chunk_index: int
    score: float
    metadata: Optional[dict] = None

class ChunkRetrievalResults(BaseModel):
    results: List[ChunkRetrievalResult]

class RoutingDecision(BaseModel):
    backend: str  # 'graph', 'vector', or 'hybrid'
    reason: str
    entity_match: bool = False
    entity_name: Optional[str] = None
