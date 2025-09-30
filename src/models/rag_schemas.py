from pydantic import BaseModel, Field
from typing import List

class RetrievalRequest(BaseModel):
    query: str
    target_files: list[str, str]
    
    
class RetrievalResponse(BaseModel):
    status: str
    message: str
    data: list
    
class Chunk(BaseModel):
    chunk_id: int
    text: str
    page_number: int
    filename: str


class ChunksByIdsRequest(BaseModel):
    """Request model for retrieving chunks by their IDs."""
    chunk_ids: List[int] = Field(..., description="List of chunk IDs to retrieve")
    document_names: List[str] = Field(..., description="List of document names (without .pdf extension)")


class ChunksByIdsResponse(BaseModel):
    """Response model for chunks retrieved by IDs."""
    status: str = Field(..., description="Status of the response (success/error)")
    message: str = Field(..., description="Status message")
    chunks: List[Chunk] = Field(..., description="List of retrieved chunks")
    count: int = Field(..., description="Number of chunks retrieved")