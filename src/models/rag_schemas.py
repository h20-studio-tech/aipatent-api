from pydantic import BaseModel

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