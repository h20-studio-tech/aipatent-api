from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: int
    text: str
    page_number: int
    filename: str