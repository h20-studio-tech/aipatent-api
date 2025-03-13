<<<<<<< HEAD
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: int
    text: str
    page_number: int
=======
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: int
    text: str
    page_number: int
>>>>>>> 5b6e3e1f6bb904635df1f05e870b8aeeed94cf1b
    filename: str