from pydantic import BaseModel

class FileProcessedError(BaseModel):
    is_processed: bool
    error: str