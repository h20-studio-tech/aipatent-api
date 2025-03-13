<<<<<<< HEAD
from pydantic import BaseModel

class FileProcessedError(BaseModel):
    is_processed: bool
=======
from pydantic import BaseModel

class FileProcessedError(BaseModel):
    is_processed: bool
>>>>>>> 5b6e3e1f6bb904635df1f05e870b8aeeed94cf1b
    error: str