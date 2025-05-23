from pydantic import BaseModel
from typing import Optional

class FileProcessedError(Exception):
    """Exception raised when a file has already been processed or encountered an error during processing.
    
    This exception preserves the original error and details to aid in debugging.
    """
    def __init__(self, file_id: Optional[str], filename: Optional[str], is_processed: bool, error: str, original_error=None):
        self.file_id = file_id
        self.filename = filename
        self.is_processed = is_processed
        self.error_message = error
        self.original_error = original_error
        super().__init__(self.error_message)
    
    def __str__(self):
        if self.original_error:
            return f"{self.error_message} Original error: {self.original_error}"
        return self.error_message