import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union 
from src.models.ocr_schemas import Embodiment, DetailedDescriptionEmbodiment
from src.models.rag_schemas import Chunk
from enum import Enum
from src.models.ocr_schemas import Glossary
class FileUploadResponse(BaseModel):
    """
    Response model for file upload operations.
    
    Attributes:
        filename: The name of the uploaded file
        message: Status message describing the result of the upload operation
        status_code: HTTP status code indicating the success or failure of the operation
    """
    filename: str = Field(..., description="The name of the uploaded file")
    message: str = Field(..., description="Status message for the upload operation")
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )


class PatentProject(BaseModel):
    """
    Represents a patent project for creation.
    
    Attributes:
        name: Name of the patent project
        antigen: Target antigen for the technology
        disease: Disease targeted by the technology
    """
    name: str = Field(..., description="Name of the patent project")
    antigen: str = Field(..., description="Target antigen for the technology")
    disease: str = Field(..., description="Disease targeted by the technology")


class PatentProjectItem(BaseModel):
    """
    Represents a patent project item retrieved from the database.
    
    Attributes:
        patent_id: Unique identifier for the patent project
        name: Name of the patent project
        antigen: Target antigen for the technology
        disease: Disease targeted by the technology
        created_at: ISO format timestamp of when the project was created
        updated_at: ISO format timestamp of when the project was last updated
    """
    patent_id: str
    name: str
    antigen: str
    disease: str
    created_at: str
    updated_at: str


class PatentProjectListResponse(BaseModel):
    """
    Response model for listing patent projects.
    
    Attributes:
        status: Status of the response (success/error)
        projects: List of patent project items retrieved from the database
    """
    status: str = Field(..., description="Status of the response (success/error)")
    projects: List[PatentProjectItem] = Field(..., description="List of patent projects")


class PatentProjectResponse(BaseModel):
    """
    Response model for patent project creation.
    
    Attributes:
        patent_id: Unique identifier for the created patent project
        message: Status message describing the result of the creation operation
        status_code: HTTP status code indicating the success or failure of the operation
    """
    patent_id: uuid.UUID
    message: str = Field(..., description="Status message for the upload operation")
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )


class PatentUploadResponse(BaseModel):
    """
    Response model for patent document upload operations.
    
    Attributes:
        filename: The name of the uploaded patent file
        message: Status message describing the result of the upload operation
        data: List of extracted embodiments from the patent document
        abstract: The extracted abstract from the patent document
        status_code: HTTP status code indicating the success or failure of the operation
    """
    filename: str = Field(..., description="The name of the uploaded file")
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    message: str = Field(..., description="Status message for the upload operation")
    data: list[Union[Embodiment, DetailedDescriptionEmbodiment]] = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )
    terms: Glossary = Field(..., description="The glossary of terms in the patent document")
    abstract: Optional[str] = Field(None, description="The extracted abstract from the patent document")
    abstract_page: Optional[int] = Field(None, description="The page number where the abstract was found")
    abstract_pattern: Optional[int] = Field(None, description="The pattern index that matched the abstract")
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )
# New models for patent_files list endpoint
class PatentFile(BaseModel):
    """Individual patent file information from database."""
    id: str = Field(..., description="Unique identifier for the patent file")
    filename: str = Field(..., description="Name of the patent file")
    mime_type: str = Field(default="application/pdf", description="MIME type of the patent file")
    uploaded_at: datetime = Field(..., description="Timestamp when the patent was uploaded")

class PatentFilesListResponse(BaseModel):
    """Response model for listing patent files."""
    status: str = Field(default="success", description="Status of the request")
    message: str = Field(default="Patent files retrieved successfully", description="Status message")
    data: list[PatentFile] = Field(..., description="List of patent files")
    count: int = Field(..., description="Total number of patent files")
    status_code: int = Field(default=200, description="Status code of the response")
class MultiQueryResponse(BaseModel):
    """
    Response model for multi-query search operations.
    
    Attributes:
        status: Status of the response (success/error)
        message: Optional summary message generated from the retrieved chunks
        data: List of chunks matching the search query
    """
    status: str = Field(..., description="Status of the response (success/error)")
    message: Optional[str] = Field(None, description="Summary message generated from the retrieved chunks")
    data: List[Chunk] = Field(..., description="List of chunks matching the search query")


class Metadata(BaseModel):
    """
    Metadata for a document stored in the system.
    
    Attributes:
        eTag: Entity tag for the document
        size: Size of the document in bytes
        mimetype: MIME type of the document
        cacheControl: Cache control directives for the document
        lastModified: Timestamp when the document was last modified
        contentLength: Content length of the document in bytes
        httpStatusCode: HTTP status code associated with the document
    """
    eTag: str
    size: int
    mimetype: str
    cacheControl: str
    lastModified: datetime
    contentLength: int
    httpStatusCode: int


class Document(BaseModel):
    """
    Represents a document stored in the system.
    
    Attributes:
        name: Name of the document
        id: Unique identifier for the document
        updated_at: Timestamp when the document was last updated
        created_at: Timestamp when the document was created
        last_accessed_at: Timestamp when the document was last accessed
        metadata: Additional metadata for the document
        queryable: Flag indicating if the document is available for querying in LanceDB
    """
    name: str
    id: str
    updated_at: datetime
    created_at: datetime
    last_accessed_at: datetime
    metadata: Metadata
    queryable: bool = False  # Flag to indicate if the file is available for querying in LanceDB


class FilesResponse(BaseModel):
    """
    Response model for file listing operations.
    
    Attributes:
        status: Status of the response (success/error)
        response: List of documents retrieved from storage
        diagnostics: Optional diagnostic information about the retrieval process
    """
    status: str = Field(..., description="Status of the response (success/error)")
    response: List[Document] = Field(..., description="List of documents retrieved from storage")
    diagnostics: Optional[dict] = Field(None, description="Diagnostic information about the retrieval process")
class SyntheticEmbodimentRequest(BaseModel):
    """
    Request model for generating a synthetic embodiment.
    
    Attributes:
        file_id (str): The UUIDv4 of the patent.
        inspiration (float): Degree of inspiration to apply (e.g., creativity factor).
        knowledge (list[dict]): List of knowledge components.
        patent_title (str): Title of the patent for context.
        disease (str): Disease relevant to the embodiment.
        antigen (str): Antigen relevant to the embodiment.
    """
    file_id: str = Field(..., description="The UUIDv4 of the patent")
    inspiration: float = Field(..., description="Degree of inspiration to apply (e.g., creativity factor)")
    knowledge : list[dict] = Field(..., description="list of knowledge components")
    patent_title: str = Field(..., description="Title of the patent for context")
    disease: str = Field(..., description="Disease relevant to the embodiment")
    antigen: str = Field(..., description="Antigen relevant to the embodiment")
    
class EmbodimentApproveSuccessResponse(BaseModel):
    """
    Response model for successful embodiment approval.
    
    Attributes:
        status: Status of the response (always "success")
        message: Status message describing the result of the approval operation
        data: Optional data returned from the database update operation
    """
    status: str = "success"
    message: str
    data: Optional[Any]

class EmbodimentsListResponse(BaseModel):
    """
    Response model for a list of embodiments.
    """
    status: str = "success"
    message: str
    data: list
    terms: list

class EmbodimentApproveErrorResponse(BaseModel):
    """
    Response model for failed embodiment approval.
    
    Attributes:
        status: Status of the response (always "error")
        message: Error message describing what went wrong
    """
    status: str = "error"
    message: str

class ApprovedEmbodimentRequest(BaseModel):
    """
    Request model for approving and storing an embodiment.
    
    Attributes:
        patent_id (str): The UUIDv4 of the patent to update.
        embodiment (Union[Embodiment, DetailedDescriptionEmbodiment]): The embodiment object to store.
    """
    patent_id: str = Field(..., description="The UUIDv4 of the patent to update")
    embodiment: Union[Embodiment, DetailedDescriptionEmbodiment] = Field(..., description="The embodiment object to store")

    # @field_validator('patent_id')
    # @classmethod
    # def validate_uuid4(cls, v):
    #     if v.version != 4:
    #         raise ValueError('patent_id must be a valid UUID4')
    #     return v
    
    
class ApproachKnowledge(BaseModel):
    """
    Request class to store approach knowledge about a patent.
    
    Attributes:
        patent_id: Unique identifier for the associated patent
        question: The question about the approach
        answer: The answer providing knowledge about the approach
        created_at: A timestamp representing the time when the knowledge item was generated
    """
    patent_id: str
    question: str
    answer: str
    created_at: str
    
    # @field_validator('patent_id')
    # @classmethod
    # def validate_uuid4(cls, v):
    #     if v.version != 4:
    #         raise ValueError('patent_id must be a valid UUID4')
    #     return v
    
class InnovationKnowledge(BaseModel):
    """
    Request class to store innovation knowledge about a patent.
    """
    patent_id: str
    question: str
    answer: str
    created_at: str

    # @field_validator('patent_id')
    # @classmethod
    # def validate_uuid4(cls, v):
    #     if v.version != 4:
    #         raise ValueError('patent_id must be a valid UUID4')
    #     return v

class TechnologyKnowledge(BaseModel):
    """
    Request class to store technology knowledge about a patent.
    """
    patent_id: str
    question: str
    answer: str
    created_at: str

    # @field_validator('patent_id')
    # @classmethod
    # def validate_uuid4(cls, v):
    #     if v.version != 4:
    #         raise ValueError('patent_id must be a valid UUID4')
    #     return v
  


class NoteCategory(str, Enum):
    APPROACH = "approach"
    INNOVATION = "innovation"
    TECHNOLOGY = "technology"    
class ResearchNote(BaseModel):
    """
    Request class to store research notes about a patent.
    
    Attributes:
        patent_id: Unique identifier for the associated patent
        category: Category of the research note (e.g., "Approach", "Innovation")
        content: The actual note content
        created_at: Timestamp when the note was created
    """
    patent_id: str = Field(..., description="Unique identifier for the associated patent")
    category: str = Field(..., description="'Approach', 'Innovation',", json_schema_extra=["approach", "innovation","technology"])
    content: str = Field(..., description="The actual note content")
    created_at: str = Field(..., description="Timestamp when the note was created")

    # @field_validator('patent_id')
    # @classmethod
    # def validate_uuid4(cls, v):
    #     if v.version != 4:
    #         raise ValueError('patent_id must be a valid UUID4')
    #     return v

# Response models for fetching stored knowledge
class ApproachKnowledgeListResponse(BaseModel):
    """
    Response model for listing approach knowledge items.
    """
    status: str = Field(..., description="Status of the response (success/error)")
    data: List[ApproachKnowledge] = Field(..., description="List of approach knowledge items")

class InnovationKnowledgeListResponse(BaseModel):
    """
    Response model for listing innovation knowledge items.
    """
    status: str = Field(..., description="Status of the response (success/error)")
    data: List[InnovationKnowledge] = Field(..., description="List of innovation knowledge items")

class TechnologyKnowledgeListResponse(BaseModel):
    """
    Response model for listing technology knowledge items.
    """
    status: str = Field(..., description="Status of the response (success/error)")
    data: List[TechnologyKnowledge] = Field(..., description="List of technology knowledge items")

class ResearchNoteListResponse(BaseModel):
    """
    Response model for listing research notes.
    """
    status: str = Field(..., description="Status of the response (success/error)")
    data: List[ResearchNote] = Field(..., description="List of research notes")

class DropTablesResponse(BaseModel):
    """
    Response model for dropping all LanceDB tables.
    """
    status: str = Field(..., description="Status of the drop operation")
    tables: List[str] = Field(..., description="Names of tables that were dropped")

# Response model for deleting Supabase files
class DeleteFileResponse(BaseModel):
    """
    Response model for deleting Supabase files.
    """
    status: str = Field(..., description="Status of the deletion operation")
    filename: str = Field(..., description="Name of the deleted file")

# Response model for deleting all Supabase files
class DeleteAllFilesResponse(BaseModel):
    """
    Response model for deleting all Supabase files.
    """
    status: str = Field(..., description="Status of the deletion operation")
    filenames: List[str] = Field(..., description="Names of deleted files")

# Response model for dropping a single LanceDB table
class DropTableResponse(BaseModel):
    """
    Response model for dropping a single LanceDB table.
    """
    status: str = Field(..., description="Status of the drop operation")
    table: str = Field(..., description="Name of the dropped table")

# Response model for fetching stored embodiments
class EmbodimentListResponse(BaseModel):
    """
    Response model for fetching stored embodiments for a patent.
    """
    status: str = Field(..., description="Status of the fetch operation")
    data: List[Any] = Field(..., description="List of stored embodiments")