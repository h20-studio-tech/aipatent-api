import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union, Dict 
from src.models.ocr_schemas import (
    Embodiment,
    DetailedDescriptionEmbodiment,
    SectionHierarchy,
)
from src.models.rag_schemas import Chunk
from enum import Enum
from src.models.ocr_schemas import Glossary


class PageAnalysis(BaseModel):
    """Analysis result for a single page of a document."""
    page_number: int = Field(..., description="Page number")
    content: str = Field(..., description="Extracted content from the page")
    word_count: int = Field(..., description="Number of words on the page")
    analysis: str = Field(..., description="AI analysis of the page content")


class SectionAnalysis(BaseModel):
    """Analysis result for a document section."""
    section_title: str = Field(..., description="Title of the section")
    section_type: str = Field(..., description="Type of section (e.g., 'abstract', 'claims', 'description')")
    content: str = Field(..., description="Section content")
    analysis: str = Field(..., description="AI analysis of the section")
    key_insights: List[str] = Field(default_factory=list, description="Key insights extracted from the section")


class ComprehensiveAnalysisRequest(BaseModel):
    """Request model for comprehensive document analysis."""
    analysis_type: str = Field(
        default="full", 
        description="Type of analysis: 'full', 'sections_only', or 'summary_only'"
    )
    include_page_breakdown: bool = Field(
        default=True, 
        description="Whether to include page-by-page analysis"
    )
    custom_instructions: Optional[str] = Field(
        None, 
        description="Custom instructions for the analysis"
    )


class ComprehensiveAnalysisResponse(BaseModel):
    """Response model for comprehensive document analysis."""
    filename: str = Field(..., description="Name of the analyzed file")
    total_pages: int = Field(..., description="Total number of pages processed")
    total_characters: int = Field(..., description="Total characters in the document")
    
    # Overall analysis
    document_summary: str = Field(..., description="Overall document summary")
    document_type: str = Field(..., description="Type of document identified")
    key_technical_concepts: List[str] = Field(..., description="Main technical concepts found")
    main_claims_findings: List[str] = Field(..., description="Key claims or findings")
    innovation_assessment: str = Field(..., description="Assessment of document's innovation")
    potential_applications: List[str] = Field(..., description="Potential applications")
    
    # Detailed breakdowns
    page_analyses: Optional[List[PageAnalysis]] = Field(None, description="Page-by-page analysis")
    section_analyses: List[SectionAnalysis] = Field(..., description="Section-based analysis")
    
    # Processing metadata
    processing_time_seconds: float = Field(..., description="Time taken to process the document")
    analysis_timestamp: datetime = Field(..., description="When the analysis was completed")
    status: str = Field(..., description="Processing status")


class ComprehensiveAnalysisErrorResponse(BaseModel):
    """Error response for comprehensive analysis."""
    error: str = Field(..., description="Error message")
    filename: str = Field(..., description="Name of the file that failed to process")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="When the error occurred")

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
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )
    abstract: Optional[str] = Field(None, description="The extracted abstract from the patent document")
    abstract_page: Optional[int] = Field(None, description="The page number where the abstract was found")
    abstract_pattern: Optional[int] = Field(None, description="The pattern index that matched the abstract")
    data: list[Union[Embodiment, DetailedDescriptionEmbodiment]] = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )
    sections: Optional[list[SectionHierarchy]] = Field(
        None,
        description="Hierarchical structure of sections → subsections → embodiments",
    )
    terms: Optional[Glossary] = Field(
        None,
        description="The glossary of terms in the patent document (may be absent)",
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
    
class EmbodimentStatusUpdateRequest(BaseModel):
    """
    Request model for updating embodiment status.

    Attributes:
        embodiment_id: The unique ID of the embodiment (UUID)
        status: The new status (pending, approved, or rejected)
    """
    embodiment_id: str = Field(..., description="Unique ID (UUID) of the embodiment")
    status: str = Field(..., pattern="^(pending|approved|rejected)$",
                       description="New status for the embodiment")

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
    filename: str = Field(..., description="The name of the uploaded file")
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    message: str
    abstract: str
    sections: list
    terms: list
    data: list

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


class RawSectionsResponse(BaseModel):
    """Response model for retrieving raw section text for a patent."""
    status: str = Field(default="success", description="Status of the response (success/error)")
    file_id: str = Field(..., description="Unique identifier for the patent file")
    filename: Optional[str] = Field(None, description="Filename of the patent document")
    sections: Dict[str, str] = Field(..., description="Mapping of section name to raw text")


class PageData(BaseModel):
    """Represents the raw text content of a single page."""
    page_number: int = Field(..., description="Page number in the document")
    text: str = Field(..., description="Raw text content of the page")
    section: str = Field(..., description="Section this page belongs to")
    filename: str = Field(..., description="Source filename")


class PageBasedSectionsResponse(BaseModel):
    """Response model for retrieving patent content organized by pages."""
    status: str = Field(default="success", description="Status of the response (success/error)")
    file_id: str = Field(..., description="Unique identifier for the patent file")
    filename: Optional[str] = Field(None, description="Filename of the patent document")
    pages: List[PageData] = Field(..., description="List of pages with their content and metadata")
    total_pages: int = Field(..., description="Total number of pages in the document")


# Patent Content Draft Models for AIP-1
class ComponentUpdateRequest(BaseModel):
    """Request model for updating a single patent draft component."""
    component_id: str = Field(..., description="Unique component identifier")
    type: str = Field(..., description="Component type (background, claims, abstract, etc.)")
    title: str = Field(..., description="Component title")
    content: str = Field(..., description="Generated text content")
    order: int = Field(..., description="Display order")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ComponentUpdateResponse(BaseModel):
    """Response model for component update operations."""
    status: str = Field(default="success", description="Status of the response")
    message: str = Field(..., description="Status message")
    patent_id: str = Field(..., description="Patent project ID")
    component_id: str = Field(..., description="Updated component ID")
    updated_at: datetime = Field(..., description="Update timestamp")


class PatentDraftSaveRequest(BaseModel):
    """Request model for saving complete patent draft state."""
    components: List[Dict[str, Any]] = Field(..., description="Complete list of draft components")


class PatentDraftSaveResponse(BaseModel):
    """Response model for draft save operations."""
    status: str = Field(default="success", description="Status of the response")
    message: str = Field(..., description="Status message")
    patent_id: str = Field(..., description="Patent project ID")
    version: int = Field(..., description="Draft version number")
    last_saved_at: datetime = Field(..., description="Last save timestamp")
    components_count: int = Field(..., description="Number of components saved")


class PatentDraftResponse(BaseModel):
    """Response model for retrieving patent draft."""
    status: str = Field(default="success", description="Status of the response")
    patent_id: str = Field(..., description="Patent project ID")
    components: List[Dict[str, Any]] = Field(..., description="List of draft components")
    version: int = Field(..., description="Draft version number")
    last_saved_at: datetime = Field(..., description="Last save timestamp")
    created_at: datetime = Field(..., description="Draft creation timestamp")