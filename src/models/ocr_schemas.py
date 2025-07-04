from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CategoryResponse(BaseModel):
    sub_category: str = Field(
        ...,
        description="The category of the embodiment",
        json_schema_extra=["disease rationale", "product composition"],
    )


class GlossaryDefinition(BaseModel):
    term: str = Field(..., description="The defined key term")
    definition: str = Field(..., description="The definition of the key term")
    page_number: int = Field(
        ..., description="The page number of the glossary definition"
    )


class ProcessedPage(BaseModel):
    text: str = Field(
        ..., description="The content of a page that contains embodiments"
    )
    filename: str = Field(..., description="The source file of the page")
    page_number: int = Field(
        ..., description="The page number of the page in the source file"
    )
    section: str = Field(..., description="The section of the page in the source file")
    image: Optional[str] = Field(
        None, description="The base64 encoded image of the page"
    )


class Glossary(BaseModel):
    definitions: list[GlossaryDefinition] = Field(
        ...,
        description="List of key term definitions extracted from the definitions subsection.",
    )
    filename: str = Field(..., description="The source file of the glossary")


class GlossarySubsectionPage(BaseModel):
    """Glossary definitions extracted from a single page."""

    definitions: list[GlossaryDefinition] = Field(
        ..., description="Glossary definitions found on the page"
    )
    text: str = Field("", description="Raw text of the page")
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number in the PDF")
    section: str = Field(..., description="Section label for the page")


class Embodiment(BaseModel):
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(
        ..., description="The section of the embodiment in the source file"
    )
    # Allow initial creation without summary
    summary: str = Field("", description="the embodiment summary")


class DetailedDescriptionEmbodiment(BaseModel):
    # Define all fields explicitly instead of using inheritance
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(
        ..., description="The section of the embodiment in the source file"
    )
    sub_category: str = Field(
        ...,
        description="The category of the embodiment",
        json_schema_extra=["disease rationale", "product composition"],
    )
    # New optional header field populated when a page-level header is detected
    header: Optional[str] = Field(
        None, description="Header text detected on the same page (if any)"
    )
    # Allow initial creation without summary
    summary: str = Field("", description="the embodiment summary")


class EmbodimentSummary(BaseModel):
    summary: str = Field(..., description="the embodiment summmary")


class EmbodimentSpellCheck(BaseModel):
    text: str = Field(..., description="the source text of the embodiment")
    checked_text: str = Field(..., description="the spell-checked embodiment text")


class Embodiments(BaseModel):
    content: list[Embodiment] | list = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )


class PatentSection(BaseModel):
    """Classification of a patent document section."""

    section: str = Field(
        ...,
        description="The section of the patent document",
        json_schema_extra=["summary of invention", "detailed description", "claims"],
    )


class PatentSectionWithConfidence(BaseModel):
    """Classification of a patent document section with confidence score.

    Attributes:
        section: The section type of the patent document
        confidence: Confidence score for the section classification (0-1)
    """

    section: str = Field(
        ...,
        description="The section of the patent document",
        json_schema_extra=["summary of invention", "detailed description", "claims"],
    )
    confidence: float = Field(
        ..., description="Confidence score for section classification"
    )

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return v


class HeaderDetection(BaseModel):
    header: Optional[str] = Field(
        None, description="the header detected in the page (None if not detected)"
    )
    has_header: bool = Field(..., description="Confidence score for header detection")


class HeaderDetectionPage(ProcessedPage, HeaderDetection):
    pass


class GlossaryPageFlag(BaseModel):
    is_glossary_page: bool = Field(
        ..., description="True if the page contains glossary definitions"
    )


# --------------------
# New hierarchical models
# --------------------


class Subsection(BaseModel):
    """Represents a subsection header, its summary, and contained embodiments."""

    header: str = Field(..., description="Subsection header/title")
    summary: str = Field("", description="Summary of the subsection")
    embodiments: list[Embodiment | DetailedDescriptionEmbodiment] = Field(
        ..., description="Embodiments belonging to this subsection"
    )


class SectionHierarchy(BaseModel):
    """A main patent section with its subsections."""

    section: str = Field(
        ...,
        description="Main section name (e.g., Summary of Invention, Detailed Description, Claims)",
    )
    subsections: list[Subsection] = Field(
        ..., description="Subsections within the section"
    )
