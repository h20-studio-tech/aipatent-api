from pydantic import BaseModel, Field, field_validator
    
class CategoryResponse(BaseModel):
    sub_category: str = Field(..., 
                        description="The category of the embodiment",
                        json_schema_extra=["disease rationale", "product composition"])

class ProcessedPage(BaseModel):
    text: str = Field(
        ..., description="The content of a page that contains embodiments"
    )
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(..., description="The section of the embodiment in the source file")


class Embodiment(BaseModel):
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(..., description="The page number of the embodiment in the source file")
    section: str = Field(..., description="The section of the embodiment in the source file")

class DetailedDescriptionEmbodiment(BaseModel):
    # Define all fields explicitly instead of using inheritance
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(..., description="The section of the embodiment in the source file")
    sub_category: str = Field(..., 
                          description="The category of the embodiment",
                          json_schema_extra=["disease rationale", "product composition"])

class Embodiments(BaseModel):
    content: list[Embodiment] | list = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )


class PatentSection(BaseModel):
    """Classification of a patent document section."""
    section: str = Field(
        ..., 
        description="The section of the patent document",
        json_schema_extra=["summary of invention", "detailed description", "claims"]
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
        json_schema_extra=["summary of invention", "detailed description", "claims"]
    )
    confidence: float = Field(..., description="Confidence score for section classification")

    @field_validator('confidence')
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('Confidence must be between 0 and 1')
        return v