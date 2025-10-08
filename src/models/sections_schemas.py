from pydantic import BaseModel, Field

from src.models.llm import TargetOverview


class SectionGenerationRequest(BaseModel):
    innovation: str = Field(..., description="The innovation for the patent draft")
    technology: str = Field(..., description="The technology for the patent draft")
    approach: str = Field(..., description="The approach for the patent draft")
    antigen: str = Field(..., description="The antigen for the patent draft")
    disease: str = Field(..., description="The disease for the patent draft")
    additional: str = Field(..., description="Additional information for the patent draft")

class KeyTermsGenerationRequest(SectionGenerationRequest):
    context: str = Field(..., description="The context for the patent draft")

class KeyTermsResponse(KeyTermsGenerationRequest):
    prediction: str = Field(..., description="The prediction for the section")
    trace_id: str = Field(..., description="The trace id for the section")

class SectionResponse(SectionGenerationRequest):
    prediction: str = Field(..., description="The prediction for the section")
    trace_id: str = Field(..., description="The trace id for the section")

class TargetOverviewRequest(SectionGenerationRequest):
    context: str = Field(..., description="Context for the target overview")
    
class TargetOverviewResponse(SectionResponse):
    context: str = Field(..., description="Context for the target overview")

# Subsection schemas for sections with extra parameters
class DiseaseSpecificOverviewRequest(SectionGenerationRequest):
    disease_name: str = Field(..., description="Specific disease name for this overview")

class DiseaseSpecificOverviewResponse(SectionResponse):
    disease_name: str = Field(..., description="Specific disease name for this overview")

class TargetInDiseaseRequest(SectionGenerationRequest):
    target_name: str = Field(..., description="Specific target name")
    disease_name: str = Field(..., description="Specific disease name")

class TargetInDiseaseResponse(SectionResponse):
    target_name: str = Field(..., description="Specific target name")
    disease_name: str = Field(..., description="Specific disease name")
