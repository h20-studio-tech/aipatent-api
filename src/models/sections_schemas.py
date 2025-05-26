from pydantic import BaseModel, Field

from src.models.llm import TargetOverview


class SectionGenerationRequest(BaseModel):
    innovation: str = Field(..., description="The innovation for the patent draft")
    technology: str = Field(..., description="The technology for the patent draft")
    approach: str = Field(..., description="The approach for the patent draft")
    antigen: str = Field(..., description="The antigen for the patent draft")
    disease: str = Field(..., description="The disease for the patent draft")
    additional: str = Field(..., description="Additional information for the patent draft")

class SectionResponse(SectionGenerationRequest):
    prediction: str = Field(..., description="The prediction for the section")
    trace_id: str = Field(..., description="The trace id for the section")

class TargetOverviewRequest(SectionGenerationRequest):
    context: str = Field(..., description="Context for the target overview")
    
class TargetOverviewResponse(SectionResponse):
    target_overview: TargetOverview = Field(..., description="The target overview")
    context: str = Field(..., description="Context for the target overview")
