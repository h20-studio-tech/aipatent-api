from pydantic import BaseModel, Field
from typing import Optional, List

class Technology(BaseModel):
    """
    Input values for our patent draft
    
    args:
        target_antigen: The target antigen for the technology
        disease: The disease for the technology, more than one disease is possible
    """
    target_antigen: str = Field(..., description="The target antigen for the technology. Only generate antigens associated with Immunoglobulin Y")
    disease: List[str] = Field(..., description="The disease for the technology, more than one disease is possible. Only generate diseases that Immunoglobulin Y has the potential to cure")

class PrimaryInvention(BaseModel):
    prediction: str | None  = Field(..., description="the redacted primary invention section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the primary invention generation")

class FieldOfInvention(BaseModel):
    prediction: str | None = Field(..., description="the redacted field of invention section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the field of invention generation")

class BackgroundAndNeed(BaseModel):
    prediction: str | None = Field(..., description="the redacted background and need section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the background and need generation")

class BriefSummary(BaseModel):
    prediction: str = Field(..., description="the redacted brief summary section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the brief summary generation")

class TechnologyPlatform(BaseModel):
    prediction: str = Field(..., description="the redacted technology platform section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the technology platform generation")

class DescriptionOfInvention(BaseModel):
    prediction: str = Field(..., description="the redacted description of invention section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the description of invention generation")

class ProductOrProducts(BaseModel):
    prediction: str = Field(..., description="the redacted product section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the product generation")

class Uses(BaseModel):
    prediction: str = Field(..., description="the redacted uses section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the uses generation")

class TargetOverview(BaseModel):
    prediction: str = Field(..., description="the redacted target overview section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the target overview generation")

class HighLevelConcept(BaseModel):
    prediction: str = Field(..., description="the redacted high level concept section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the high level concept generation")

class UnderlyingMechanism(BaseModel):
    prediction: str = Field(..., description="the redacted underlying mechanism section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the underlying mechanism generation")

class Embodiment(BaseModel):
    prediction: str = Field(..., description="the redacted embodiment section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the embodiment generation")

class Claims(BaseModel):
    prediction: str = Field(..., description="the redacted claims section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the claims generation")

class Abstract(BaseModel):
    prediction: str = Field(..., description="the redacted abstract section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the abstract generation")

class KeyTerms(BaseModel):
    prediction: str = Field(..., description="the redacted key terms section for the patent draft")
    trace_id: str = Field(..., description="the langfuse trace id for the key terms generation")

class DiseaseOverview(BaseModel):
    prediction: str = Field(
        ...,
        description="the redacted disease overview section for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the disease overview generation"
    )

# Summary of Invention Subsections
class TargetPatientPopulations(BaseModel):
    prediction: str = Field(
        ...,
        description="the target patient populations subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the target patient populations generation"
    )

class TherapeuticComposition(BaseModel):
    prediction: str = Field(
        ...,
        description="the therapeutic composition subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the therapeutic composition generation"
    )

class AlternativeEmbodiments(BaseModel):
    prediction: str = Field(
        ...,
        description="the alternative embodiments subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the alternative embodiments generation"
    )

class CoreClaims(BaseModel):
    prediction: str = Field(
        ...,
        description="the core claims subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the core claims generation"
    )

# Detailed Description - Disease & Pathology Subsections
class DiseaseSpecificOverview(BaseModel):
    prediction: str = Field(
        ...,
        description="the disease-specific overview subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the disease-specific overview generation"
    )
    disease_name: str = Field(
        ...,
        description="the specific disease name this overview is for"
    )

class TargetInDisease(BaseModel):
    prediction: str = Field(
        ...,
        description="the target in disease subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the target in disease generation"
    )
    target_name: str = Field(
        ...,
        description="the specific target name"
    )
    disease_name: str = Field(
        ...,
        description="the specific disease name"
    )

class EpidemiologyClinicalNeed(BaseModel):
    prediction: str = Field(
        ...,
        description="the epidemiology and clinical need subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the epidemiology and clinical need generation"
    )

# Detailed Description - Therapeutic Formulation Subsections
class HyperimmunizedEggProducts(BaseModel):
    prediction: str = Field(
        ...,
        description="the hyperimmunized egg products subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the hyperimmunized egg products generation"
    )

class AntigenicTargets(BaseModel):
    prediction: str = Field(
        ...,
        description="the antigenic targets subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the antigenic targets generation"
    )

class ProductionMethods(BaseModel):
    prediction: str = Field(
        ...,
        description="the production methods subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the production methods generation"
    )

class PharmaceuticalCompositions(BaseModel):
    prediction: str = Field(
        ...,
        description="the pharmaceutical compositions subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the pharmaceutical compositions generation"
    )

# Detailed Description - Definitions Subsection
class KeyTerminology(BaseModel):
    prediction: str = Field(
        ...,
        description="the key terminology subsection for the patent draft"
    )
    trace_id: str = Field(
        ...,
        description="the langfuse trace id for the key terminology generation"
    )
