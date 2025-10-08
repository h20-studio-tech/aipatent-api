from fastapi import APIRouter
from src.make_patent_component import (
    generate_abstract,
    generate_background,
    generate_claims,
    generate_disease_overview,
    generate_embodiment,
    generate_field_of_invention,
    generate_high_level_concept,
    generate_summary,
    generate_key_terms,
    generate_target_overview,
    generate_underlying_mechanism,
    # Summary of Invention Subsections
    generate_target_patient_populations,
    generate_therapeutic_composition,
    generate_alternative_embodiments,
    generate_core_claims,
    # Detailed Description - Disease & Pathology
    generate_disease_specific_overview,
    generate_target_in_disease,
    generate_epidemiology_clinical_need,
    # Detailed Description - Therapeutic Formulation
    generate_hyperimmunized_egg_products,
    generate_antigenic_targets,
    generate_production_methods,
    generate_pharmaceutical_compositions,
    # Detailed Description - Definitions
    generate_key_terminology,
)
from src.models.sections_schemas import (
    SectionGenerationRequest,
    SectionResponse,
    TargetOverviewRequest,
    TargetOverviewResponse,
    KeyTermsGenerationRequest,
    KeyTermsResponse,
    DiseaseSpecificOverviewRequest,
    DiseaseSpecificOverviewResponse,
    TargetInDiseaseRequest,
    TargetInDiseaseResponse,
)
router = APIRouter(prefix="/api/v1/sections", tags=["sections"])

@router.post("/background", response_model=SectionResponse)
async def background(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_background(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    
@router.post("/summary", response_model=SectionResponse)
async def summary(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_summary(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )


@router.post("/field_of_invention", response_model=SectionResponse)
async def field_of_invention(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_field_of_invention(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    
@router.post("/target_overview", response_model=TargetOverviewResponse)
async def target_overview(req: TargetOverviewRequest) -> TargetOverviewResponse:
    res = generate_target_overview(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        context=req.context,
    )
    return TargetOverviewResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        context=req.context,
    )
    
@router.post("/disease_overview", response_model=SectionResponse)
async def disease_overview(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_disease_overview(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    
@router.post("/underlying_mechanism", response_model=SectionResponse)
async def underlying_mechanism(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_underlying_mechanism(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    
@router.post("/high_level_concept", response_model=SectionResponse)
async def high_level_concept(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_high_level_concept(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/claims", response_model=SectionResponse)
async def claims(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_claims(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/abstract", response_model=SectionResponse)
async def abstract(req: SectionGenerationRequest) -> SectionResponse:
    res = generate_abstract(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    
@router.post("/key_terms", response_model=KeyTermsResponse)
async def key_terms(req: KeyTermsGenerationRequest) -> KeyTermsResponse:
    res = generate_key_terms(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        context=req.context
    )
    return KeyTermsResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        context=req.context,
    )

# ============================================
# Patent Subsections Endpoints

# ============================================
# Summary of Invention Subsections
# ============================================

@router.post("/subsections/target_patient_populations", response_model=SectionResponse)
async def target_patient_populations(req: SectionGenerationRequest) -> SectionResponse:
    """Generate target patient populations subsection"""
    res = generate_target_patient_populations(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/therapeutic_composition", response_model=SectionResponse)
async def therapeutic_composition(req: SectionGenerationRequest) -> SectionResponse:
    """Generate therapeutic composition subsection"""
    res = generate_therapeutic_composition(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/alternative_embodiments", response_model=SectionResponse)
async def alternative_embodiments(req: SectionGenerationRequest) -> SectionResponse:
    """Generate alternative embodiments subsection"""
    res = generate_alternative_embodiments(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/core_claims", response_model=SectionResponse)
async def core_claims(req: SectionGenerationRequest) -> SectionResponse:
    """Generate core claims subsection"""
    res = generate_core_claims(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

# ============================================
# Detailed Description - Disease & Pathology
# ============================================

@router.post("/subsections/disease_specific_overview", response_model=DiseaseSpecificOverviewResponse)
async def disease_specific_overview(req: DiseaseSpecificOverviewRequest) -> DiseaseSpecificOverviewResponse:
    """Generate disease-specific overview subsection"""
    res = generate_disease_specific_overview(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        disease_name=req.disease_name,
    )
    return DiseaseSpecificOverviewResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        disease_name=res.disease_name,
    )

@router.post("/subsections/target_in_disease", response_model=TargetInDiseaseResponse)
async def target_in_disease(req: TargetInDiseaseRequest) -> TargetInDiseaseResponse:
    """Generate target in disease subsection"""
    res = generate_target_in_disease(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        target_name=req.target_name,
        disease_name=req.disease_name,
    )
    return TargetInDiseaseResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
        target_name=res.target_name,
        disease_name=res.disease_name,
    )

@router.post("/subsections/epidemiology_clinical_need", response_model=SectionResponse)
async def epidemiology_clinical_need(req: SectionGenerationRequest) -> SectionResponse:
    """Generate epidemiology and clinical need subsection"""
    res = generate_epidemiology_clinical_need(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

# ============================================
# Detailed Description - Therapeutic Formulation
# ============================================

@router.post("/subsections/hyperimmunized_egg_products", response_model=SectionResponse)
async def hyperimmunized_egg_products(req: SectionGenerationRequest) -> SectionResponse:
    """Generate hyperimmunized egg products subsection"""
    res = generate_hyperimmunized_egg_products(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/antigenic_targets", response_model=SectionResponse)
async def antigenic_targets(req: SectionGenerationRequest) -> SectionResponse:
    """Generate antigenic targets subsection"""
    res = generate_antigenic_targets(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/production_methods", response_model=SectionResponse)
async def production_methods(req: SectionGenerationRequest) -> SectionResponse:
    """Generate production methods subsection"""
    res = generate_production_methods(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

@router.post("/subsections/pharmaceutical_compositions", response_model=SectionResponse)
async def pharmaceutical_compositions(req: SectionGenerationRequest) -> SectionResponse:
    """Generate pharmaceutical compositions subsection"""
    res = generate_pharmaceutical_compositions(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )

# ============================================
# Detailed Description - Definitions
# ============================================

@router.post("/subsections/key_terminology", response_model=SectionResponse)
async def key_terminology(req: SectionGenerationRequest) -> SectionResponse:
    """Generate key terminology subsection"""
    res = generate_key_terminology(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return SectionResponse(
        prediction=res.prediction,
        trace_id=res.trace_id,
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )