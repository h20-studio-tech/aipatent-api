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
)
from src.models.sections_schemas import (
    SectionGenerationRequest,
    SectionResponse,
    TargetOverviewRequest,
    TargetOverviewResponse,
    KeyTermsGenerationRequest,
    KeyTermsResponse,
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