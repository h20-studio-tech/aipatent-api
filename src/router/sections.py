from fastapi import APIRouter
from src.models.llm import (
    BackgroundAndNeed,
    Technology,
    BriefSummary,
    PrimaryInvention,
    FieldOfInvention,
    DescriptionOfInvention,
    HighLevelConcept,
    TargetOverview,
    UnderlyingMechanism,
    Embodiment,
    Claims,
    Abstract,
    KeyTerms,
)
from make_patent_component import (
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
from src.models.sections_schemas import SectionGenerationRequest
sections_router = APIRouter(prefix="/api/v1/sections", tags=["sections"])

@router.post("/summary", response_model=BriefSummary)
async def summary(req: SectionGenerationRequest) -> BriefSummary:
    res = generate_summary(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return BriefSummary(
        prediction=res,
        trace_id=res.trace_id,
    )

@router.post("/background", response_model=BackgroundAndNeed)
async def background(req: SectionGenerationRequest) -> BackgroundAndNeed:
    res = generate_background(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return BackgroundAndNeed(
        prediction=res,
        trace_id=res.trace_id,
    )

@router.post("/field_of_invention", response_model=FieldOfInvention)
async def field_of_invention(req: SectionGenerationRequest) -> FieldOfInvention:
    res = generate_field_of_invention(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return FieldOfInvention(
        prediction=res,
        trace_id=res.trace_id,
    )

@router.post("/background", response_model=BackgroundAndNeed)
async def background(req: SectionGenerationRequest) -> BackgroundAndNeed:
    res = generate_background(
        innovation=req.innovation,
        technology=req.technology,
        approach=req.approach,
        antigen=req.antigen,
        disease=req.disease,
        additional=req.additional,
    )
    return BackgroundAndNeed(
        prediction=res,
        trace_id=res.trace_id,
    )
