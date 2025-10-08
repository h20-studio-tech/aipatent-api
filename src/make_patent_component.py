import uuid
import os
from datetime import datetime
from openai import OpenAI
from src.models.llm import (
    FieldOfInvention,
    BackgroundAndNeed,
    BriefSummary,
    DiseaseOverview,
    TargetOverview,
    HighLevelConcept,
    UnderlyingMechanism,
    Embodiment,
    Claims,
    Abstract,
    KeyTerms,
    # Summary of Invention Subsections
    TargetPatientPopulations,
    TherapeuticComposition,
    AlternativeEmbodiments,
    CoreClaims,
    # Detailed Description - Disease & Pathology
    DiseaseSpecificOverview,
    TargetInDisease,
    EpidemiologyClinicalNeed,
    # Detailed Description - Therapeutic Formulation
    HyperimmunizedEggProducts,
    AntigenicTargets,
    ProductionMethods,
    PharmaceuticalCompositions,
    # Detailed Description - Definitions
    KeyTerminology,
)
from src.utils.values_to_json import values_to_json
from src.utils.langfuse_client import get_langfuse_instance

provider = os.getenv("AI_PROVIDER")
r_reasoning = os.getenv("r_reasoning")

client = None

if provider == "gemini":
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
    )
else:
    client = OpenAI()

langfuse = get_langfuse_instance()
model = "gemini-2.5-flash"
g_reasoning = os.getenv("g_reasoning")


def generate_field_of_invention(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional:str,
    model: str = model,
    client=client,
    lf=None) -> FieldOfInvention:
    lf = lf or langfuse

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = lf.trace(
        id=trace_id,
        name=f"generate_field_of_invention_{model}",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
       ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = lf.get_prompt("generate_field_of_invention")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology, 
            antigen=antigen, 
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional
        ),
        output=raw_prompt,
    )
    generation = trace.generation(
        name="field_of_invention",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        reasoning_effort=g_reasoning,
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return FieldOfInvention(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_background(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str,
    model: str = model) -> BackgroundAndNeed:

    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name=f"generate_background_and_need_{model}",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_background")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
       ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="background_and_need",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], 
        model=model,
        reasoning_effort=g_reasoning,
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )
    return BackgroundAndNeed(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_summary(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model) -> BriefSummary:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name=f"generate_brief_summary_{model}",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
      ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_brief_summary")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
       )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="brief_summary",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return BriefSummary(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_target_overview(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    context: str,
    model: str = model) -> TargetOverview:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_target_overview",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_target_overview")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,   
        context=context 
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        context=context
       ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="target_overview",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning,
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return TargetOverview(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_high_level_concept(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model) -> HighLevelConcept:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_high_level_concept",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_high_level_concept")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="high_level_concept",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning,
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return HighLevelConcept(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_underlying_mechanism(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model
) -> UnderlyingMechanism:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_underlying_mechanism",
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_underlying_mechanism")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,)
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,  
        additional=additional,
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="underlying_mechanism",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return UnderlyingMechanism(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_embodiment(
    previous_embodiment: str, 
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str, 
    model: str = model
) -> Embodiment:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_embodiment",
        input=values_to_json(
            previous_embodiment=previous_embodiment, 
            innovation=innovation, 
            technology=technology, 
            antigen=antigen, 
            disease=disease
        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_embodiment")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease, 
        innovation=innovation,
        approach=approach, 
        previous_embodiment=previous_embodiment
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology, 
            antigen=antigen, 
            disease=disease,
            innovation=innovation,
            previous_embodiment=previous_embodiment
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="embodiment",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return Embodiment(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_disease_overview(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model
) -> DiseaseOverview:
    
    if not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_disease_overview",
        input=values_to_json(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_disease_overview")
    prompt = raw_prompt.compile(
        innovation=innovation,
        technology=technology,
        approach=approach,
        antigen=antigen,
        disease=disease,
        additional=additional,
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            disease=disease,
            additional=additional,
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="disease_overview",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return DiseaseOverview(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_claims(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model) -> Claims:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_claims",
        input=values_to_json(
            technology=technology, 
            antigen=antigen, 
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_claims")
    prompt = raw_prompt.compile(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
        )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
    ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="claims",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return Claims(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_key_terms(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    context: str,
    model: str = model
) -> KeyTerms:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_key_terms",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional,
            context=context),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_key_terms")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        context=context
        )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        context=context),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="key_terms",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return KeyTerms(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_abstract(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model
) -> Abstract:
    
    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_abstract",
        input=values_to_json(
            technology=technology, 
            antigen=antigen, 
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
        )

    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_abstract")
    prompt = raw_prompt.compile(
            technology=technology, 
            antigen=antigen, 
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional    
    )

    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
        technology=technology, 
        antigen=antigen, 
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="abstract",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return Abstract(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

# ============================================
# Summary of Invention Subsections
# ============================================

def generate_target_patient_populations(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> TargetPatientPopulations:

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_target_patient_populations",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_target_patient_populations")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="target_patient_populations",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return TargetPatientPopulations(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_therapeutic_composition(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> TherapeuticComposition:

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_therapeutic_composition",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_therapeutic_composition")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="therapeutic_composition",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return TherapeuticComposition(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_alternative_embodiments(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> AlternativeEmbodiments:

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_alternative_embodiments",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_alternative_embodiments")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="alternative_embodiments",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return AlternativeEmbodiments(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_core_claims(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> CoreClaims:

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_core_claims",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_core_claims")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="core_claims",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, reasoning_effort=g_reasoning
    )
    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return CoreClaims(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

# ============================================
# Detailed Description - Disease & Pathology Subsections
# ============================================

def generate_disease_specific_overview(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    disease_name: str,  # Specific disease for this overview
    model: str = model
) -> DiseaseSpecificOverview:
    """Generate disease-specific overview for a particular disease"""

    if not disease_name:
        raise ValueError("Disease name must be non-empty string")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_disease_specific_overview",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional,
            disease_name=disease_name),
        tags=["evaluation"],
    )

    # Inline prompt for disease-specific overview
    prompt = f"""Generate a comprehensive disease-specific overview for {disease_name} as a subsection of a patent application.

This subsection should focus specifically on {disease_name} and include:
1. Definition and classification of {disease_name}
2. Pathophysiology and disease mechanisms
3. Current standard of care and limitations
4. Unmet medical needs specific to {disease_name}
5. How the invention addresses these needs

Context:
- Technology: {technology}
- Target Antigen: {antigen}
- Disease Focus: {disease}
- Innovation: {innovation}
- Approach: {approach}
- Additional Information: {additional}

Write in formal patent language, using present tense and third person. Include specific medical terminology and cite relevant disease statistics where appropriate. Focus on establishing the medical need and context for the invention."""

    generation = trace.generation(
        name="disease_specific_overview",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return DiseaseSpecificOverview(
        prediction=response.choices[0].message.content,
        trace_id=trace_id,
        disease_name=disease_name
    )

def generate_target_in_disease(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    target_name: str,  # Specific target
    disease_name: str,  # Specific disease
    model: str = model
) -> TargetInDisease:
    """Generate target in disease subsection for specific target/disease combination"""

    if not target_name or not disease_name:
        raise ValueError("Target name and disease name must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_target_in_disease",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional,
            target_name=target_name,
            disease_name=disease_name),
        tags=["evaluation"],
    )

    # Inline prompt for target in disease
    prompt = f"""Generate a detailed subsection describing the role of {target_name} in {disease_name} for a patent application.

This subsection should comprehensively cover:
1. Molecular biology and function of {target_name}
2. Expression patterns and localization in {disease_name}
3. Pathogenic role of {target_name} in {disease_name} progression
4. Evidence supporting {target_name} as a therapeutic target
5. Prior attempts to target {target_name} and their limitations
6. Advantages of the current approach targeting {target_name}

Context:
- Technology: {technology}
- Primary Antigen: {antigen}
- Disease Area: {disease}
- Innovation: {innovation}
- Therapeutic Approach: {approach}
- Additional Information: {additional}

Use formal patent language with technical precision. Include molecular mechanisms, relevant research findings, and establish clear rationale for targeting {target_name} in {disease_name}."""

    generation = trace.generation(
        name="target_in_disease",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return TargetInDisease(
        prediction=response.choices[0].message.content,
        trace_id=trace_id,
        target_name=target_name,
        disease_name=disease_name
    )

def generate_epidemiology_clinical_need(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> EpidemiologyClinicalNeed:
    """Generate epidemiology and clinical need subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_epidemiology_clinical_need",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for epidemiology and clinical need
    prompt = f"""Generate an epidemiology and clinical need subsection for a patent application.

This subsection should provide comprehensive coverage of:
1. Global and regional epidemiology statistics for {disease}
2. Incidence and prevalence trends
3. Patient demographics and risk factors
4. Disease burden (mortality, morbidity, quality of life impact)
5. Economic burden (healthcare costs, productivity loss)
6. Current treatment landscape and success rates
7. Treatment gaps and unmet medical needs
8. Market opportunity for novel therapeutics

Context:
- Technology: {technology}
- Target Antigen: {antigen}
- Disease Focus: {disease}
- Innovation: {innovation}
- Approach: {approach}
- Additional Information: {additional}

Write in formal patent language using authoritative data sources. Include specific statistics, trends, and projections. Establish clear justification for the medical and commercial need for the invention."""

    generation = trace.generation(
        name="epidemiology_clinical_need",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return EpidemiologyClinicalNeed(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

# ============================================
# Detailed Description - Therapeutic Formulation Subsections
# ============================================

def generate_hyperimmunized_egg_products(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> HyperimmunizedEggProducts:
    """Generate hyperimmunized egg products subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_hyperimmunized_egg_products",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for hyperimmunized egg products
    prompt = f"""Generate a detailed subsection on hyperimmunized egg products for a patent application focused on IgY antibody technology.

This subsection should comprehensively describe:
1. Production process of hyperimmunized eggs
   - Hen immunization protocols
   - Antigen preparation and administration
   - Antibody response monitoring
2. IgY antibody characteristics
   - Structure and properties vs mammalian antibodies
   - Advantages (no complement activation, reduced inflammatory response)
   - Stability and bioavailability
3. Egg yolk processing methods
   - Extraction and purification techniques
   - Quality control parameters
   - Standardization methods
4. Final product formulations
   - Powder, liquid, or encapsulated forms
   - Excipients and stabilizers
   - Storage conditions
5. Advantages over conventional antibody production

Context:
- Technology: {technology}
- Target Antigen: {antigen}
- Disease Application: {disease}
- Innovation: {innovation}
- Approach: {approach}
- Additional Information: {additional}

Use technical patent language with specific process parameters, concentrations, and methods. Include advantages of avian IgY system for this specific application."""

    generation = trace.generation(
        name="hyperimmunized_egg_products",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return HyperimmunizedEggProducts(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_antigenic_targets(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> AntigenicTargets:
    """Generate antigenic targets subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_antigenic_targets",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for antigenic targets
    prompt = f"""Generate a comprehensive subsection on antigenic targets for a patent application.

This subsection should detail:
1. Primary antigenic targets
   - {antigen} structure and epitopes
   - Conservation across strains/variants
   - Immunogenicity profile
2. Secondary or alternative targets
   - Related antigens that may be included
   - Cross-reactive epitopes
   - Combination targeting strategies
3. Antigen selection rationale
   - Why these specific targets were chosen
   - Advantages over other potential targets
   - Coverage of disease-relevant strains
4. Antigen preparation methods
   - Recombinant expression systems
   - Purification protocols
   - Modification or conjugation strategies
5. Immunogenic compositions
   - Antigen formulations for immunization
   - Adjuvants and delivery systems
   - Dose optimization

Context:
- Technology: {technology}
- Primary Target: {antigen}
- Disease Focus: {disease}
- Innovation: {innovation}
- Therapeutic Approach: {approach}
- Additional Information: {additional}

Use precise technical language with specific antigen characteristics, sequences regions, and immunological properties. Establish clear rationale for target selection."""

    generation = trace.generation(
        name="antigenic_targets",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return AntigenicTargets(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_production_methods(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> ProductionMethods:
    """Generate production methods subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_production_methods",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for production methods
    prompt = f"""Generate a detailed production methods subsection for a patent application involving IgY antibody production.

This subsection should comprehensively describe:
1. Immunization protocols
   - Animal selection and housing
   - Immunization schedule and routes
   - Antigen doses and formulations
   - Booster regimens
2. Antibody production monitoring
   - Titer determination methods
   - Specificity testing
   - Quality control checkpoints
3. Harvesting procedures
   - Egg collection protocols
   - Yolk separation techniques
   - Storage conditions
4. Purification processes
   - IgY extraction methods (water dilution, PEG precipitation, etc.)
   - Chromatography techniques if applicable
   - Filtration and sterilization
5. Scale-up considerations
   - Industrial production capabilities
   - Batch consistency measures
   - GMP compliance aspects
6. Quality assurance
   - Purity specifications
   - Activity assays
   - Stability testing

Context:
- Technology: {technology}
- Target Antigen: {antigen}
- Disease Application: {disease}
- Innovation: {innovation}
- Approach: {approach}
- Additional Information: {additional}

Use specific technical parameters, equipment specifications, and process controls. Include ranges for key parameters and alternative methods where applicable."""

    generation = trace.generation(
        name="production_methods",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return ProductionMethods(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

def generate_pharmaceutical_compositions(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> PharmaceuticalCompositions:
    """Generate pharmaceutical compositions subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_pharmaceutical_compositions",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for pharmaceutical compositions
    prompt = f"""Generate a comprehensive pharmaceutical compositions subsection for a patent application involving IgY-based therapeutics.

This subsection should detail:
1. Active ingredient specifications
   - IgY antibody content and purity
   - Specific activity requirements
   - Concentration ranges
2. Formulation types
   - Oral formulations (tablets, capsules, liquids)
   - Enteric coated preparations
   - Controlled release formulations
   - Topical applications if relevant
3. Excipients and carriers
   - Stabilizers (sugars, proteins, polymers)
   - Preservatives
   - Buffering agents
   - Coating materials
4. Dosage forms
   - Unit dose specifications
   - Multi-dose preparations
   - Combination products
5. Stability considerations
   - pH optimization
   - Temperature stability
   - Shelf life specifications
6. Administration routes
   - Oral delivery strategies
   - Protection from gastric degradation
   - Intestinal absorption enhancement
7. Combination therapies
   - Co-formulation with other actives
   - Synergistic combinations
   - Adjuvant therapies

Context:
- Technology: {technology}
- Target: {antigen}
- Disease Indication: {disease}
- Innovation: {innovation}
- Therapeutic Approach: {approach}
- Additional Information: {additional}

Use specific pharmaceutical terminology with concentration ranges, percentages, and technical specifications. Include multiple embodiments and alternatives."""

    generation = trace.generation(
        name="pharmaceutical_compositions",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return PharmaceuticalCompositions(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )

# ============================================
# Detailed Description - Definitions Subsection
# ============================================

def generate_key_terminology(
    innovation: str,
    technology: str,
    approach: str,
    antigen: str,
    disease: str,
    additional: str,
    model: str = model
) -> KeyTerminology:
    """Generate key terminology subsection"""

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
        id=trace_id,
        name="generate_key_terminology",
        input=values_to_json(
            technology=technology,
            antigen=antigen,
            disease=disease,
            innovation=innovation,
            approach=approach,
            additional=additional),
        tags=["evaluation"],
    )

    # Inline prompt for key terminology
    prompt = f"""Generate a key terminology/definitions subsection for a patent application.

This subsection should provide clear, technical definitions for all important terms used in the patent, including:

1. Technical terms specific to the technology
   - "IgY antibody" - definition and characteristics
   - "Hyperimmunized" - specific meaning in context
   - "Egg yolk immunoglobulin" - technical definition

2. Disease-related terminology
   - Terms specific to {disease}
   - Pathological processes
   - Clinical endpoints

3. Target-related terms
   - "{antigen}" - molecular definition
   - Related proteins or pathways
   - Epitope terminology

4. Formulation terminology
   - Pharmaceutical terms used
   - Dosage form definitions
   - Excipient categories

5. Process terminology
   - Production method terms
   - Purification terminology
   - Quality metrics

6. General patent terms
   - "Therapeutically effective amount"
   - "Substantially pure"
   - "Pharmaceutical composition"
   - "Subject" or "patient"

Context:
- Technology: {technology}
- Target: {antigen}
- Disease: {disease}
- Innovation: {innovation}
- Approach: {approach}
- Additional Information: {additional}

Format as a glossary with clear, concise technical definitions. Each term should be in quotes followed by its definition. Definitions should be technically accurate and legally precise."""

    generation = trace.generation(
        name="key_terminology",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        reasoning_effort=g_reasoning
    )

    generation.end(
        output=response.choices[0].message.content,
        end_time=datetime.now(),
        usage=response.usage,
        metadata={"prompt_usage": response.usage},
        model=model,
    )

    trace.update(
        status_message="Generation completed",
        output=response.choices[0].message.content,
        metadata={"prompt_usage": response.usage},
    )

    return KeyTerminology(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )
