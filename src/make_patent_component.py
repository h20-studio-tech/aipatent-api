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

)

from src.utils.values_to_json import values_to_json
from src.utils.langfuse_client import get_langfuse_instance

langfuse = get_langfuse_instance()

model = os.getenv("MODEL")

def generate_field_of_invention(
    innovation: str, 
    technology: str,
    approach: str, 
    antigen: str, 
    disease: str,
    additional:str, 
    model: str = model) -> FieldOfInvention:
    client = OpenAI()

    if not antigen or not disease:
        raise ValueError("Disease and antigen must be non-empty strings")

    trace_id = str(uuid.uuid4())
    trace = langfuse.trace(
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
    raw_prompt = langfuse.get_prompt("generate_field_of_invention")
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

    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    context: str, model: str = model
) -> DiseaseOverview:
    client = OpenAI()
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
            context=context
        ),
        tags=["evaluation"],
    )
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_disease_overview")
    prompt = raw_prompt.compile(
        disease=disease,
        additional=additional,
        context=context
    )
    fetch_prompt.end(
        end_time=datetime.now(),
        input=values_to_json(
            disease=disease,
            additional=additional,
            context=context
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
    additional:str, 
    model: str = model
) -> KeyTerms:
    client = OpenAI()
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
            additional=additional),
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
        additional=additional,        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="key_terms",
        input=prompt,
        completion_start_time=datetime.now(),
        model=model,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model
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
    client = OpenAI()
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
        messages=[{"role": "user", "content": prompt}], model=model
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
