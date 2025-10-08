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
    model: str = "gemini-2.5-flash"
) -> DiseaseSpecificOverview:
    """Generate disease-specific overview for a particular disease"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import DiseaseSpecificOverview
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_disease_specific_overview")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        disease_name=disease_name
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
            disease_name=disease_name
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="disease_specific_overview",
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
    model: str = "gemini-2.5-flash"
) -> TargetInDisease:
    """Generate target in disease subsection for specific target/disease combination"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import TargetInDisease
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_target_in_disease")
    prompt = raw_prompt.compile(
        technology=technology,
        antigen=antigen,
        disease=disease,
        innovation=innovation,
        approach=approach,
        additional=additional,
        target_name=target_name,
        disease_name=disease_name
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
            target_name=target_name,
            disease_name=disease_name
        ),
        output=raw_prompt,
    )

    generation = trace.generation(
        name="target_in_disease",
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
    model: str = "gemini-2.5-flash"
) -> EpidemiologyClinicalNeed:
    """Generate epidemiology and clinical need subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import EpidemiologyClinicalNeed
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_epidemiology_clinical_need")
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
        name="epidemiology_clinical_need",
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
    model: str = "gemini-2.5-flash"
) -> HyperimmunizedEggProducts:
    """Generate hyperimmunized egg products subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import HyperimmunizedEggProducts
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_hyperimmunized_egg_products")
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
        name="hyperimmunized_egg_products",
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
    model: str = "gemini-2.5-flash"
) -> AntigenicTargets:
    """Generate antigenic targets subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import AntigenicTargets
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_antigenic_targets")
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
        name="antigenic_targets",
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
    model: str = "gemini-2.5-flash"
) -> ProductionMethods:
    """Generate production methods subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import ProductionMethods
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_production_methods")
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
        name="production_methods",
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
    model: str = "gemini-2.5-flash"
) -> PharmaceuticalCompositions:
    """Generate pharmaceutical compositions subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import PharmaceuticalCompositions
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_pharmaceutical_compositions")
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
        name="pharmaceutical_compositions",
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
    model: str = "gemini-2.5-flash"
) -> KeyTerminology:
    """Generate key terminology subsection"""
    import uuid
    import os
    from datetime import datetime
    from openai import OpenAI
    from src.models.llm import KeyTerminology
    from src.utils.values_to_json import values_to_json
    from src.utils.langfuse_client import get_langfuse_instance

    langfuse = get_langfuse_instance()
    g_reasoning = os.getenv("g_reasoning")

    provider = os.getenv("AI_PROVIDER")
    if provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        )
    else:
        client = OpenAI()

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
    fetch_prompt = trace.span(name="fetch_prompt", start_time=datetime.now())
    raw_prompt = langfuse.get_prompt("generate_key_terminology")
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
        name="key_terminology",
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

    return KeyTerminology(
        prediction=response.choices[0].message.content, trace_id=trace_id
    )