import os
from src.make_patent_component import (
    generate_abstract,
    generate_background,
    generate_claims,
    generate_disease_overview,
    generate_field_of_invention,
    generate_high_level_concept,
    generate_summary,
    generate_key_terms,
    generate_target_overview,
    generate_underlying_mechanism,
)
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

oai =  OpenAI()

rs = oai.chat.completions.create(
    model='gpt-5-2025-08-07',
    messages=[
        {'role': 'system', 'content' : 'hello there'}
    ]
)

gemini = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


response = gemini.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ],
)