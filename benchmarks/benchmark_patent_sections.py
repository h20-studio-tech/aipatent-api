import os
from src.patent_sections import (
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
from src.utils.ai import oai, gemini
from dotenv import load_dotenv

load_dotenv()

# Test parameters
test_params = {
    "innovation": "mRNA vaccine technology",
    "technology": "lipid nanoparticles",
    "approach": "targeted delivery",
    "antigen": "spike protein",
    "disease": "COVID-19",
    "additional": "enhanced immune response"
}

# Test GPT-5
print("Testing GPT-5...")
result_gpt5 = generate_field_of_invention(
    **test_params,
    client=oai,
    model="gpt-5-2025-08-07"
)
print(f"GPT-5 Result: {result_gpt5.prediction[:100]}...")

# Test Gemini
print("\nTesting Gemini...")
result_gemini = generate_field_of_invention(
    **test_params,
    client=gemini,
    model="gemini-2.5-pro"
)
print(f"Gemini Result: {result_gemini.prediction[:100]}...")

print("\nBenchmark complete!")
