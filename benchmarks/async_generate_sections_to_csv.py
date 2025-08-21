import os
import csv
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from src.utils.ai import async_oai
from src.patent_sections_async import (
    generate_field_of_invention,
    generate_background,
    generate_summary,
    generate_target_overview,
    generate_high_level_concept,
    generate_underlying_mechanism,
    generate_embodiment,
    generate_disease_overview,
    generate_claims,
    generate_key_terms,
    generate_abstract,
)

load_dotenv()

# Inputs
innovation = "IgY-based neutralization therapy"
technology = "Immunoglobulin Y platform"
approach = "Targeted neutralization"
antigen = "SARS-CoV-2 Spike"
disease = "COVID-19"
additional = "Emphasize thermostability and low-cost production"
context = "Expression patterns and binding epitopes"

# Model (use exactly as provided by the user)
OAI_MODEL = "gpt-5"


async def generate_all_for_provider(provider_name: str, client, model: str):
    """Run all section generations concurrently for a given provider."""
    tasks = [
        ("field_of_invention", generate_field_of_invention(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("background_and_need", generate_background(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("brief_summary", generate_summary(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("target_overview", generate_target_overview(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            context=context,
            client=client,
            model=model,
        )),
        ("high_level_concept", generate_high_level_concept(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("underlying_mechanism", generate_underlying_mechanism(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("embodiment", generate_embodiment(
            previous_embodiment="",
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            client=client,
            model=model,
        )),
        ("disease_overview", generate_disease_overview(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("claims", generate_claims(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("key_terms", generate_key_terms(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
        ("abstract", generate_abstract(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=client,
            model=model,
        )),
    ]

    # Run concurrently
    results = await asyncio.gather(*[t for _, t in tasks])

    rows = []
    for (section_name, _), result in zip(tasks, results):
        rows.append({
            "provider": provider_name,
            "model": model,
            "section": section_name,
            "trace_id": result.trace_id,
            "content": result.prediction,
        })
    return rows


async def main():
    # Run OpenAI provider only (Gemini disabled due to API errors)
    provider_jobs = [
        generate_all_for_provider("openai", async_oai, OAI_MODEL),
    ]
    provider_rows = await asyncio.gather(*provider_jobs)

    # Flatten
    all_rows = [row for rows in provider_rows for row in rows]

    # Ensure outputs folder
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Timestamped single CSV
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"patent_sections_{ts}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["provider", "model", "section", "trace_id", "content"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
