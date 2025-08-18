import asyncio
from src.utils.ai import async_oai
from src.patent_sections_async import (
    generate_field_of_invention,
    generate_summary,
    generate_high_level_concept,
)


async def main():
    # Example inputs
    innovation = "IgY-based neutralization therapy"
    technology = "Immunoglobulin Y platform"
    approach = "Targeted neutralization"
    antigen = "SARS-CoV-2 Spike"
    disease = "COVID-19"
    additional = "Emphasize thermostability and low-cost production"
    context = "Expression patterns and binding epitopes"

    # Run multiple section generations concurrently on the same async client
    tasks = [
        generate_field_of_invention(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=async_oai,
        ),
        generate_summary(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=async_oai,
        ),
        generate_high_level_concept(
            innovation=innovation,
            technology=technology,
            approach=approach,
            antigen=antigen,
            disease=disease,
            additional=additional,
            client=async_oai,
        ),
    ]

    results = await asyncio.gather(*tasks)

    for r in results:
        print("\n=== Result ===")
        print("trace_id:", r.trace_id)
        print(r.prediction[:500], "...")


if __name__ == "__main__":
    asyncio.run(main())
