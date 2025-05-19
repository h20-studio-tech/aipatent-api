import pytest
import json
from pathlib import Path

from src.utils.ocr import (
    pdf_pages,
    process_pdf_pages,
    segment_pages,
    extract_glossary_subsection,
    client
)
from src.models.ocr_schemas import GlossarySubsectionPage, GlossaryDefinition

@pytest.fixture
def snake_bite_pdf_path():
    return Path("experiments/sample_patents/snake_bites.pdf")

@pytest.mark.asyncio
async def test_extract_glossary_and_store_results(snake_bite_pdf_path, monkeypatch):
    # Patch the LLM client to avoid real API calls
    async def dummy_create(model, reasoning_effort, messages, response_model):
        return response_model(
            definitions=[GlossaryDefinition(term="venom", definition="toxin")],
            text="",
            filename=snake_bite_pdf_path.name,
            page_number=1,
            section="detailed description -> definitions"
        )
    monkeypatch.setattr(client.chat.completions, "create", dummy_create)

    # Read the real PDF file
    pdf_data = snake_bite_pdf_path.read_bytes()
    raw_pages = pdf_pages(pdf_data, snake_bite_pdf_path.name)
    processed_pages = process_pdf_pages(raw_pages)
    segmented_pages = await segment_pages(processed_pages)

    # Extract glossary subsection
    result = await extract_glossary_subsection(segmented_pages)

    # Prepare output directory and file
    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{snake_bite_pdf_path.stem}_glossary.json"

    # Serialize and store results
    data = result.model_dump() if result else None
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Assertions
    assert result is not None, "No glossary definitions extracted"
    assert isinstance(result, GlossarySubsectionPage)
    assert result.definitions, "Extracted definitions list is empty"
