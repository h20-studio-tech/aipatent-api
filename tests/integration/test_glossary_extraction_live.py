import os
import json
import pytest
from pathlib import Path

from src.utils.ocr import (
    pdf_pages,
    process_pdf_pages,
    segment_pages,
    extract_glossary_subsection,
)
from src.models.ocr_schemas import GlossarySubsectionPage

@pytest.fixture
def snake_bite_pdf_path():
    return Path("experiments/sample_patents/snake_bites.pdf")

@pytest.mark.asyncio
async def test_extract_glossary_live(snake_bite_pdf_path):
    # Skip test if no API key provided
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping live glossary extraction test")

    # Read the real PDF file
    pdf_data = snake_bite_pdf_path.read_bytes()
    raw_pages = pdf_pages(pdf_data, snake_bite_pdf_path.name)
    processed_pages = process_pdf_pages(raw_pages)
    segmented_pages = await segment_pages(processed_pages)

    # Extract glossary subsection using real API
    result = await extract_glossary_subsection(segmented_pages)

    # Store results to JSON
    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{snake_bite_pdf_path.stem}_glossary_live.json"
    data = result.model_dump() if result else None
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Assertions on live response
    assert result is not None, "No glossary definitions extracted (live run)"
    assert isinstance(result, GlossarySubsectionPage)
    assert result.definitions, "Extracted definitions list is empty (live run)"
