import pytest
import os
import json
from pathlib import Path

from src.utils.ocr import (
    pdf_pages,
    process_pdf_pages,
    segment_pages,
    extract_glossary_subsection,
)
from src.models.ocr_schemas import Glossary
@pytest.fixture
def snake_bite_pdf_path():
    return str(Path("experiments\sample_patents\ALD_GvHD provisional patent.pdf"))

@pytest.mark.asyncio
async def test_extract_glossary_subsection_from_snake_bites(snake_bite_pdf_path):

    # Read and preprocess pages
    with open(snake_bite_pdf_path, "rb") as f:
        pdf_data = f.read()
    raw_pages = pdf_pages(pdf_data, os.path.basename(snake_bite_pdf_path))
    processed_pages = process_pdf_pages(raw_pages)
    segmented_pages = await segment_pages(processed_pages)

    # Extract glossary
    result = await extract_glossary_subsection(segmented_pages)
    
    # Save to JSON
    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(snake_bite_pdf_path).stem}_glossary.json"
    with open(out_file, "w", encoding="utf-8") as wf:
        json.dump(result.model_dump() if result else None, wf, indent=2)

    # Assertions
    assert isinstance(result, Glossary)
    assert result.definitions, "Definitions list is empty"
    for d in result.definitions:
        assert d.term, "Term should be non-empty"
        assert d.definition, "Definition should be non-empty"
    assert result.filename == os.path.basename(snake_bite_pdf_path)
    assert result.page_number >= 1
    assert result.section == "detailed description -> definitions"

@pytest.mark.asyncio
async def test_extract_glossary_subsection_no_detailed():
    # When no detailed description pages, should return None
    result = await extract_glossary_subsection([])
    assert result is None