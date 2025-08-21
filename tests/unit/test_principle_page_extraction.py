from typing import List

import pytest

from src.principle_page_extraction import (
    PageEmbodimentAnnotation,
    process_pages,
)
from src.models.ocr_schemas import Embodiment, ProcessedPage
from src.embodiment_extraction import Principle


@pytest.mark.asyncio
async def test_process_pages_basic(monkeypatch):
    """process_pages should return Embodiment objects with correct page info."""

    pages: List[ProcessedPage] = [
        ProcessedPage(
            text=(
                "In one embodiment, the device comprises X.\n\n"
                "According to another embodiment, Y is provided.\n\n"
                "Background information only."
            ),
            filename="dummy.pdf",
            page_number=1,
            section="Summary of Invention",
            image=None,
        ),
        ProcessedPage(
            text="Irrelevant paragraph without the keyword.",
            filename="dummy.pdf",
            page_number=2,
            section="Detailed Description",
            image=None,
        ),
    ]

    # ----- stub out LLM-dependent functions -----
    async def fake_extract_principles(abstract: str, summary: str, claims: list):
        return [Principle(id="P1", text="Principle stub", source_claims=[1])]

    async def fake_classify_paragraph(paragraph: str, page: ProcessedPage, principles):
        is_emb = "embodiment" in paragraph.lower()
        return PageEmbodimentAnnotation(
            paragraph=paragraph,
            is_embodiment=is_emb,
            justification="stub",
            mapped_principles=["P1"] if is_emb else [],
            mapped_claims=[1] if is_emb else [],
            page_number=page.page_number,
            section=page.section,
        )

    monkeypatch.setattr(
        "src.principle_page_extraction.extract_principles", fake_extract_principles
    )
    monkeypatch.setattr(
        "src.principle_page_extraction._classify_paragraph", fake_classify_paragraph
    )

    res = await process_pages(pages, {"abstract": "", "summary": "", "claims": []})

    # Only the paragraphs with the keyword should be returned (2 from page 1)
    assert len(res) == 2
    assert all(isinstance(e, Embodiment) for e in res)
    assert {e.page_number for e in res} == {1}
