from __future__ import annotations

"""Page-centric principle-based embodiment extraction.

This module reuses the existing ``extract_principles`` function but performs
classification *page by page* so that each resulting embodiment keeps its
`page_number` (and `section`) context.  It outputs a list of Pydantic
``Embodiment`` objects ready to be stored exactly like the current OCR-based
pipeline.
"""

import json
import re
import asyncio
from typing import List

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.embodiment_extraction import Principle, extract_principles
from src.models.ocr_schemas import Embodiment, ProcessedPage

# Initialise OpenAI with instructor patch so we can directly enforce schemas
async_openai = instructor.from_openai(AsyncOpenAI())


class PageEmbodimentAnnotation(BaseModel):
    """LLM label for a paragraph, preserving page context."""

    paragraph: str
    is_embodiment: bool
    justification: str
    mapped_principles: List[str]
    mapped_claims: List[int]
    filename: str
    page_number: int = Field(..., description="Page number where the paragraph appears")
    section: str = Field(..., description="Document section for the page (e.g. Summary)")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------



async def _classify_paragraph(
    paragraph: str,
    page: ProcessedPage,
    principles: List[Principle],
) -> PageEmbodimentAnnotation:
    """Ask the LLM if *paragraph* is an embodiment and map to principles/claims."""

    principles_text = "\n".join(f"{p.id}: {p.text}" for p in principles)

    prompt = (
        f"You are analysing page {page.page_number} of a patent (section: {page.section}).\n"
        f"Consider these invention principles:\n{principles_text}\n\n"
        "Does the paragraph below describe a *specific embodiment* (i.e. an implementation) "
        "of at least one of the principles? If yes, label it as an embodiment, provide a brief "
        "justification, and list the principle IDs and claim numbers it supports. If no, mark "
        "it as not an embodiment.\n\n"
        f"Paragraph:\n{paragraph}"
    )

    ann: PageEmbodimentAnnotation = await async_openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3",
        response_model=PageEmbodimentAnnotation,
    )
    ann.filename = page.filename
    return ann


# ---------------------------------------------------------------------------
# Paragraph utilities
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str, sentences_per_paragraph: int = 3) -> List[str]:
    """Return coarse paragraphs; fallback to sentence grouping when no blank lines."""
    # Primary: blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        # Fallback: group sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        paras = [" ".join(sentences[i : i + sentences_per_paragraph]) for i in range(0, len(sentences), sentences_per_paragraph)]
        paras = [p.strip() for p in paras if p.strip()]
    return paras

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def process_pages(
    processed_pages: List[ProcessedPage],
    sections: dict,
    export_path: str | None = None,
    max_concurrency: int = 50,
) -> List[Embodiment]:
    """End-to-end page-based extraction pipeline.

    Parameters
    ----------
    processed_pages: List[ProcessedPage]
        Pages already segmented and labelled with their `.section`.
    sections: dict
        Raw text for ``{"abstract": str, "summary": str, "description": str, "claims": List[str]}``.
        Used only for the principle derivation step.
    export_path: str | None
        If provided, write label annotations to the given JSONL file (for fine-tuning).

    Returns
    -------
    List[Embodiment]
        One Embodiment object per positive annotation.
    """

    principles = await extract_principles(
        sections.get("abstract", ""),
        sections.get("summary", ""),
        sections.get("claims", []),
    )

    sem = asyncio.Semaphore(max_concurrency)
    tasks: list[asyncio.Task[PageEmbodimentAnnotation]] = []

    async def _classify_with_limit(para: str, page: ProcessedPage):
        async with sem:
            return await _classify_paragraph(para, page, principles)

    for page in processed_pages:
        for para in _split_into_paragraphs(page.text):
            tasks.append(asyncio.create_task(_classify_with_limit(para, page)))

    annotations: List[PageEmbodimentAnnotation] = await asyncio.gather(*tasks)

    # Optional: persist annotations for offline inspection / dataset building
    if export_path:
        with open(export_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann.model_dump()) + "\n")

    # Convert positive annotations -> Embodiment objects
    embodiments: List[Embodiment] = []
    for ann in annotations:
        if ann.is_embodiment:
            embodiments.append(
                Embodiment(
                    text=ann.paragraph,
                    filename=ann.filename,
                    page_number=ann.page_number,
                    section=ann.section,
                    summary="",
                    start_char=0,  # Default value - this module doesn't track positions
                    end_char=len(ann.paragraph),  # Default to text length
                )
            )

    return embodiments
