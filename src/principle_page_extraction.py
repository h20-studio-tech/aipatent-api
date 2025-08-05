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
from typing import List, Tuple

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
    start_char: int = Field(0, description="Starting character position in the page")
    end_char: int = Field(0, description="Ending character position in the page")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------



async def _classify_paragraph(
    paragraph: str,
    page: ProcessedPage,
    principles: List[Principle],
    start_char: int,
    end_char: int,
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
    ann.start_char = start_char
    ann.end_char = end_char
    return ann


# ---------------------------------------------------------------------------
# Paragraph utilities
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str, sentences_per_paragraph: int = 3) -> List[Tuple[str, int, int]]:
    """Return paragraphs with their character positions.
    
    Returns:
        List of tuples (paragraph_text, start_pos, end_pos)
    """
    results = []
    
    # Primary: split by blank lines
    parts = re.split(r'(\n\s*\n)', text)
    
    if len([p for p in parts if p.strip() and not re.match(r'^\n\s*\n$', p)]) > 1:
        # We have multiple paragraphs separated by blank lines
        current_pos = 0
        for part in parts:
            if part.strip() and not re.match(r'^\n\s*\n$', part):
                # This is actual content, not a separator
                stripped = part.strip()
                # Find where the stripped content starts in the original part
                strip_offset = part.find(stripped)
                start_pos = current_pos + strip_offset
                end_pos = start_pos + len(stripped)
                results.append((stripped, start_pos, end_pos))
            current_pos += len(part)
    else:
        # Fallback: group sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(f'({sentence_pattern})', text)
        
        # Reconstruct sentences with their positions
        current_pos = 0
        sentence_groups = []
        
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(sentence_pattern, sentences[i + 1]):
                # This is a sentence followed by separator
                sentence_groups.append((sentences[i] + sentences[i + 1], current_pos))
                current_pos += len(sentences[i]) + len(sentences[i + 1])
                i += 2
            else:
                # This is a sentence without separator or last sentence
                sentence_groups.append((sentences[i], current_pos))
                current_pos += len(sentences[i])
                i += 1
        
        # Group sentences into paragraphs
        for i in range(0, len(sentence_groups), sentences_per_paragraph):
            group = sentence_groups[i:i + sentences_per_paragraph]
            if group:
                combined_text = ''.join(s[0] for s in group).strip()
                if combined_text:
                    start_pos = group[0][1]
                    # Find actual start of non-whitespace content
                    first_sentence = group[0][0]
                    strip_offset = len(first_sentence) - len(first_sentence.lstrip())
                    start_pos += strip_offset
                    end_pos = start_pos + len(combined_text)
                    results.append((combined_text, start_pos, end_pos))
    
    return results

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

    async def _classify_with_limit(para: str, page: ProcessedPage, start: int, end: int):
        async with sem:
            return await _classify_paragraph(para, page, principles, start, end)

    for page in processed_pages:
        for para_text, start_pos, end_pos in _split_into_paragraphs(page.text):
            tasks.append(asyncio.create_task(_classify_with_limit(para_text, page, start_pos, end_pos)))

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
                    start_char=ann.start_char,
                    end_char=ann.end_char
                )
            )

    return embodiments
