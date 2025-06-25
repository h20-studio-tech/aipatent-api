from __future__ import annotations
import json
import re
from typing import List, Iterable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import instructor

# Patch the OpenAI client using instructor so that responses
# can be validated against Pydantic models
async_openai = instructor.from_openai(AsyncOpenAI())


class ChunkedSection(BaseModel):
    """Output schema for the semantic chunking step."""

    chunks: List[str] = Field(..., description="List of standalone chunks")


class Principle(BaseModel):
    """Represents an invention principle extracted from the patent."""

    id: str
    text: str
    source_claims: List[int]


class EmbodimentAnnotation(BaseModel):
    """LLM label for whether a chunk is an embodiment."""

    paragraph: str
    is_embodiment: bool
    justification: str
    mapped_principles: List[str]
    mapped_claims: List[int]


async def chunk_section(section_name: str, text: str) -> List[str]:
    """Break a section into semantically coherent chunks using the o3 model."""
    prompt = (
        f"Break the following {section_name} into concise, standalone technical chunks. "
        "Each chunk should describe one implementation detail, variation, or step. "
        "Keep each 2–6 sentences long. Return as a list of strings."
    )
    response = await async_openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt + "\n" + text}],
        model="o3",
        response_model=ChunkedSection,
    )
    return response.chunks


async def extract_principles(
    abstract: str, summary: str, claims: Iterable[str]
) -> List[Principle]:
    """Derive invention principles from abstract, summary and claims."""
    claims_text = "\n".join(claims)
    prompt = (
        "Based on the abstract, summary, and claims, generate a list of invention principles. "
        "Each principle should describe a core requirement of the invention. "
        "Return an ID (e.g., 'P1'), a short description, and the claims it relates to."
    )
    response = await async_openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
                + f"\nAbstract:\n{abstract}\nSummary:\n{summary}\nClaims:\n{claims_text}",
            }
        ],
        model="o3",
        response_model=List[Principle],
    )
    return response


_EMBODIMENT_PATTERNS = [
    r"in one (?:aspect|embodiment)",
    r"according to (?:the|one) (?:embodiment|implementation)",
    r"the method comprises",
]


def filter_candidates(chunks: Iterable[str]) -> List[str]:
    """Filter chunks using simple data-adaptive patterns."""
    pattern = re.compile("|".join(_EMBODIMENT_PATTERNS), re.IGNORECASE)
    return [c for c in chunks if pattern.search(c)]


async def classify_chunk(
    chunk: str, principles: List[Principle]
) -> EmbodimentAnnotation:
    """Classify a chunk as an embodiment and map to principles."""
    principles_text = "\n".join(f"{p.id}: {p.text}" for p in principles)
    prompt = (
        "Given the following invention principles:\n"
        + principles_text
        + "\n\n"
        + "Does the paragraph below describe a specific embodiment of one or more of these principles? "
        "If so, label it as an embodiment, explain why, and indicate which principles and claims it supports.\n\n"
        + f"Paragraph: \n{chunk}"
    )
    response = await async_openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3",
        response_model=EmbodimentAnnotation,
    )
    return response


def export_dataset(
    annotations: Iterable[EmbodimentAnnotation], file_path: str, source_section: str
) -> None:
    """Write annotations to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            record = ann.model_dump()
            record["source_section"] = source_section
            f.write(json.dumps(record) + "\n")


async def process_sections(sections: dict) -> None:
    """High level pipeline for a patent document."""
    summary_chunks = await chunk_section(
        "Summary of Invention", sections.get("summary", "")
    )
    desc_chunks = await chunk_section(
        "Detailed Description", sections.get("description", "")
    )
    claim_chunks = await chunk_section("Claims", "\n".join(sections.get("claims", [])))

    principles = await extract_principles(
        sections.get("abstract", ""),
        sections.get("summary", ""),
        sections.get("claims", []),
    )

    for name, chunks in {
        "summary": summary_chunks,
        "description": desc_chunks,
        "claims": claim_chunks,
    }.items():
        filtered = filter_candidates(chunks)
        annotations: List[EmbodimentAnnotation] = []
        for chunk in filtered:
            annotations.append(await classify_chunk(chunk, principles))
        export_dataset(annotations, f"labeled_{name}.jsonl", name)
