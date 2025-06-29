#!/usr/bin/env python
"""Quick manual test for SectionHierarchy construction.

Loads a sample patent PDF, runs the relevant OCR utilities step-by-step,
   and prints the resulting hierarchy so we can visually inspect that:
   • Every detected header becomes a Subsection
   • Embodiments are grouped correctly

This **does not** hit the FastAPI endpoint – it works entirely on the
   low-level utils to validate logic before wiring into the API layer.
"""
import asyncio
import os
import sys
import json
from pprint import pprint

# Add project root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from time import time

from src.utils.ocr import (
    pdf_pages,
    process_pdf_pages,
    segment_pages,
    detect_section_headers,
    find_embodiments,
    add_headers_to_embodiments,
    build_section_hierarchy,
    summarize_subsections,
)

DEFAULT_PDF = os.path.join("experiments", "sample_patents", "COVID-19 NEUTRALIZING ANTIBODY DETE.pdf")

# Allow custom PDF path via command-line argument
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF


def load_pdf(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def run_test():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Sample PDF not found: {PDF_PATH}")
        return

    print("🧪 Running SectionHierarchy test on:", os.path.basename(PDF_PATH))

    pdf_data = load_pdf(PDF_PATH)
    pages, filename = pdf_pages(pdf_data, os.path.basename(PDF_PATH))

    # Process pages
    processed = process_pdf_pages((pages, filename))
    segmented = await segment_pages(processed)

    # Header detection
    headers_detected = await detect_section_headers(segmented)
    print(f"🔎 Detected {len([h for h in headers_detected if h.has_header])} headers (across {len(headers_detected)} pages)")

    # Embodiment extraction (can be slow)
    t0 = time()
    emb_list = await find_embodiments(segmented)
    print(f"✂️  Extracted {len(emb_list)} embodiments in {time()-t0:.1f}s")

    # Carry forward headers and attach to detailed-description embodiments
    from src.models.ocr_schemas import DetailedDescriptionEmbodiment
    dd_embs = [e for e in emb_list if e.section == "detailed description"]
    dd_with_headers = await add_headers_to_embodiments(dd_embs, headers_detected)

    # Replace in full list
    emb_with_headers = [e for e in emb_list if e.section != "detailed description"] + dd_with_headers

    # Build hierarchy
    hierarchy = await build_section_hierarchy(emb_with_headers, headers_detected)

    # Summarize subsections
    hierarchy = await summarize_subsections(hierarchy, headers_detected)

    # Pretty print hierarchy overview
    def summarise():
        out = {}
        for sec in hierarchy:
            out[sec.section] = {sub.header: len(sub.embodiments) for sub in sec.subsections}
        return out

    print("\n📐 Hierarchy summary (header -> embodiment count):")
    pprint(summarise(), sort_dicts=False, compact=True)

    # Dump JSON snapshot (optional)
    snapshot_path = os.path.join("experiments", "hierarchy_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump([sec.model_dump(mode="json") for sec in hierarchy], f, ensure_ascii=False, indent=2)
    print(f"💾 Full hierarchy JSON written to {snapshot_path}\n")


if __name__ == "__main__":
    asyncio.run(run_test())
