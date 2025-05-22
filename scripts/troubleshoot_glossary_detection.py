"""
Usage:
    python scripts/troubleshoot_glossary_detection.py path/to/patent.pdf [--save-json]

This helper script is intended for manual troubleshooting of the glossary-term
extraction pipeline.  It will perform the following steps on the supplied PDF
using the *real* OpenAI API (make sure the relevant environment variables are
set):

1. Convert the PDF to individual `ProcessedPage` objects (with OCR fallback).
2. Heuristically segment the pages into patent sections (summary, detailed
   description, claims).
3. Flag pages that are likely to contain glossary-style definitions.
4. Attempt to extract glossary definitions from the flagged pages.

It prints verbose information to the console and can optionally persist the
resulting glossary JSON next to the original PDF for further inspection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys, os
from typing import List, Set

# Ensure project root is on PYTHONPATH so that 'src' package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ocr import (
    pdf_pages,
    process_pdf_pages,
    segment_pages,
    detect_glossary_pages,
    extract_glossary_subsection,
)


async def _run(pdf_path: Path, save_json: bool) -> None:
    print(f"[INFO] Reading PDF → {pdf_path.resolve()}")
    pdf_data: bytes = pdf_path.read_bytes()

    # Step 1 – load and OCR/parse pages
    raw_pages = pdf_pages(pdf_data, pdf_path.name)
    processed_pages = process_pdf_pages(raw_pages)
    print(f"[INFO] Processed {len(processed_pages)} page(s) from PDF")

    # Step 2 – section segmentation
    segmented_pages = await segment_pages(processed_pages)
    sections: Set[str] = {p.section for p in segmented_pages}
    print(f"[INFO] Segmentation complete → sections detected: {', '.join(sections)}")

    # Step 3 – detect glossary pages
    detailed_description_pages = [page for page in segmented_pages if page.section == "detailed description"]
    glossary_page_results = await detect_glossary_pages(detailed_description_pages)
    glossary_pages = [page for page, flag in glossary_page_results if flag.is_glossary_page]
    print(
        f"[INFO] {len(glossary_pages)} page(s) flagged as glossary pages: "
        f"{[p.page_number for p in glossary_pages]}"
    )

    # Step 4 – extract glossary definitions
    glossary = await extract_glossary_subsection(glossary_pages)
    if glossary is None or not glossary.definitions:
        print("[WARN] No glossary definitions extracted.")
        return

    print(f"[INFO] Extracted {len(glossary.definitions)} glossary definition(s):")
    for d in glossary.definitions:
        print(f"   • {d.term}: {d.definition}")

    if save_json:
        out_path = pdf_path.with_suffix(".glossary.json")
        out_path.write_text(json.dumps(glossary.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Glossary JSON persisted to {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Troubleshoot glossary-term detection/extraction on a patent PDF."
    )
    parser.add_argument("pdf", type=Path, help="Path to the patent PDF to analyse")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Persist extracted glossary definitions to '<pdf>.glossary.json'",
    )

    args = parser.parse_args()
    if not args.pdf.exists() or not args.pdf.is_file():
        parser.error(f"File not found: {args.pdf}")

    try:
        asyncio.run(_run(args.pdf, args.save_json))
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")


if __name__ == "__main__":
    main()
