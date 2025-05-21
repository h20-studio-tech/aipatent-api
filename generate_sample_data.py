import os
import sys
import json
import asyncio
from time import time

def ensure_project_root_in_path():
    """Ensure src directory is on sys.path for imports."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

ensure_project_root_in_path()

from src.utils.ocr import pdf_pages, process_pdf_pages, segment_pages
from src.models.ocr_schemas import ProcessedPage

async def process_folder(input_folder: str) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(input_folder, fname)
        with open(path, 'rb') as f:
            pdf_data = f.read()
        # Extract raw pages
        pages, filename = pdf_pages(pdf_data, fname)
        # Process PDF pages (text and images)
        processed_pages = process_pdf_pages((pages, filename))
        # Segment pages into sections
        segmented = await segment_pages(processed_pages)
        # Convert each ProcessedPage to dict
        page_dicts = [page.model_dump() for page in segmented]
        results[fname] = page_dicts
    return results

async def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_folder> <output_json>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    start = time()
    data = await process_folder(input_folder)
    duration = time() - start
    print(f"Processed {len(data)} files in {duration:.2f} seconds")
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    print(f"Wrote output to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())
