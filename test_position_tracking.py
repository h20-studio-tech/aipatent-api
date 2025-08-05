#!/usr/bin/env python3
"""Test script to verify character position tracking with partition JSON files."""

import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ocr_schemas import ProcessedPage
from src.principle_page_extraction import process_pages


async def load_and_process_partition(partition_file: str):
    """Load a partition JSON file and process it with position tracking."""
    
    print(f"Loading partition file: {partition_file}")
    
    # Load the partition data
    with open(partition_file, 'r') as f:
        partition_data = json.load(f)
    
    # Convert to ProcessedPage objects
    processed_pages = []
    for page_data in partition_data:
        page = ProcessedPage(
            text=page_data['text'],
            filename=page_data['filename'],
            page_number=page_data['page_number'],
            section=page_data['section'],
            image=None  # No images in partition files
        )
        processed_pages.append(page)
    
    print(f"Loaded {len(processed_pages)} pages")
    
    # Extract raw sections for principle extraction
    sections_dict = {}
    for page in processed_pages:
        section = page.section.lower()
        if section not in sections_dict:
            sections_dict[section] = []
        sections_dict[section].append(page.text)
    
    # Prepare sections for principle extraction
    principle_sections = {
        "abstract": "\n\n".join(sections_dict.get("abstract", [])),
        "summary": "\n\n".join(sections_dict.get("summary of invention", [])),
        "description": "\n\n".join(sections_dict.get("detailed description", [])),
        "claims": sections_dict.get("claims", [])
    }
    
    print("\nProcessing pages to extract embodiments with position tracking...")
    
    # Process pages with our updated position tracking
    embodiments = await process_pages(
        processed_pages, 
        principle_sections,
        max_concurrency=10  # Lower concurrency for testing
    )
    
    print(f"\nExtracted {len(embodiments)} embodiments")
    
    # Analyze position tracking results
    print("\nAnalyzing position tracking:")
    print("-" * 80)
    
    for i, emb in enumerate(embodiments[:5]):  # Show first 5 for brevity
        print(f"\nEmbodiment {i+1}:")
        print(f"  Page: {emb.page_number}")
        print(f"  Section: {emb.section}")
        print(f"  Position: [{emb.start_char}:{emb.end_char}]")
        print(f"  Text length: {len(emb.text)}")
        print(f"  Text preview: {emb.text[:100]}...")
        
        # Verify position tracking
        page = next((p for p in processed_pages if p.page_number == emb.page_number), None)
        if page:
            extracted = page.text[emb.start_char:emb.end_char]
            matches = extracted.strip() == emb.text.strip()
            print(f"  Position tracking valid: {matches}")
            if not matches:
                print(f"    Expected: {emb.text[:50]}...")
                print(f"    Got: {extracted[:50]}...")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print(f"  Total embodiments: {len(embodiments)}")
    
    # Check how many have non-zero start positions
    non_zero_starts = sum(1 for emb in embodiments if emb.start_char > 0)
    print(f"  Embodiments with non-zero start_char: {non_zero_starts}/{len(embodiments)}")
    
    # Group by page
    by_page = {}
    for emb in embodiments:
        if emb.page_number not in by_page:
            by_page[emb.page_number] = []
        by_page[emb.page_number].append(emb)
    
    print(f"  Pages with embodiments: {len(by_page)}")
    for page_num, embs in sorted(by_page.items())[:5]:
        print(f"    Page {page_num}: {len(embs)} embodiments")
    
    return embodiments


async def main():
    """Main test function."""
    partition_file = "/workspaces/aipatent-api/tests/outputs/snake_bites_partition.json"
    
    try:
        await load_and_process_partition(partition_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())