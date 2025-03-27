import asyncio
import os
import sys
import time
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.ocr import pdf_pages, process_pdf_pages, segment_pages
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def test_full_document(pdf_path):
    """Test section detection on a full patent document with enhanced logging"""
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*20} Testing {filename} {'='*20}")
    
    # Read the file
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    # Process all pages
    print(f"Processing PDF: {pdf_path}")
    pages_tuple = pdf_pages(pdf_data, filename)
    
    print(f"Processing all {len(pages_tuple[0])} pages")
    
    # Time the processing
    start_time = time.time()
    processed_pages = process_pdf_pages(pages_tuple)
    process_time = time.time() - start_time
    print(f"Processed {len(processed_pages)} pages in {process_time:.2f} seconds")
    
    # Perform section detection on the full document
    print("\nPerforming section detection with enhanced logging...")
    start_time = time.time()
    
    segmented_pages = await segment_pages(processed_pages)
    
    detection_time = time.time() - start_time
    print(f"\nCompleted section detection in {detection_time:.2f} seconds")
    
    # Print results
    print("\nSection detection results:")
    print("=" * 80)
    
    # Group pages by section for a concise summary
    sections = {}
    for page in sorted(segmented_pages, key=lambda p: p.page_number):
        if page.section not in sections:
            sections[page.section] = []
        sections[page.section].append(page.page_number)
    
    print("\nSection distribution:")
    for section, pages in sections.items():
        page_range = f"Pages {min(pages)}-{max(pages)}"
        print(f"{section}: {page_range} ({len(pages)} pages)")
    
    # Print section transitions for clarity
    print("\nSection transitions:")
    print("=" * 40)
    prev_section = None
    
    for page in sorted(segmented_pages, key=lambda p: p.page_number):
        if page.section != prev_section:
            if prev_section is not None:
                print(f"Page {page.page_number}: {prev_section} -> {page.section}")
            else:
                print(f"Page {page.page_number}: Starting with {page.section}")
            prev_section = page.section
    
    return segmented_pages

async def full_document_test():
    """Run a test on the full patent document"""
    # Select a specific patent file
    sample_dir = os.path.join(project_root, "experiments", "sample_patents")
    
    # Choose the file that showed section headers at the end of pages
    target_file = "ALD_GvHD provisional patent.pdf"
    pdf_path = os.path.join(sample_dir, target_file)
    
    print("\nRUNNING FULL DOCUMENT TEST")
    print("=" * 60)
    print("Processing the entire document at once with enhanced logging")
    print("This will show exactly what triggers each section detection")
    print("=" * 60)
    
    # Process the full document
    await test_full_document(pdf_path)
    
    print("\nFull document test completed")

if __name__ == "__main__":
    asyncio.run(full_document_test())
