#!/usr/bin/env python3
"""
Test character position extraction with ALD_GvHD provisional patent.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.ocr import process_patent_document
from src.utils.position_debug import validate_position_extraction, show_text_with_positions


async def test_patent_position_extraction():
    """Test position extraction on the ALD_GvHD patent document."""
    
    # Read the patent file
    patent_path = "experiments/sample_patents/test file.pdf"
    
    if not os.path.exists(patent_path):
        print(f"❌ Patent file not found: {patent_path}")
        return
    
    print(f"📄 Processing patent: {patent_path}")
    
    with open(patent_path, 'rb') as f:
        pdf_data = f.read()
    
    filename = os.path.basename(patent_path)
    
    try:
        # Process the patent document
        print("\n🔄 Processing patent document...")
        glossary, embodiments, sections, raw_sections, segmented_pages = await process_patent_document(
            pdf_data, filename
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   - Pages processed: {len(segmented_pages)}")
        print(f"   - Embodiments found: {len(embodiments)}")
        print(f"   - Sections identified: {len(sections)}")
        
        # Check position extraction results
        print("\n📍 Checking character positions...")
        
        embodiments_with_positions = 0
        position_validations = []
        
        for i, emb in enumerate(embodiments):
            if hasattr(emb, 'start_char') and hasattr(emb, 'end_char') and emb.start_char is not None:
                embodiments_with_positions += 1
                
                # Find the page this embodiment came from
                page_text = None
                for page in segmented_pages:
                    if page.page_number == emb.page_number:
                        page_text = page.text
                        break
                
                if page_text:
                    # Validate the position
                    validation = validate_position_extraction(
                        page_text, 
                        emb.text, 
                        emb.start_char, 
                        emb.end_char
                    )
                    position_validations.append(validation)
                    
                    if validation['valid']:
                        print(f"\n✅ Embodiment {i+1} (Page {emb.page_number}):")
                        print(f"   Position: [{emb.start_char}:{emb.end_char}]")
                        print(f"   Text: {emb.text[:80]}...")
                    else:
                        print(f"\n❌ Embodiment {i+1} (Page {emb.page_number}):")
                        print(f"   Position: [{emb.start_char}:{emb.end_char}]")
                        print(f"   Error: {validation['error']}")
                        print(validation['debug_view'])
        
        # Summary statistics
        print(f"\n📊 Position Extraction Summary:")
        print(f"   - Total embodiments: {len(embodiments)}")
        print(f"   - With positions: {embodiments_with_positions}")
        print(f"   - Position coverage: {embodiments_with_positions/len(embodiments)*100:.1f}%")
        
        if position_validations:
            valid_count = sum(1 for v in position_validations if v['valid'])
            print(f"   - Valid positions: {valid_count}/{len(position_validations)}")
            print(f"   - Validation rate: {valid_count/len(position_validations)*100:.1f}%")
        
        # Show some example pages with positions
        print("\n📃 Example Page Analysis:")
        for page_num in [1, 5, 10]:  # Sample pages
            page = next((p for p in segmented_pages if p.page_number == page_num), None)
            if page:
                page_embodiments = [e for e in embodiments if e.page_number == page_num and hasattr(e, 'start_char')]
                if page_embodiments:
                    print(f"\n--- Page {page_num} ---")
                    print(f"Section: {page.section}")
                    print(f"Embodiments with positions: {len(page_embodiments)}")
                    
                    # Show first embodiment position
                    if page_embodiments:
                        emb = page_embodiments[0]
                        if emb.start_char is not None:
                            print(show_text_with_positions(
                                page.text, 
                                emb.start_char, 
                                emb.end_char,
                                context_chars=30
                            ))
        
    except Exception as e:
        print(f"\n❌ Error processing patent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_patent_position_extraction())