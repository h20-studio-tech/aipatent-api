#!/usr/bin/env python3
"""
Simple test of position extraction logic without full pipeline.
"""

import asyncio
from src.models.ocr_schemas import ProcessedPage, Embodiment
from src.utils.ocr import get_embodiments


async def test_simple_position_extraction():
    """Test position extraction with a simple example."""
    
    # Create a test page with known embodiments
    test_page = ProcessedPage(
        text="""
DETAILED DESCRIPTION OF THE INVENTION

The present invention provides methods and compositions for treating various diseases.

In certain aspects, the disclosure relates to a method for preventing or treating graft-versus-host disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product.

Furthermore, the methods can be combined with standard treatments.

In certain embodiments, the hyperimmunized egg product comprises antibodies specific to Enterococcus faecalis.

In some embodiments, the effective amount is between 100 mg and 1000 mg per day.

Additional details are provided below.
""",
        filename="test.pdf",
        page_number=1,
        section="detailed description"
    )
    
    print("🧪 Testing embodiment position extraction...")
    print(f"Page text length: {len(test_page.text)} characters\n")
    
    # Extract embodiments with positions
    try:
        embodiments = await get_embodiments(test_page)
        
        print(f"✅ Found {len(embodiments)} embodiments\n")
        
        for i, emb in enumerate(embodiments):
            print(f"Embodiment {i+1}:")
            print(f"  Text: {emb.text[:80]}...")
            
            if hasattr(emb, 'start_char') and hasattr(emb, 'end_char'):
                print(f"  Position: [{emb.start_char}:{emb.end_char}]")
                
                # Validate the position
                extracted = test_page.text[emb.start_char:emb.end_char]
                matches = extracted.strip() == emb.text.strip()
                
                if matches:
                    print(f"  ✅ Position validation: PASSED")
                else:
                    print(f"  ❌ Position validation: FAILED")
                    print(f"     Expected: '{emb.text[:50]}...'")
                    print(f"     Got: '{extracted[:50]}...'")
            else:
                print(f"  ⚠️  No position information")
            
            print()
        
        # Summary
        with_positions = sum(1 for e in embodiments if hasattr(e, 'start_char') and e.start_char is not None)
        print(f"\n📊 Summary:")
        print(f"   Total embodiments: {len(embodiments)}")
        print(f"   With positions: {with_positions}")
        print(f"   Position coverage: {with_positions/len(embodiments)*100:.1f}%" if embodiments else "N/A")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_position_extraction())