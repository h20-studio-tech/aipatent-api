#!/usr/bin/env python3
"""
Test position extraction using preprocessed patent partition file.
"""

import json
import asyncio
from src.models.ocr_schemas import ProcessedPage
from src.utils.ocr import get_embodiments


async def test_partition_position_extraction():
    """Test position extraction with preprocessed patent data."""
    
    partition_file = "experiments/partitions/ald_gvhd_provisional_patent_processed.json"
    
    print(f"📄 Loading partition file: {partition_file}")
    
    try:
        with open(partition_file, 'r') as f:
            partition_data = json.load(f)
        
        print(f"✅ Loaded partition data")
        
        # The partition data is a list of pages
        if isinstance(partition_data, list):
            print(f"   Pages: {len(partition_data)}")
            pages_to_test = partition_data[:3]  # Test first 3 pages
        else:
            print(f"   Pages: {len(partition_data.get('pages', []))}")
            pages_to_test = partition_data.get('pages', [])[:3]
        
        all_embodiments = []
        
        for page_data in pages_to_test:
            # Create ProcessedPage from partition data
            page = ProcessedPage(
                text=page_data.get('text', ''),
                filename=page_data.get('filename', 'ald_gvhd_provisional_patent.pdf'),
                page_number=page_data.get('page_number', 1),
                section=page_data.get('section', 'unknown')
            )
            
            print(f"\n📃 Processing page {page.page_number} ({page.section})")
            print(f"   Text length: {len(page.text)} characters")
            
            # Extract embodiments with positions
            try:
                embodiments = await get_embodiments(page)
                all_embodiments.extend(embodiments)
                
                print(f"   Found {len(embodiments)} embodiments")
                
                for i, emb in enumerate(embodiments):
                    print(f"\n   Embodiment {i+1}:")
                    print(f"     Text: {emb.text[:60]}...")
                    
                    if hasattr(emb, 'start_char') and hasattr(emb, 'end_char') and emb.start_char is not None:
                        print(f"     Position: [{emb.start_char}:{emb.end_char}]")
                        
                        # Validate the position
                        extracted = page.text[emb.start_char:emb.end_char]
                        matches = extracted.strip() == emb.text.strip()
                        
                        if matches:
                            print(f"     ✅ Position valid")
                        else:
                            print(f"     ❌ Position mismatch")
                            print(f"        Expected: '{emb.text[:40]}...'")
                            print(f"        Got: '{extracted[:40]}...'")
                            
                            # Show character context
                            start_context = max(0, emb.start_char - 10)
                            end_context = min(len(page.text), emb.end_char + 10)
                            context = page.text[start_context:end_context]
                            print(f"        Context: ...{repr(context)}...")
                    else:
                        print(f"     ⚠️  No position information")
                
            except Exception as e:
                print(f"   ❌ Error processing page: {e}")
        
        # Summary
        print(f"\n📊 Overall Summary:")
        print(f"   Total embodiments found: {len(all_embodiments)}")
        
        with_positions = sum(1 for e in all_embodiments if hasattr(e, 'start_char') and e.start_char is not None)
        print(f"   With positions: {with_positions}")
        print(f"   Position coverage: {with_positions/len(all_embodiments)*100:.1f}%" if all_embodiments else "N/A")
        
        # Check if we're getting positions from the LLM
        if with_positions == 0 and all_embodiments:
            print("\n⚠️  No positions were extracted. The LLM might not be providing character positions.")
            print("   Check that the prompt is requesting start_char and end_char correctly.")
        
    except FileNotFoundError:
        print(f"❌ Partition file not found: {partition_file}")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_partition_position_extraction())