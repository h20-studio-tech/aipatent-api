#!/usr/bin/env python3
"""
Demonstration of how position extraction would work with the enhanced schema.
"""

import re
from src.models.ocr_schemas import Embodiment


def demonstrate_position_extraction():
    """Show how the position extraction enhancement works."""
    
    # Sample page text
    page_text = """
DETAILED DESCRIPTION OF THE INVENTION

The present invention provides methods for treating diseases.

In certain aspects, the disclosure relates to a method for preventing or treating graft-versus-host disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product.

Furthermore, the methods can be combined with other treatments.

In certain embodiments, the hyperimmunized egg product comprises antibodies specific to Enterococcus faecalis.

Additional details follow.
"""
    
    print("🧪 Position Extraction Concept Demonstration\n")
    print(f"Page text length: {len(page_text)} characters\n")
    
    # Find embodiments using regex (simulating what the LLM should do)
    patterns = [
        r'In certain aspects[^.]+\.',
        r'In certain embodiments[^.]+\.'
    ]
    
    mock_embodiments = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, page_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # Create Embodiment with position information
            embodiment = Embodiment(
                text=match.group(0),
                filename="test.pdf",
                page_number=1,
                section="detailed description",
                start_char=match.start(),  # Character position where embodiment starts
                end_char=match.end(),      # Character position where embodiment ends
                summary=""
            )
            mock_embodiments.append(embodiment)
    
    # Display results
    print("📍 Embodiments with Character Positions:\n")
    
    for i, emb in enumerate(mock_embodiments):
        print(f"Embodiment {i+1}:")
        print(f"  Text: {emb.text[:60]}...")
        print(f"  Start position: {emb.start_char}")
        print(f"  End position: {emb.end_char}")
        print(f"  Length: {emb.end_char - emb.start_char} characters")
        
        # Validate extraction
        extracted = page_text[emb.start_char:emb.end_char]
        if extracted == emb.text:
            print(f"  ✅ Position validation: PASSED")
        else:
            print(f"  ❌ Position validation: FAILED")
        
        # Show context with highlighting
        context_start = max(0, emb.start_char - 20)
        context_end = min(len(page_text), emb.end_char + 20)
        context = page_text[context_start:context_end]
        
        # Highlight the embodiment
        rel_start = emb.start_char - context_start
        rel_end = emb.end_char - context_start
        highlighted = (
            "..." + context[:rel_start] + 
            "[" + context[rel_start:rel_end] + "]" + 
            context[rel_end:] + "..."
        )
        newline_char = chr(10)
        backslash_n = '\\n'
        print(f"  Context: {highlighted.replace(newline_char, backslash_n)}")
        print()
    
    # Explain the enhancement
    print("\n💡 How the Enhancement Works:\n")
    print("1. The Embodiment schema now includes start_char and end_char fields")
    print("2. When extracting embodiments, the LLM identifies exact character positions")
    print("3. The positions are validated by extracting text[start:end] and comparing")
    print("4. This enables precise source tracing back to the original document")
    print("\n✨ Benefits:")
    print("- Users can click on an embodiment and jump to its exact location")
    print("- Better debugging when extraction issues occur")
    print("- Enables highlighting in the source document")
    print("- Supports advanced features like context windows around embodiments")


if __name__ == "__main__":
    demonstrate_position_extraction()