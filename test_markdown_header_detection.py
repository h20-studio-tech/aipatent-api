#!/usr/bin/env python3
"""
Test script to demonstrate the new markdown-based header detection.
"""

import asyncio
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.ocr_schemas import ProcessedPage
from src.utils.ocr import detect_description_header_from_text

# Your example markdown content
test_markdown = """

WO 2022/221616                                                       PCT/US2022/024946

NaCl at pH 4 and allowed to precipitate for 2 h. IgY was separated from the precipitate by centrifugation and excess salt was removed by dialysis in PBS. Total IgY was quantified by A280 (absorption at 280 nm) values using a NanoDrop™ One Microvolume UV-Vis Spectrophotometer (Thermo Scientific). Purity of IgY was analyzed on an SDS-PAGE gel.

Detection of antibody specificity and titer by ELISA

Antibody specificity and titer was determined by ELISA. Briefly, to determine anti- E. faecalis antibody titer, a 96 well plate was coated with 1 mg/mL of E. faecalis lysate in carbonate-bicarbonate buffer (pH 9.3) and incubated overnight at 4°C. The plate was washed once with PBS and blocked with 1% BSA in PBS for 1 h at 37°C. The plate was washed once with PBS containing 0.05 wt% Tween-20 (PBST) and treated with primary anti-E. faecalis antibody serially diluted on the plate for 1 h at 37°C. The plate was washed and treated with goat anti-chicken HRP secondary antibody. TMB substrate was added to develop the signal and the reaction was stopped by HCl. The plate was read at 450 nm on a plate reader. To determine anti-cytolysin antibody titer, another 96 well plate was coated with 0.5 mg/ml cytolysin and a similar experiment was performed using anti-cytolysin antibodies.

Test for neutralization abilities of IgY

Enterococcus faecalis was cultured overnight in BHI Broth to an OD600 of 1.5 and was diluted 1:50 in the broth and incubated until OD600 0.3. Culture was further diluted further to 10³ cfu/ml in BHI broth and was treated with antibodies at 15 mg/ml. After 24 h incubation at 37°C, the cultures were serially diluted 10⁴ fold and plated on sterile plates and incubated at 37°C for 12 hours. Cfu/mL was quantitated by counting the number of colonies using ImageJ Count particle's function. Percent inhibition of bacterial colony formation was calculated.

Results

Total soluble proteins including IgY from hyperimmunized chicken egg yolks were extracted by water dilution and NaCl method. Purity of IgY was analyzed by SDS-PAGE electrophoresis under reducing conditions. IgY has a molecular weight of 180 KD and is composed of two subunits, a heavy-chain of 67 kDa and light-chain of 23 KD. The electrophoretic pattern of the IgY extracted from hyperimmunized egg yolks was similar to that of the IgY standard. See Figure 2.

30

SUBSTITUTE SHEET (RULE 26)"""

async def test_header_detection():
    """Test the new markdown-based header detection."""
    
    print("Testing markdown-based header detection...")
    print("=" * 60)
    
    # Create a test ProcessedPage with markdown content
    test_page = ProcessedPage(
        text="Raw OCR text with errors...",  # This would be the messy OCR text
        md=test_markdown,  # Clean markdown content
        filename="test_patent.pdf",
        page_number=31,
        section="detailed description",
        image=None
    )
    
    # Run header detection
    result = await detect_description_header_from_text(test_page)
    
    print(f"Page: {result.page_number}")
    print(f"Section: {result.section}")
    print(f"Has Header: {result.has_header}")
    print(f"Detected Header: '{result.header}'")
    print()
    
    # Test with multiple pages to show different headers
    test_pages = [
        # Page with "Detection of antibody..." header
        ProcessedPage(
            text="Raw text...",
            md="""Detection of antibody specificity and titer by ELISA

Antibody specificity and titer was determined by ELISA...""",
            filename="test_patent.pdf",
            page_number=31,
            section="detailed description"
        ),
        
        # Page with "Test for neutralization..." header  
        ProcessedPage(
            text="Raw text...",
            md="""Test for neutralization abilities of IgY

Enterococcus faecalis was cultured overnight...""",
            filename="test_patent.pdf", 
            page_number=32,
            section="detailed description"
        ),
        
        # Page with "Results" header
        ProcessedPage(
            text="Raw text...",
            md="""Results

Total soluble proteins including IgY...""",
            filename="test_patent.pdf",
            page_number=33, 
            section="detailed description"
        ),
        
        # Page with no clear header
        ProcessedPage(
            text="Raw text...",
            md="""WO 2022/221616                                                       PCT/US2022/024946

some continuing text from previous section without a clear header...""",
            filename="test_patent.pdf",
            page_number=34,
            section="detailed description"
        )
    ]
    
    print("Testing multiple pages:")
    print("-" * 40)
    
    for page in test_pages:
        result = await detect_description_header_from_text(page)
        print(f"Page {result.page_number}: {result.has_header} - '{result.header}'")
    
    print("\nComparison with your example:")
    print("-" * 40)
    print("Expected headers from your markdown:")
    print("✓ Detection of antibody specificity and titer by ELISA")
    print("✓ Test for neutralization abilities of IgY") 
    print("✓ Results")
    print("\nThese should be detected much more reliably than from OCR text!")

if __name__ == "__main__":
    asyncio.run(test_header_detection())
