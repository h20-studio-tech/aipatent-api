import pytest
import asyncio
from src.utils.ocr import get_embodiments
from src.models.ocr_schemas import ProcessedPage, Embodiment


class TestEmbodimentPositions:
    """Test cases for embodiment character position extraction."""
    
    @pytest.mark.asyncio
    async def test_simple_embodiment_position(self):
        """Test basic embodiment position extraction."""
        page_text = """
Background of the Invention

The present invention relates to immunotherapy.

In certain aspects the disclosure relates to a method for preventing or treating alcoholic liver disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product.

Additional text follows here.
"""
        
        page = ProcessedPage(
            text=page_text,
            filename="test_patent.pdf",
            page_number=1,
            section="detailed description"
        )
        
        embodiments = await get_embodiments(page)
        
        assert len(embodiments) > 0, "Should extract at least one embodiment"
        
        # Verify first embodiment
        emb = embodiments[0]
        assert hasattr(emb, 'start_char'), "Embodiment should have start_char"
        assert hasattr(emb, 'end_char'), "Embodiment should have end_char"
        
        # Validate position accuracy
        if emb.start_char and emb.end_char:
            start = int(emb.start_char)
            end = int(emb.end_char)
            extracted = page_text[start:end]
            assert extracted.strip() == emb.text.strip(), f"Position extraction mismatch: got '{extracted}'"
    
    @pytest.mark.asyncio
    async def test_multiple_embodiments_positions(self):
        """Test position extraction for multiple embodiments on same page."""
        page_text = """
Detailed Description

In certain embodiments, the composition comprises at least 0.01% of the hyperimmunized egg product.

The method may further include monitoring the subject for improvement.

In certain embodiments, the hyperimmunized egg product is administered orally.

In another aspect, the invention provides a pharmaceutical composition.
"""
        
        page = ProcessedPage(
            text=page_text,
            filename="test_patent.pdf",
            page_number=2,
            section="detailed description"
        )
        
        embodiments = await get_embodiments(page)
        
        assert len(embodiments) >= 2, "Should extract multiple embodiments"
        
        # Check all embodiments have valid positions
        for i, emb in enumerate(embodiments):
            if emb.start_char and emb.end_char:
                start = int(emb.start_char)
                end = int(emb.end_char)
                extracted = page_text[start:end]
                assert extracted.strip() == emb.text.strip(), f"Embodiment {i} position mismatch"
    
    @pytest.mark.asyncio
    async def test_embodiment_with_special_characters(self):
        """Test position extraction with special characters and formatting."""
        page_text = """
The invention includes:

    • In certain aspects, the disclosure relates to a method for treating graft-versus-host disease (GvHD).
    
    • The method comprises administering 10-50 mg/kg of the compound.

Additional embodiments follow.
"""
        
        page = ProcessedPage(
            text=page_text,
            filename="test_patent.pdf",
            page_number=3,
            section="summary of invention"
        )
        
        embodiments = await get_embodiments(page)
        
        # Verify special characters are handled correctly
        for emb in embodiments:
            if emb.start_char and emb.end_char:
                start = int(emb.start_char)
                end = int(emb.end_char)
                extracted = page_text[start:end]
                # Allow for some whitespace normalization
                assert extracted.strip() == emb.text.strip(), "Special character handling failed"
    
    @pytest.mark.asyncio
    async def test_edge_case_positions(self):
        """Test edge cases like embodiments at start/end of page."""
        # Embodiment at the very start
        page_text = """In certain aspects the disclosure relates to a novel treatment. This is followed by more text."""
        
        page = ProcessedPage(
            text=page_text,
            filename="test_patent.pdf",
            page_number=4,
            section="detailed description"
        )
        
        embodiments = await get_embodiments(page)
        
        if embodiments and embodiments[0].start_char is not None:
            assert int(embodiments[0].start_char) == 0, "First embodiment should start at position 0"
    
    def test_position_validation_helper(self):
        """Test the position validation logic."""
        page_text = "Some text. In certain aspects the method includes treatment. More text."
        
        # Simulate an embodiment with positions
        class MockEmbodiment:
            text = "In certain aspects the method includes treatment."
            start_char = "11"  # Position of 'I'
            end_char = "59"   # Position after '.'
        
        emb = MockEmbodiment()
        start = int(emb.start_char)
        end = int(emb.end_char)
        extracted = page_text[start:end]
        
        assert extracted == emb.text, f"Manual position calculation failed: '{extracted}' != '{emb.text}'"


if __name__ == "__main__":
    # Run a simple test
    async def main():
        test = TestEmbodimentPositions()
        await test.test_simple_embodiment_position()
        print("✓ Basic test passed")
    
    asyncio.run(main())