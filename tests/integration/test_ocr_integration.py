import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from src.utils.ocr import (
    process_pdf_pages,
    segment_pages,
    get_embodiments,
    find_embodiments,
    process_patent_document,
    pdf_pages,
    ProcessedPage,
    Embodiments
)


@pytest.fixture
def sample_patent_path():
    """Return the path to a sample patent PDF file."""
    sample_dir = Path("experiments/sample_patents")
    # Using the smallest file for faster tests
    sample_file = next(sample_dir.glob("*.pdf"))
    return str(sample_file)


@pytest.fixture
def mock_instructor_client():
    """Mock the instructor client for tests."""
    with patch('src.utils.ocr.client.chat.completions.create', new_callable=AsyncMock) as mock_client:
        # Create a generic response for section detection
        section_response = MagicMock()
        section_response.section = "Detailed Description"
        
        # Create a generic response for embodiment extraction
        embodiments_response = Embodiments(content=[])
        
        # Configure the mock to return the appropriate response based on context
        async def side_effect(*args, **kwargs):
            if 'response_model' in kwargs and hasattr(kwargs['response_model'], '__name__'):
                if kwargs['response_model'].__name__ == 'PatentSection':
                    return section_response
                elif kwargs['response_model'].__name__ == 'Embodiments':
                    return embodiments_response
            return MagicMock()
        
        mock_client.side_effect = side_effect
        yield mock_client


@pytest.mark.asyncio
async def test_process_pdf_pages_with_real_file(sample_patent_path):
    """Test that PDF pages can be processed from a real file."""
    with open(sample_patent_path, 'rb') as f:
        pdf_data = f.read()
    
    raw_pages = pdf_pages(pdf_data, os.path.basename(sample_patent_path))
    processed_pages = process_pdf_pages(raw_pages)
    
    # Basic validation that pages were extracted
    assert len(processed_pages) > 0
    
    # Check that pages have the expected structure
    for page in processed_pages:
        assert isinstance(page, ProcessedPage)
        assert page.text != ""
        assert page.filename == os.path.basename(sample_patent_path)
        assert isinstance(page.page, int)
        assert page.section == ""  # Initially empty


@pytest.mark.asyncio
async def test_segment_pages_with_real_content(mock_instructor_client):
    """Test page segmentation with realistic content."""
    # Create sample pages with realistic content
    pages = [
        ProcessedPage(
            text="FIELD OF THE INVENTION\nThis invention relates to biomedical applications...",
            filename="test.pdf",
            page_number=1,
            section=""
        ),
        ProcessedPage(
            text="The present invention provides methods for detecting and preventing diseases...",
            filename="test.pdf",
            page_number=2,
            section=""
        ),
        ProcessedPage(
            text="DETAILED DESCRIPTION\nIn one embodiment, the composition comprises...",
            filename="test.pdf",
            page_number=3,
            section=""
        )
    ]
    
    # Process the pages
    result = await segment_pages(pages)
    
    # Validate results
    assert len(result) == 3
    assert result[0].section == "summary of invention"  # First page default
    assert result[2].section == "detailed description"  # Keyword match
    
    # The middle page should have triggered the language model
    assert mock_instructor_client.called


@pytest.mark.asyncio
async def test_get_embodiments_with_realistic_content(mock_instructor_client):
    """Test embodiment extraction with realistic content."""
    page = ProcessedPage(
        text="""In certain aspects the disclosure relates to a method for preventing or treating alcoholic liver 
        disease in a subject in need thereof, comprising administering to the subject a therapeutically 
        effective amount of a hyperimmunized egg product obtained from an egg‑producing animal, thereby 
        preventing or treating the alcoholic liver disease in the subject, wherein the hyperimmunized egg 
        product comprises a therapeutically effective amount of one or more antibodies to an antigen selected 
        from the group consisting of Enterococcus faecalis and Enterococcus faecalis cytolysin toxin.""",
        filename="test.pdf",
        page_number=5,
        section="detailed description"
    )
    
    await get_embodiments(page)
    
    # Verify the API was called with the correct inputs
    assert mock_instructor_client.called
    call_args = mock_instructor_client.call_args[1]
    
    # Check for the embodiment content in the request
    assert "alcoholic liver disease" in call_args['context']['page_text']
    assert call_args['context']['page_number'] == 5
    assert call_args['context']['filename'] == "test.pdf"


@pytest.mark.asyncio
async def test_full_pipeline_with_real_file(sample_patent_path, mock_instructor_client):
    """Test the entire document processing pipeline with a real file."""
    with open(sample_patent_path, 'rb') as f:
        pdf_data = f.read()
    
    # Create a mock embodiment to be returned
    mock_embodiment = {
        "text": "A method for preventing disease comprising administering an egg product...",
        "filename": os.path.basename(sample_patent_path),
        "page_number": 3,
        "section": "detailed description"
    }
    
    # Configure mock to return an embodiment
    embodiments_response = Embodiments(content=[mock_embodiment])
    
    def new_side_effect(*args, **kwargs):
        if 'response_model' in kwargs and kwargs['response_model'].__name__ == 'Embodiments':
            return embodiments_response
        section_response = MagicMock()
        section_response.section = "detailed description"
        return section_response
    
    mock_instructor_client.side_effect = AsyncMock(side_effect=new_side_effect)
    
    # Run the full pipeline
    results = await process_patent_document(pdf_data, os.path.basename(sample_patent_path))
    
    # Validate that we got results
    assert len(results) > 0
    
    # Verify the mocked embodiment is in the results
    assert any(e.text == mock_embodiment["text"] for e in results)


# Add more comprehensive error tests
@pytest.mark.asyncio
async def test_segment_pages_handles_empty_pages():
    """Test that segmentation properly handles empty page lists."""
    result = await segment_pages([])
    assert result == []


@pytest.mark.asyncio
async def test_segment_pages_preserves_page_order(mock_instructor_client):
    """Test that segmentation preserves page order even with unsorted input."""
    # Create pages in reverse order
    pages = [
        ProcessedPage(text="Page 3", filename="test.pdf", page=3, section=""),
        ProcessedPage(text="Page 2", filename="test.pdf", page=2, section=""),
        ProcessedPage(text="Page 1", filename="test.pdf", page=1, section="")
    ]
    
    result = await segment_pages(pages)
    
    # Pages should be sorted by page number
    assert result[0].page == 1
    assert result[1].page == 2
    assert result[2].page == 3
