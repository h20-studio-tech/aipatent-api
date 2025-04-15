import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from src.utils.ocr import (
    Embodiment,
    DetailedDescriptionEmbodiment,
    categorize_embodiment,
    categorize_detailed_description
)
from src.main import patent
from fastapi import UploadFile
from fastapi.exceptions import HTTPException


@pytest.fixture
def sample_embodiments():
    """Create a sample list of embodiments with different sections."""
    return [
        # Detailed description embodiments
        Embodiment(
            text="This embodiment describes the treatment of condition X using compound Y.",
            filename="test_patent.pdf",
            page_number=5,
            section="detailed_description",
        ),
        Embodiment(
            text="This invention provides a composition comprising polymer Z.",
            filename="test_patent.pdf",
            page_number=6,
            section="detailed_description",
        ),
        # Summary embodiment
        Embodiment(
            text="The invention relates to novel treatments for disease X.",
            filename="test_patent.pdf",
            page_number=2,
            section="Summary of Invention",
        ),
        # Claims embodiment
        Embodiment(
            text="1. A method of treating disease X comprising administering compound Y.",
            filename="test_patent.pdf",
            page_number=10,
            section="Claims",
        ),
    ]


@pytest.fixture
def detailed_description_embodiments(sample_embodiments):
    """Filter only detailed_description embodiments from sample_embodiments."""
    return [embodiment for embodiment in sample_embodiments if embodiment.section == "detailed_description"]


@pytest.fixture
def categorized_embodiments():
    """Create sample categorized detailed description embodiments."""
    return [
        DetailedDescriptionEmbodiment(
            text="This embodiment describes the treatment of condition X using compound Y.",
            filename="test_patent.pdf",
            page_number=5,
            section="detailed_description",
            sub_category="disease rationale",
        ),
        DetailedDescriptionEmbodiment(
            text="This invention provides a composition comprising polymer Z.",
            filename="test_patent.pdf",
            page_number=6,
            section="detailed_description",
            sub_category="product composition",
        ),
    ]


@pytest.mark.asyncio
async def test_categorize_detailed_description(detailed_description_embodiments, categorized_embodiments):
    """Test that categorize_detailed_description correctly categorizes all embodiments."""
    # Patch the categorize_embodiment function to return predefined categorized embodiments
    with patch('src.utils.ocr.categorize_embodiment', new_callable=AsyncMock) as mock_categorize:
        # Set up the mock to return lists containing our predefined categorized embodiments
        # The function expects categorize_embodiment to return a list for each input
        mock_categorize.side_effect = [[embodiment] for embodiment in categorized_embodiments]
        
        # Call the function under test
        result = await categorize_detailed_description(detailed_description_embodiments)
        
        # Verify mock was called with each detailed description embodiment
        assert mock_categorize.call_count == len(detailed_description_embodiments)
        for i, embodiment in enumerate(detailed_description_embodiments):
            mock_categorize.assert_any_call(embodiment)
        
        # Verify results match expected categorized embodiments
        assert len(result) == len(categorized_embodiments)
        for i, embodiment in enumerate(result):
            assert isinstance(embodiment, DetailedDescriptionEmbodiment)
            assert embodiment.sub_category in ["disease rationale", "product composition"]
            assert embodiment.text == categorized_embodiments[i].text
            assert embodiment.sub_category == categorized_embodiments[i].sub_category


@pytest.mark.asyncio
async def test_process_patent_document_integration(sample_embodiments, categorized_embodiments):
    """Test the integration of process_patent_document with categorize_detailed_description."""
    # Create a mock for the UploadFile
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_patent.pdf"
    mock_file.read = AsyncMock(return_value=b"mock pdf content")
    
    # Patch process_patent_document to return our sample embodiments
    with patch('src.main.process_patent_document', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = sample_embodiments
        
        # Patch categorize_detailed_description to return our categorized embodiments
        with patch('src.main.categorize_detailed_description', new_callable=AsyncMock) as mock_categorize:
            mock_categorize.return_value = categorized_embodiments
            
            # Call the patent endpoint
            response = await patent(mock_file)
            
            # Verify process_patent_document was called with correct parameters
            mock_process.assert_called_once()
            assert mock_process.call_args[0][0] == b"mock pdf content"
            assert mock_process.call_args[0][1] == "test_patent.pdf"
            
            # Verify categorize_detailed_description was called with detailed_description embodiments
            mock_categorize.assert_called_once()
            expected_embodiments = [e for e in sample_embodiments if e.section == "detailed_description"]
            actual_embodiments = mock_categorize.call_args[0][0]
            assert len(actual_embodiments) == len(expected_embodiments)
            for i, embodiment in enumerate(actual_embodiments):
                assert embodiment.section == "detailed_description"
                assert embodiment.text == expected_embodiments[i].text
            
            # Verify the response contains all embodiments (non-detailed + categorized detailed)
            expected_count = len([e for e in sample_embodiments if e.section != "detailed_description"]) + len(categorized_embodiments)
            assert len(response.data) == expected_count
            
            # Check that detailed description embodiments have categories
            detailed_in_response = [e for e in response.data if e.section == "detailed_description"]
            assert len(detailed_in_response) == len(categorized_embodiments)
            for embodiment in detailed_in_response:
                assert hasattr(embodiment, 'category')
                assert embodiment.category in ["disease rationale", "product composition"]


@pytest.mark.asyncio
async def test_error_handling_in_patent_endpoint():
    """Test that the patent endpoint properly handles and reports errors."""
    # Create a mock for the UploadFile
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_patent.pdf"
    mock_file.read = AsyncMock(return_value=b"mock pdf content")
    
    # Patch process_patent_document to raise an exception
    error_message = "Test error in patent processing"
    with patch('src.main.process_patent_document', new_callable=AsyncMock) as mock_process:
        mock_process.side_effect = Exception(error_message)
        
        # Call the patent endpoint and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await patent(mock_file)
        
        # Verify the error message is included in the HTTPException
        assert excinfo.value.status_code == 500
        assert error_message in excinfo.value.detail


@pytest.mark.asyncio
async def test_categorize_empty_detailed_description():
    """Test handling of empty list of detailed description embodiments."""
    # Call with empty list
    result = await categorize_detailed_description([])
    
    # Verify empty list is returned
    assert isinstance(result, list)
    assert len(result) == 0
