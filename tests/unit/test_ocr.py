import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from src.utils.ocr import (
    get_embodiments, 
    EmbodimentsPage,
    Embodiments,
    Embodiment
)


@pytest.fixture
def sample_embodiments_page():
    """Create a sample EmbodimentsPage object for testing."""
    return EmbodimentsPage(
        text="In certain aspects the disclosure relates to a method for preventing or treating a disease...",
        filename="test_patent.pdf",
        page=5
    )


@pytest.fixture
def mock_embodiments_response():
    """Create a mock response from the LLM API."""
    return Embodiments(
        content=[
            Embodiment(
                text="In certain aspects the disclosure relates to a method for preventing or treating a disease...",
                filename="test_patent.pdf",
                page=5
            )
        ]
    )


@pytest.mark.asyncio
async def test_get_embodiments_success(sample_embodiments_page, mock_embodiments_response):
    """Test that get_embodiments correctly processes a page with embodiments."""
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return our predefined response
        mock_create.return_value = mock_embodiments_response
        
        # Call the function
        result = await get_embodiments(sample_embodiments_page)
        
        # Assert the function called the API with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        
        # Verify model is correct
        assert call_args['model'] == 'o3-mini'
        
        # Verify context contains expected values
        assert call_args['context']['page_number'] == sample_embodiments_page.page
        assert call_args['context']['filename'] == sample_embodiments_page.filename
        assert call_args['context']['page_text'] == sample_embodiments_page.text
        assert isinstance(call_args['context']['example_embodiments'], str)
        
        # Verify the result is as expected
        assert result == mock_embodiments_response
        assert len(result.content) == 1
        assert result.content[0].text == mock_embodiments_response.content[0].text


@pytest.mark.asyncio
async def test_get_embodiments_empty_response(sample_embodiments_page):
    """Test that get_embodiments handles a page with no embodiments."""
    
    empty_response = Embodiments(content=[])
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return an empty response
        mock_create.return_value = empty_response
        
        # Call the function
        result = await get_embodiments(sample_embodiments_page)
        
        # Verify the result is as expected
        assert result == empty_response
        assert len(result.content) == 0


@pytest.mark.asyncio
async def test_get_embodiments_api_error(sample_embodiments_page):
    """Test that get_embodiments handles API errors properly."""
    
    # Mock the client.chat.completions.create call to raise an exception
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to raise an exception
        mock_create.side_effect = Exception("API Error")
        
        # Call the function and expect an exception
        with pytest.raises(Exception) as exc_info:
            await get_embodiments(sample_embodiments_page)
        
        # Verify the exception details
        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_embodiments_template_rendering(sample_embodiments_page, mock_embodiments_response):
    """Test that the templated message is correctly rendered with the provided context."""
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return our predefined response
        mock_create.return_value = mock_embodiments_response
        
        # Call the function
        await get_embodiments(sample_embodiments_page)
        
        # Check template variables are passed correctly
        call_args = mock_create.call_args[1]
        context = call_args['context']
        
        # Verify all required context variables are present
        assert 'page_number' in context
        assert 'filename' in context
        assert 'page_text' in context
        assert 'example_embodiments' in context
        
        # Ensure template variables match the page attributes
        assert context['page_number'] == sample_embodiments_page.page
        assert context['filename'] == sample_embodiments_page.filename
        assert context['page_text'] == sample_embodiments_page.text
