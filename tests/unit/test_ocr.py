import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Assuming the models are defined in src.utils.ocr
# Adjust the import path if necessary based on your project structure
from src.utils.ocr import (
    get_embodiments,
    ProcessedPage,
    Embodiments,
    Embodiment,
    categorize_embodiment,
    DetailedDescriptionEmbodiment
)


@pytest.fixture
def sample_processed_page():
    """Create a sample ProcessedPage object for testing."""
    return ProcessedPage(
        text="In certain aspects the disclosure relates to a method for preventing or treating a disease...",
        filename="test_patent.pdf",
        page_number=5,
        section="Detailed Description"
    )


@pytest.fixture
def mock_embodiments_response():
    """Create a mock response from the LLM API."""
    return Embodiments(
        content=[
            Embodiment(
                text="In certain aspects the disclosure relates to a method for preventing or treating a disease...",
                filename="test_patent.pdf",
                page_number=5,
                section="Detailed Description"
            )
        ]
    )


@pytest.mark.asyncio
async def test_get_embodiments_success(sample_processed_page, mock_embodiments_response):
    """Test that get_embodiments correctly processes a page with embodiments."""
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return our predefined response
        mock_create.return_value = mock_embodiments_response
        
        # Call the function
        result = await get_embodiments(sample_processed_page)
        
        # Assert the function called the API with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        
        # Verify model is correct
        assert call_args['model'] == 'o3-mini'
        
        # Verify context contains expected values
        assert call_args['context']['page_number'] == sample_processed_page.page_number
        assert call_args['context']['filename'] == sample_processed_page.filename
        assert call_args['context']['page_text'] == sample_processed_page.text
        assert isinstance(call_args['context']['example_embodiments'], str)
        
        # Verify the result is as expected (list of embodiments)
        assert result == mock_embodiments_response.content # Compare to the list inside
        assert len(result) == 1 # Check length of the list
        assert result[0].text == mock_embodiments_response.content[0].text # Access item in the list
        assert result[0].section == mock_embodiments_response.content[0].section # Access item in the list


@pytest.mark.asyncio
async def test_get_embodiments_empty_response(sample_processed_page):
    """Test that get_embodiments handles a page with no embodiments."""
    
    empty_response = Embodiments(content=[])
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return an empty response
        mock_create.return_value = empty_response
        
        # Call the function
        result = await get_embodiments(sample_processed_page)
        
        # Verify the result is as expected (empty list)
        assert result == empty_response.content # Compare to the empty list inside
        assert len(result) == 0 # Check length of the list


@pytest.mark.asyncio
async def test_get_embodiments_api_error(sample_processed_page):
    """Test that get_embodiments handles API errors properly."""
    
    # Mock the client.chat.completions.create call to raise an exception
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to raise an exception
        mock_create.side_effect = Exception("API Error")
        
        # Call the function and expect an exception
        with pytest.raises(Exception) as exc_info:
            await get_embodiments(sample_processed_page)
        
        # Verify the exception details
        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_embodiments_template_rendering(sample_processed_page, mock_embodiments_response):
    """Test that the templated message is correctly rendered with the provided context."""
    
    # Mock the client.chat.completions.create call
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Configure the mock to return our predefined response
        mock_create.return_value = mock_embodiments_response
        
        # Call the function
        await get_embodiments(sample_processed_page)
        
        # Check template variables are passed correctly
        call_args = mock_create.call_args[1]
        context = call_args['context']
        
        # Verify all required context variables are present
        assert 'page_number' in context
        assert 'filename' in context
        assert 'page_text' in context
        assert 'example_embodiments' in context
        
        # Ensure template variables match the page attributes
        assert context['page_number'] == sample_processed_page.page_number
        assert context['filename'] == sample_processed_page.filename
        assert context['page_text'] == sample_processed_page.text


@pytest.mark.asyncio
async def test_segment_pages_basic_classification():
    """Test that segment_pages correctly classifies pages based on keywords."""
    with patch('src.utils.ocr.client.chat.completions.create', 
               new_callable=AsyncMock) as mock_create:
        # Create pages with different content
        summary_page = ProcessedPage(
            text="SUMMARY OF THE INVENTION\nThis invention relates to...",
            filename="test.pdf",
            page_number=1,
            section=""
        )
        
        detailed_page = ProcessedPage(
            text="DETAILED DESCRIPTION\nIn one embodiment...",
            filename="test.pdf",
            page_number=2,
            section=""
        )
        
        claims_page = ProcessedPage(
            text="CLAIMS\n1. A method comprising...",
            filename="test.pdf",
            page_number=3,
            section=""
        )
        
        from src.utils.ocr import segment_pages
        pages = [summary_page, detailed_page, claims_page]
        
        # Run the function
        result = await segment_pages(pages)
        
        # Check sections assigned correctly based on keywords
        assert result[0].section == "Summary of Invention"
        assert result[1].section == "Detailed Description"
        assert result[2].section == "Claims"
        
        # Ensure the mock wasn't called (because keywords were used)
        mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_segment_pages_with_language_model():
    """Test that segment_pages uses language model when keywords aren't found."""
    from src.utils.ocr import segment_pages, client
    
    # Create two pages - one with summary to establish the section flow,
    # and a second ambiguous page where the model should be called
    summary_page = ProcessedPage(
        text="SUMMARY OF THE INVENTION",
        filename="test.pdf",
        page_number=1,
        section=""
    )
    
    ambiguous_page = ProcessedPage(
        text="This part describes various aspects of the invention without clear section headers...",
        filename="test.pdf",
        page_number=2,
        section=""
    )
            
    # Set up the mock for the instructor-enhanced client
    original_client = client.chat.completions.create
    
    async def mock_completion_create(*args, **kwargs):
        # Simulate the PatentSectionWithConfidence model structure
        mock_response = MagicMock()
        mock_response.section = "Detailed Description"
        mock_response.confidence = 0.9 # Add confidence attribute
        return mock_response
            
    try:
        client.chat.completions.create = mock_completion_create
        result = await segment_pages([summary_page, ambiguous_page])
                
        # First page should be Summary, second should be Detailed Description from our mock
        assert result[0].section == "Summary of Invention"
        assert result[1].section == "Detailed Description"
    finally:
        client.chat.completions.create = original_client


# --- Test Data ---
TEST_EMBODIMENT_INPUT = Embodiment(
    text="This embodiment describes the treatment of condition X using compound Y.",
    filename="test_patent.pdf",
    page_number=5,
    section="Detailed Description",
)

# --- Mocks ---
# Mock return value for disease rationale
MOCK_DISEASE_RATIONALE_OUTPUT = DetailedDescriptionEmbodiment(
    **TEST_EMBODIMENT_INPUT.model_dump(), category="disease rationale"
)

# Mock return value for product composition
MOCK_PRODUCT_COMPOSITION_OUTPUT = DetailedDescriptionEmbodiment(
    **TEST_EMBODIMENT_INPUT.model_dump(), category="product composition"
)


# --- Tests ---
@pytest.mark.asyncio
async def test_categorize_embodiment_disease_rationale(mocker):
    """
    Test that categorize_embodiment correctly identifies 'disease rationale'
    when the mocked LLM returns that category.
    """
    # Arrange: Patch the client's create method
    mock_create = mocker.patch(
        "src.utils.ocr.client.chat.completions.create",
        new_callable=AsyncMock,  # Use AsyncMock for async methods
        return_value=MOCK_DISEASE_RATIONALE_OUTPUT,
    )

    # Act: Call the function under test
    result = await categorize_embodiment(TEST_EMBODIMENT_INPUT)

    # Assert: Check the result and that the mock was called
    assert isinstance(result, DetailedDescriptionEmbodiment)
    assert result.category == "disease rationale"
    assert result.text == TEST_EMBODIMENT_INPUT.text # Ensure other fields are preserved
    mock_create.assert_awaited_once() # Verify the mock was called as expected


@pytest.mark.asyncio
async def test_categorize_embodiment_product_composition(mocker):
    """
    Test that categorize_embodiment correctly identifies 'product composition'
    when the mocked LLM returns that category.
    """
    # Arrange: Patch the client's create method
    mock_create = mocker.patch(
        "src.utils.ocr.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=MOCK_PRODUCT_COMPOSITION_OUTPUT,
    )

    # Act: Call the function under test
    result = await categorize_embodiment(TEST_EMBODIMENT_INPUT)

    # Assert: Check the result and that the mock was called
    assert isinstance(result, DetailedDescriptionEmbodiment)
    assert result.category == "product composition"
    assert result.filename == TEST_EMBODIMENT_INPUT.filename # Ensure other fields are preserved
    mock_create.assert_awaited_once()
