import pytest
import asyncio
from unittest import mock
from fastapi.testclient import TestClient
from src.main import app
from src.rag import Chunk


client = TestClient(app)


# Create an async mock that can properly be awaited
class AsyncMock(mock.MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Helper function for creating awaitable results
def get_awaitable(result):
    async def mock_coro(*args, **kwargs):
        return result
    return mock_coro


@pytest.fixture
def mock_db_connection():
    """Mock the database connection."""
    with mock.patch("src.main.db_connection") as mock_conn:
        mock_conn.__getitem__.return_value = {"db": mock.MagicMock()}
        yield mock_conn


@pytest.fixture
def mock_multiquery_functions():
    """Mock both multiquery_search and chunks_summary functions."""
    # Create sample chunks to return
    chunks = [
        Chunk(
            chunk_id="1",
            text="This is sample chunk 1 about antibodies.",
            page_number=1,
            filename="test.pdf",
        ),
        Chunk(
            chunk_id="2",
            text="This is sample chunk 2 about antigens.",
            page_number=2,
            filename="test.pdf",
        ),
    ]
    
    summary = "This is a sample summary of the retrieved chunks."
    
    # Create awaitable mocks
    search_mock = mock.MagicMock()
    search_mock.side_effect = get_awaitable(chunks)
    
    summary_mock = mock.MagicMock()
    summary_mock.side_effect = get_awaitable(summary)
    
    # Patch the functions
    with mock.patch("src.main.multiquery_search", search_mock) as mock_search, \
         mock.patch("src.main.chunks_summary", summary_mock) as mock_summary:
        yield mock_search, mock_summary


def test_multiquery_search_success(mock_db_connection, mock_multiquery_functions):
    """Test successful multiquery search endpoint call."""
    mock_search, mock_summary = mock_multiquery_functions
    
    # Make the request
    response = client.post(
        "/api/v1/rag/multiquery-search/",
        params={
            "query": "What are antibodies?",
            "target_files": ["document1.pdf", "document2.pdf"]
        }
    )
    
    # Print response for debugging
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "This is a sample summary of the retrieved chunks."
    
    # Verify mock calls
    assert mock_search.called
    assert mock_summary.called
    
    # Verify arguments to multiquery_search
    call_args = mock_search.call_args
    assert call_args[0][0] == "What are antibodies?"  # First arg is query
    assert call_args[0][1] == ["document1", "document2"]  # Second arg is table names


def test_multiquery_search_validation_errors(mock_db_connection):
    """Test parameter validation in the endpoint."""
    # Test missing query parameter
    response1 = client.post(
        "/api/v1/rag/multiquery-search/",
        params={
            "target_files": ["document1.pdf"]
        }
    )
    assert response1.status_code == 422
    assert "detail" in response1.json()
    
    # Test missing target_files parameter
    response2 = client.post(
        "/api/v1/rag/multiquery-search/",
        params={
            "query": "What are antibodies?"
        }
    )
    assert response2.status_code == 422
    assert "detail" in response2.json()


def test_multiquery_search_exception(mock_db_connection):
    """Test error handling in the endpoint."""
    # Mock multiquery_search to raise an exception
    error_mock = mock.MagicMock(side_effect=Exception("Test error"))
    
    with mock.patch("src.main.multiquery_search", error_mock):
        # Make request
        response = client.post(
            "/api/v1/rag/multiquery-search/",
            params={
                "query": "What are antibodies?",
                "target_files": ["document1.pdf"]
            }
        )
        
        # Print response for debugging
        print(f"Error Response: {response.json()}")
        
        # Verify error response
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Error during multiquery-search" in response.json()["detail"]
