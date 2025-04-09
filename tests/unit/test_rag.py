import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.rag import multiquery_search, search
from models.rag_typing import Chunk
from lancedb.db import AsyncConnection
from src.rag import MultiQueryQuestions

@pytest.fixture
def mock_db():
    """Create a mock LanceDB connection."""
    mock_db = AsyncMock(spec=AsyncConnection)
    return mock_db

@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    with patch('src.rag.openai') as mock:
        mock_response = MagicMock()
        mock_response.questions = [
            "What are the key findings about antigen binding?",
            "How does the antibody interact with the target?",
            "What are the experimental results?"
        ]
        mock.chat.completions.create.return_value = mock_response
        yield mock

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings client."""
    with patch('src.rag.client.embeddings') as mock:
        mock.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        yield mock

@pytest.fixture
def mock_table():
    """Create a mock LanceDB table."""
    mock_table = AsyncMock()
    mock_search_result = [
        {
            "chunk_id": 1,
            "text": "Sample text about antigen binding",
            "page_number": 1,
            "filename": "test.pdf"
        }
    ]
    # Create a mock chain for vector_search().limit().to_list()
    mock_to_list = AsyncMock(return_value=mock_search_result)
    mock_limit = AsyncMock()
    mock_limit.to_list = mock_to_list
    mock_vector_search = AsyncMock()
    mock_vector_search.limit = MagicMock(return_value=mock_limit)
    mock_table.vector_search = MagicMock(return_value=mock_vector_search)
    return mock_table

@pytest.mark.asyncio
async def test_multiquery_search_success(mock_db, mock_openai, mock_embeddings, mock_table):
    """Test successful multiquery search."""
    # Setup
    query = "What are the key findings about antigen binding?"
    table_names = ["test_table"]
    mock_db.open_table.return_value = mock_table

    # Execute
    result = await multiquery_search(query, table_names=table_names, db=mock_db)

    # Assert
    assert isinstance(result, list)
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert len(result) > 0
    assert result[0].chunk_id == 1
    assert result[0].text == "Sample text about antigen binding"
    assert result[0].page_number == 1
    assert result[0].filename == "test.pdf"

    # Verify OpenAI was called
    mock_openai.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_multiquery_search_empty_tables(mock_db, mock_openai, mock_embeddings, mock_table):
    """Test multiquery search with empty table names."""
    # Setup
    query = "What are the key findings?"
    table_names = []

    # Execute
    result = await multiquery_search(query, table_names=table_names, db=mock_db)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_multiquery_search_error_handling(mock_db, mock_openai, mock_embeddings):
    """Test error handling in multiquery search."""
    # Setup
    query = "What are the key findings?"
    table_names = ["test_table"]
    mock_db.open_table.side_effect = Exception("Database error")

    # Execute
    result = await multiquery_search(query, table_names=table_names, db=mock_db)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_multiquery_search_multiple_tables(mock_db, mock_openai, mock_embeddings, mock_table):
    """Test multiquery search with multiple tables."""
    # Setup
    query = "What are the key findings?"
    table_names = ["table1", "table2"]
    mock_db.open_table.return_value = mock_table

    # Execute
    result = await multiquery_search(query, table_names=table_names, db=mock_db)

    # Assert
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(chunk, Chunk) for chunk in result)
    
    # Verify open_table is called for each combination of query and table
    expected_calls = len(mock_openai.chat.completions.create.return_value.questions) * len(table_names)
    assert mock_db.open_table.call_count == expected_calls

@pytest.mark.asyncio
async def test_search_function(mock_db, mock_embeddings, mock_table):
    """Test the search function independently."""
    # Setup
    query = "Test query"
    table_name = "test_table"
    mock_db.open_table.return_value = mock_table

    # Execute
    result = await search(query, table_name=table_name, db=mock_db)

    # Assert
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["chunk_id"] == 1
    assert result[0]["text"] == "Sample text about antigen binding"
    assert result[0]["page_number"] == 1
    assert result[0]["filename"] == "test.pdf" 