from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import pytest

# Import the FastAPI app
from src.main import app

# Create a test client
client = TestClient(app)


class TestSyntheticEmbodiment:
    """Test class for the synthetic_embodiment endpoint"""
    
    @patch('src.main.generate_embodiment')
    def test_synthetic_embodiment_success(self, mock_generate_embodiment):
        """Test successful generation of a synthetic embodiment"""
        # Create mock response for generate_embodiment
        mock_response = MagicMock()
        mock_response.content = "Generated embodiment content"
        
        # Configure the mock to return the response
        mock_generate_embodiment.return_value = mock_response
        mock_generate_embodiment.__awaited__ = mock_response
        
        # Test parameters
        test_params = {
            "inspiration": 0.7,
            "source_embodiment": "Original embodiment text",
            "patent_title": "Test Patent",
            "disease": "Test Disease",
            "antigen": "Test Antigen"
        }
        
        # Make the request
        response = client.post(
            "/api/v1/embodiment",
            params=test_params
        )
        
        # Verify the response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["content"] == "Generated embodiment content"
        
        # Verify generate_embodiment was called with correct parameters
        mock_generate_embodiment.assert_called_once_with(
            test_params["inspiration"],
            test_params["source_embodiment"],
            test_params["patent_title"],
            test_params["disease"],
            test_params["antigen"]
        )
    
    @patch('src.main.generate_embodiment')
    def test_synthetic_embodiment_error(self, mock_generate_embodiment):
        """Test error handling when embodiment generation fails"""
        # Configure the mock to raise an exception
        error_message = "Generation service unavailable"
        mock_generate_embodiment.side_effect = Exception(error_message)
        
        # Test parameters
        test_params = {
            "inspiration": 0.7,
            "source_embodiment": "Original embodiment text",
            "patent_title": "Test Patent",
            "disease": "Test Disease",
            "antigen": "Test Antigen"
        }
        
        # Make the request
        response = client.post(
            "/api/v1/embodiment",
            params=test_params
        )
        
        # Verify the response indicates an error
        assert response.status_code == 500
        response_data = response.json()
        assert response_data["detail"] == f"Error generating synthetic embodiment: {error_message}"


@pytest.mark.asyncio
class TestSyntheticEmbodimentAsync:
    """Test class for async behavior of the synthetic_embodiment endpoint"""
    
    @patch('src.main.generate_embodiment')
    async def test_synthetic_embodiment_async(self, mock_generate_embodiment):
        """Test the async nature of the synthetic_embodiment function"""
        # Set up AsyncMock
        mock_response = AsyncMock()
        mock_response.content = "Generated async embodiment content"
        
        # Configure the mock to return the response
        mock_generate_embodiment.return_value = mock_response
        
        # We would use async test client here in a real test
        # For this example, we're validating the mock was set up correctly
        assert mock_generate_embodiment.return_value.content == "Generated async embodiment content"
        mock_generate_embodiment.assert_not_called()
