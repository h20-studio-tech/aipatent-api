from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app and the ApprovedEmbodiment model
from src.main import app

# Create a test client
client = TestClient(app)


class TestEmbodimentApprove:
    """Test class for the embodiment_approve endpoint"""
    
    @patch('src.main.dynamodb')
    def test_embodiment_approve_success(self, mock_dynamodb):
        """Test successful addition of an embodiment"""
        # Mock DynamoDB table and its update_item method
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        
        # Mock the response from DynamoDB
        mock_table.update_item.return_value = {
            "Attributes": {
                "embodiments": [
                    {"type": "test_type", "content": {"key": "value"}}
                ]
            }
        }
        
        # Test data
        patent_id = "test-patent-123"
        embodiment_data = {
            "type": "test_type", 
            "content": {"key": "value"}
        }
        
        # Make the request
        response = client.post(
            f"/api/v1/embodiment/approve?patent_id={patent_id}",
            json=embodiment_data
        )
        
        # Verify the response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Embodiment added to patent"
        assert "data" in response_data
        
        # Verify DynamoDB Table was called with correct parameters
        mock_dynamodb.Table.assert_called_once()
        
        # Verify update_item was called with the right parameters
        mock_table.update_item.assert_called_once()
        # Extract the call arguments for verification
        call_args = mock_table.update_item.call_args[1]
        assert call_args["Key"]["patent_id"] == patent_id
        assert ":embodiment" in call_args["ExpressionAttributeValues"]
        assert ":empty_list" in call_args["ExpressionAttributeValues"]
        assert call_args["ReturnValues"] == "UPDATED_NEW"
    
    @patch('src.main.dynamodb')
    def test_embodiment_approve_error(self, mock_dynamodb):
        """Test error handling when DynamoDB operation fails"""
        # Mock DynamoDB table and make it raise an exception
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        
        # Set up the mock to raise an exception
        error_message = "Database connection failed"
        mock_table.update_item.side_effect = Exception(error_message)
        
        # Test data
        patent_id = "test-patent-123"
        embodiment_data = {
            "type": "test_type", 
            "content": {"key": "value"}
        }
        
        # Make the request
        response = client.post(
            f"/api/v1/embodiment/approve?patent_id={patent_id}",
            json=embodiment_data
        )
        
        # Verify the response
        assert response.status_code == 200  # Note: The endpoint returns 200 even on error
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["message"] == error_message
    
    def test_embodiment_approve_validation(self):
        """Test validation error for invalid embodiment data"""
        # Test data with missing required fields
        patent_id = "test-patent-123"
        invalid_data = {"content": {"key": "value"}}  # Missing 'type' field
        
        # Make the request
        response = client.post(
            f"/api/v1/embodiment/approve?patent_id={patent_id}",
            json=invalid_data
        )
        
        # Verify the response indicates a validation error
        assert response.status_code == 422  # Unprocessable Entity for validation errors
        response_data = response.json()
        assert "detail" in response_data  # Validation errors include a detail field
