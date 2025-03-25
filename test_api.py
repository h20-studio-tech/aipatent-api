import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from botocore.exceptions import ClientError
from main import app  # Import your FastAPI app

client = TestClient(app)

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = Mock()
        mock_table.put_item.return_value = {
            'ResponseMetadata': {'HTTPStatusCode': 200}
        }
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

def test_create_patent_success(mock_dynamodb):
    test_data = {
        "name": "Test Patent",
        "description": "Test Description",
        "inventors": ["John Doe"],
        "status": "PENDING"
    }
    
    response = client.post("/api/v1/project/", json=test_data)
    
    assert response.status_code == 200
    assert "patent_id" in response.json()
    assert response.json()["message"] == "Patent project created successfully"

def test_create_patent_db_error(mock_dynamodb):
    mock_dynamodb.put_item.side_effect = ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "Test error"}},
        operation_name="PutItem"
    )
    
    test_data = {
        "name": "Test Patent",
        "description": "Test Description",
        "inventors": ["John Doe"],
        "status": "PENDING"
    }
    
    response = client.post("/api/v1/project/", json=test_data)
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

