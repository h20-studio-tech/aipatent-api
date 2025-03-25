import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from botocore.exceptions import ClientError
import src.main as main

client = TestClient(main.app)

@pytest.fixture
def mock_dynamodb(monkeypatch):
    mock_table = Mock()
    # For the success case, set put_item to return a successful response.
    mock_table.put_item.return_value = {
        'ResponseMetadata': {'HTTPStatusCode': 200}
    }
    mock_ddb = Mock()
    # When the endpoint calls dynamodb.Table('patents'),
    # it will receive our mock_table.
    mock_ddb.Table.return_value = mock_table
    # Override the dynamodb in main with our mock.
    monkeypatch.setattr(main, "dynamodb", mock_ddb)
    return mock_table

def test_create_patent_success(mock_dynamodb):
    # Updated test_data to match PatentProject model
    test_data = {
        "name": "Test Patent",
        "antigen": "Test Antigen",
        "disease": "Test Disease"
    }
    
    response = client.post("/api/v1/project/", json=test_data)
    
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    
    assert response.status_code == 200
    assert "patent_id" in response.json()
    assert response.json()["message"] == "Patent project created successfully"

def test_create_patent_db_error(mock_dynamodb):
    # Simulate a ClientError when put_item is called.
    mock_dynamodb.put_item.side_effect = ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "Test error"}},
        operation_name="PutItem"
    )

    test_data = {
        "name": "Test Patent",
        "antigen": "Test Antigen",
        "disease": "Test Disease"
    }

    response = client.post("/api/v1/project/", json=test_data)
    print(f"Error Response: {response.json()}")

    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_create_patent_validation_error():
    # Missing required fields to trigger 422
    invalid_test_data = {
        "name": "Test Patent"
    }
    
    response = client.post("/api/v1/project/", json=invalid_test_data)
    print(f"Validation Error Response: {response.json()}")
    
    assert response.status_code == 422
    # Updated assertion to match the actual error message
    assert "field required" in response.json()["detail"][0]["msg"].lower()
