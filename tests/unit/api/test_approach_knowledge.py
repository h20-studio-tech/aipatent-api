import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from src.main import app
from src.models.api_schemas import ApproachKnowledge
import pytest
from unittest.mock import patch

client = TestClient(app)

@pytest.fixture
def valid_approach_knowledge():
    return {
        "patent_id": str(uuid.uuid4()),
        "question": "What is the approach for this invention?",
        "answer": "The approach is to use Eggerthalla lenta to modulate immune activation.",
        "created_at": datetime.utcnow().isoformat()
    }

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_approach_knowledge_success(mock_getenv, mock_dynamodb, valid_approach_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.return_value = {"Attributes": {"knowledge": [valid_approach_knowledge]}}

    response = client.post("/api/v1/knowledge/approach/", json=valid_approach_knowledge)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "data" in response.json()

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_approach_knowledge_dynamodb_error(mock_getenv, mock_dynamodb, valid_approach_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.side_effect = Exception("DynamoDB failure")

    response = client.post("/api/v1/knowledge/approach/", json=valid_approach_knowledge)
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert "DynamoDB failure" in response.json()["message"]

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_approach_knowledge_invalid_uuid(mock_getenv, mock_dynamodb, valid_approach_knowledge):
    valid_approach_knowledge["patent_id"] = "not-a-uuid"
    response = client.post("/api/v1/knowledge/approach/", json=valid_approach_knowledge)
    assert response.status_code == 422
    assert ("uuid" in response.text or "patent_id" in response.text)
