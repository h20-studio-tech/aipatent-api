import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from src.main import app
from src.models.api_schemas import InnovationKnowledge, TechnologyKnowledge
import pytest
from unittest.mock import patch

client = TestClient(app)

@pytest.fixture
def valid_innovation_knowledge():
    return {
        "patent_id": str(uuid.uuid4()),
        "question": "What is the key innovation?",
        "answer": "A novel method for microbiome modulation.",
        "created_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def valid_technology_knowledge():
    return {
        "patent_id": str(uuid.uuid4()),
        "question": "What is the technology platform?",
        "answer": "A CRISPR-based gene editing system.",
        "created_at": datetime.utcnow().isoformat()
    }

# INNOVATION ENDPOINT TESTS
@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_innovation_knowledge_success(mock_getenv, mock_dynamodb, valid_innovation_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.return_value = {"Attributes": {"knowledge": [valid_innovation_knowledge]}}

    response = client.post("/api/v1/knowledge/innovation/", json=valid_innovation_knowledge)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "data" in response.json()

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_innovation_knowledge_dynamodb_error(mock_getenv, mock_dynamodb, valid_innovation_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.side_effect = Exception("DynamoDB failure")

    response = client.post("/api/v1/knowledge/innovation/", json=valid_innovation_knowledge)
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert "DynamoDB failure" in response.json()["message"]

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_innovation_knowledge_invalid_uuid(mock_getenv, mock_dynamodb, valid_innovation_knowledge):
    valid_innovation_knowledge["patent_id"] = "not-a-uuid"
    response = client.post("/api/v1/knowledge/innovation/", json=valid_innovation_knowledge)
    assert response.status_code == 422
    assert ("uuid" in response.text or "patent_id" in response.text)

# TECHNOLOGY ENDPOINT TESTS
@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_technology_knowledge_success(mock_getenv, mock_dynamodb, valid_technology_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.return_value = {"Attributes": {"knowledge": [valid_technology_knowledge]}}

    response = client.post("/api/v1/knowledge/technology/", json=valid_technology_knowledge)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "data" in response.json()

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_technology_knowledge_dynamodb_error(mock_getenv, mock_dynamodb, valid_technology_knowledge):
    table_mock = mock_dynamodb.Table.return_value
    table_mock.update_item.side_effect = Exception("DynamoDB failure")

    response = client.post("/api/v1/knowledge/technology/", json=valid_technology_knowledge)
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert "DynamoDB failure" in response.json()["message"]

@patch("src.main.dynamodb")
@patch("src.main.os.getenv", return_value="patents")
def test_store_technology_knowledge_invalid_uuid(mock_getenv, mock_dynamodb, valid_technology_knowledge):
    valid_technology_knowledge["patent_id"] = "not-a-uuid"
    response = client.post("/api/v1/knowledge/technology/", json=valid_technology_knowledge)
    assert response.status_code == 422
    assert ("uuid" in response.text or "patent_id" in response.text)
