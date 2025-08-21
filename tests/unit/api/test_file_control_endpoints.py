import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app, db_connection

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_db_connections():
    # Ensure db_connection is reset for each test
    db_connection.clear()
    yield
    db_connection.clear()

@patch("src.main.supabase_files", return_value=[{"name": "file1.pdf"}, {"name": "file2.pdf"}])
@patch("src.main.supabase_delete")
def test_delete_all_files_success(mock_delete, mock_list):
    response = client.delete("/api/v1/documents/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert set(payload["filenames"]) == {"file1.pdf", "file2.pdf"}

@patch("src.main.supabase_files", return_value=[{"name": "file1.pdf"}])
@patch("src.main.supabase_delete", side_effect=Exception("delete fail"))
def test_delete_all_files_error(mock_delete, mock_list):
    response = client.delete("/api/v1/documents/")
    assert response.status_code == 500
    assert "detail" in response.json()

@patch("src.main.supabase_delete")
def test_delete_file_success(mock_delete):
    response = client.delete("/api/v1/documents/foo.pdf")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "filename": "foo.pdf"}

@patch("src.main.supabase_delete", side_effect=Exception("fail"))
def test_delete_file_error(mock_delete):
    response = client.delete("/api/v1/documents/foo.pdf")
    assert response.status_code == 500
    assert "detail" in response.json()


def test_drop_all_lancedb_tables_no_db():
    response = client.delete("/api/v1/lancedb/tables/")
    assert response.status_code == 500
    assert "LanceDB connection not initialized" in response.json()["detail"]


def test_drop_all_lancedb_tables_success(monkeypatch):
    class DummyDB:
        async def table_names(self):
            return ["t1", "t2"]
        async def drop_table(self, name):
            pass
    monkeypatch.setitem(db_connection, "db", DummyDB())
    response = client.delete("/api/v1/lancedb/tables/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert set(payload["tables"]) == {"t1", "t2"}


def test_drop_all_lancedb_tables_error(monkeypatch):
    class DummyDB:
        async def table_names(self):
            return ["t1"]
        async def drop_table(self, name):
            raise Exception("boom")
    monkeypatch.setitem(db_connection, "db", DummyDB())
    response = client.delete("/api/v1/lancedb/tables/")
    assert response.status_code == 500
    assert "detail" in response.json()


def test_drop_table_no_db():
    response = client.delete("/api/v1/lancedb/tables/foo")
    assert response.status_code == 500
    assert "LanceDB connection not initialized" in response.json()["detail"]


def test_drop_table_success(monkeypatch):
    class DummyDB:
        async def drop_table(self, name):
            pass
    monkeypatch.setitem(db_connection, "db", DummyDB())
    response = client.delete("/api/v1/lancedb/tables/foo_table")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "table": "foo_table"}


def test_drop_table_error(monkeypatch):
    class DummyDB:
        async def drop_table(self, name):
            raise Exception("fail")
    monkeypatch.setitem(db_connection, "db", DummyDB())
    response = client.delete("/api/v1/lancedb/tables/foo_table")
    assert response.status_code == 500
    assert "detail" in response.json()

@patch("src.main.dynamodb")
def test_get_embodiments_success(mock_dynamodb):
    table = mock_dynamodb.Table.return_value
    table.get_item.return_value = {"Item": {"embodiments": [{"foo": "bar"}]}}
    response = client.get(f"/api/v1/knowledge/embodiments/{uuid.uuid4()}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["data"], list)

@patch("src.main.dynamodb")
def test_get_embodiments_error(mock_dynamodb):
    table = mock_dynamodb.Table.return_value
    table.get_item.side_effect = Exception("boom")
    response = client.get(f"/api/v1/knowledge/embodiments/{uuid.uuid4()}")
    assert response.status_code == 500
    assert "detail" in response.json()
