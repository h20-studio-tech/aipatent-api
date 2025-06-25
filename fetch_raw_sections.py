import sys
from fastapi.testclient import TestClient
from src.main import app

def main(patent_id: str):
    client = TestClient(app)
    resp = client.get(f"/api/v1/raw-sections/{patent_id}")
    print("Status:", resp.status_code)
    try:
        print(resp.json())
    except ValueError:
        print(resp.text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_raw_sections.py <patent_id>")
        sys.exit(1)
    main(sys.argv[1])
