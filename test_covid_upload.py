import uuid
import pathlib
from fastapi.testclient import TestClient
from src.main import app

# Path to sample COVID-19 patent PDF
PDF_PATH = r"experiments\sample_patents\COVID-19 NEUTRALIZING ANTIBODY DETE.pdf"

pdf_file = pathlib.Path(PDF_PATH)
if not pdf_file.exists():
    raise FileNotFoundError(f"Sample PDF not found: {pdf_file}")

client = TestClient(app)
patent_id = str(uuid.uuid4())
print("Using patent_id:", patent_id)

with pdf_file.open("rb") as f:
    resp = client.post(f"/api/v1/patent/{patent_id}/", files={"file": (pdf_file.name, f, "application/pdf")})

print("Status:", resp.status_code)
try:
    print(resp.json())
except ValueError:
    print("Non-JSON response body:")
    print(resp.text)
