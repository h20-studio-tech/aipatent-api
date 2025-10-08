#!/usr/bin/env python3
"""Create a simple test PDF document and test the upload pipeline."""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import requests
import os

def create_test_pdf():
    """Create a simple test PDF with patent-like content."""
    filename = "test_patent.pdf"

    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)

    # Page 1
    c.drawString(100, 750, "TEST PATENT APPLICATION")
    c.drawString(100, 720, "Title: Advanced AI-Powered Document Processing System")
    c.drawString(100, 690, "")
    c.drawString(100, 660, "FIELD OF THE INVENTION")
    c.drawString(100, 630, "This invention relates to artificial intelligence systems for document processing.")
    c.drawString(100, 600, "More specifically, it relates to methods and systems for parsing and analyzing")
    c.drawString(100, 570, "technical documents using large language models and vector databases.")

    c.drawString(100, 520, "BACKGROUND")
    c.drawString(100, 490, "Traditional document processing systems have limitations in understanding")
    c.drawString(100, 460, "complex technical content. Existing solutions often fail to maintain context")
    c.drawString(100, 430, "across document sections and struggle with specialized terminology.")

    # Page 2
    c.showPage()
    c.drawString(100, 750, "SUMMARY OF THE INVENTION")
    c.drawString(100, 720, "The present invention provides a novel approach to document processing")
    c.drawString(100, 690, "using advanced parsing techniques combined with vector storage systems.")
    c.drawString(100, 660, "The system employs LlamaParse for accurate document extraction and")
    c.drawString(100, 630, "LanceDB for efficient vector-based storage and retrieval.")

    c.drawString(100, 580, "DETAILED DESCRIPTION")
    c.drawString(100, 550, "The system comprises several key components:")
    c.drawString(120, 520, "1. A document parsing module using LlamaParse API")
    c.drawString(120, 490, "2. A chunking system with overlapping text segments")
    c.drawString(120, 460, "3. Vector embedding generation using OpenAI models")
    c.drawString(120, 430, "4. Cloud-based vector storage using LanceDB")
    c.drawString(120, 400, "5. Intelligent retrieval and search capabilities")

    c.save()
    print(f"Created test PDF: {filename}")
    return filename

def test_upload(pdf_file):
    """Test the document upload endpoint."""
    print(f"\n🧪 Testing upload of {pdf_file}...")

    url = "http://localhost:8000/api/v1/documents/"

    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file, f, 'application/pdf')}
            response = requests.post(url, files=files)

        print(f"Response Status: {response.status_code}")
        print(f"Response: {response.json()}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def check_lancedb_tables():
    """Check if tables were created in LanceDB."""
    print(f"\n🔍 Checking LanceDB tables...")

    try:
        import lancedb
        import os

        db = lancedb.connect(
            uri="db://aipatent-ym7e4b",
            api_key=os.getenv("LANCEDB_CLOUD_KEY"),
            region="us-east-1"
        )

        tables = db.table_names()
        print(f"✅ LanceDB tables: {tables}")

        if 'test_patent' in tables:
            table = db.open_table('test_patent')

            # Use search to get data (LanceDB Cloud compatible)
            try:
                # Simple search to get all data
                results = table.search('').limit(10).to_list()
                print(f"✅ Table 'test_patent' has {len(results)} rows (showing up to 10)")

                print("\n📄 Sample data:")
                for idx, row in enumerate(results[:3]):
                    print(f"  Row {idx}: {row.get('text', 'N/A')[:100]}...")
            except Exception as e:
                print(f"⚠️ Could not fetch data: {e}, but table exists")

        return len(tables) > 0

    except Exception as e:
        print(f"❌ LanceDB check failed: {e}")
        return False

if __name__ == "__main__":
    # Install reportlab if needed
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab...")
        os.system("pip install reportlab")
        import reportlab

    print("🚀 Starting end-to-end pipeline test...")

    # Step 1: Create test PDF
    pdf_file = create_test_pdf()

    # Step 2: Upload and process
    if test_upload(pdf_file):
        print("✅ Document uploaded successfully!")

        # Step 3: Check results
        import time
        print("⏳ Waiting for processing to complete...")
        time.sleep(10)  # Give it time to process

        if check_lancedb_tables():
            print("🎉 End-to-end pipeline test PASSED!")
        else:
            print("❌ LanceDB verification failed")
    else:
        print("❌ Document upload failed")

    # Cleanup
    if os.path.exists(pdf_file):
        os.remove(pdf_file)
        print(f"Cleaned up {pdf_file}")