#!/usr/bin/env python3
"""
Test script for the comprehensive analysis endpoint using FastAPI TestClient.
This tests the /api/v1/documents/comprehensive-analysis/ endpoint without running a server.
"""

from fastapi.testclient import TestClient
from src.main import app
import os
import json
from pathlib import Path

# Create test client
client = TestClient(app)

def test_comprehensive_analysis_endpoint():
    """Test the comprehensive analysis endpoint with a real PDF."""
    
    print("🚀 Testing Comprehensive Analysis Endpoint")
    print("=" * 60)
    
    # Path to test PDF
    pdf_path = "experiments/sample_patents/COVID-19 NEUTRALIZING ANTIBODY DETE.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Test file not found: {pdf_path}")
        return
    
    print(f"📄 Testing with: {Path(pdf_path).name}")
    print("-" * 40)
    
    try:
        # Open the PDF file
        with open(pdf_path, "rb") as pdf_file:
            print("📤 Sending request to endpoint...")
            
            # Make the request
            response = client.post(
                "/api/v1/documents/comprehensive-analysis/",
                files={"file": ("COVID-19_test.pdf", pdf_file, "application/pdf")}
            )
        
        # Print basic response info
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n📊 Response Structure:")
            print("-" * 30)
            for key in data.keys():
                if key == "parsed_content":
                    content_length = len(data[key]) if data[key] else 0
                    print(f"  • {key}: {content_length:,} characters")
                elif key == "gemini_analysis":
                    analysis_length = len(data[key]) if data[key] else 0
                    print(f"  • {key}: {analysis_length:,} characters")
                else:
                    print(f"  • {key}: {data[key]}")
            
            print(f"\n🔍 Gemini Analysis Preview:")
            print("-" * 40)
            analysis = data.get("gemini_analysis", "")
            preview = analysis[:500] + "..." if len(analysis) > 500 else analysis
            print(preview)
            
            print(f"\n📄 Parsed Content Preview:")
            print("-" * 40)
            parsed = data.get("parsed_content", "")
            parsed_preview = parsed[:300] + "..." if len(parsed) > 300 else parsed
            print(parsed_preview)
            
            # Basic validations
            assert "filename" in data, "Missing 'filename' in response"
            assert "gemini_analysis" in data, "Missing 'gemini_analysis' in response"
            assert "parsed_content" in data, "Missing 'parsed_content' in response"
            assert "timestamp" in data, "Missing 'timestamp' in response"
            
            print(f"\n✅ All assertions passed!")
            
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


def test_with_different_pdf():
    """Test with a different PDF if available."""
    
    print("\n" + "=" * 60)
    print("🔄 Testing with Alternative PDF")
    print("=" * 60)
    
    # Try alternative PDF
    alt_pdf_path = "experiments/sample_patents/test file.pdf"
    
    if not os.path.exists(alt_pdf_path):
        print(f"⚠️  Alternative test file not found: {alt_pdf_path}")
        return
    
    print(f"📄 Testing with: {Path(alt_pdf_path).name}")
    
    try:
        with open(alt_pdf_path, "rb") as pdf_file:
            response = client.post(
                "/api/v1/documents/comprehensive-analysis/",
                files={"file": ("test_file.pdf", pdf_file, "application/pdf")}
            )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Analysis Length: {len(data.get('gemini_analysis', '')):,} characters")
            print(f"📄 Content Length: {len(data.get('parsed_content', '')):,} characters")
            print("✅ Alternative PDF test passed!")
        else:
            print(f"❌ Alternative PDF test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with alternative PDF: {e}")


def test_error_handling():
    """Test error handling with invalid file."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Error Handling")
    print("=" * 60)
    
    try:
        # Test with invalid file content
        response = client.post(
            "/api/v1/documents/comprehensive-analysis/",
            files={"file": ("invalid.pdf", b"not a real pdf", "application/pdf")}
        )
        
        print(f"📊 Invalid file status: {response.status_code}")
        
        if response.status_code == 500:
            print("✅ Error handling working correctly (500 status)")
        else:
            print(f"⚠️  Unexpected status for invalid file: {response.status_code}")
            
    except Exception as e:
        print(f"Error during error handling test: {e}")


if __name__ == "__main__":
    print("🧪 Comprehensive Analysis Endpoint Test Suite")
    print("=" * 60)
    
    # Run tests
    test_comprehensive_analysis_endpoint()
    test_with_different_pdf()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("✅ Test Suite Complete!")
    print("=" * 60)