# Comprehensive Analysis Endpoint

## Overview

The Comprehensive Analysis endpoint provides intelligent document analysis using LLaMA Parse for PDF extraction and Google Gemini 2.5 Flash for AI-powered analysis. This endpoint is designed for patent documents and technical papers but works with any PDF content.

## Endpoint Details

- **URL**: `POST /api/v1/documents/comprehensive-analysis/`
- **Content-Type**: `multipart/form-data`
- **Authentication**: None required
- **Tags**: `Documents`

## Request Format

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | PDF file to analyze (multipart upload) |

### Example Request

#### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/documents/comprehensive-analysis/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### JavaScript (Fetch API)
```javascript
const formData = new FormData();
formData.append('file', pdfFile); // pdfFile is a File object

const response = await fetch('/api/v1/documents/comprehensive-analysis/', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

#### JavaScript (Axios)
```javascript
const formData = new FormData();
formData.append('file', pdfFile);

const response = await axios.post('/api/v1/documents/comprehensive-analysis/', formData, {
  headers: {
    'Content-Type': 'multipart/form-data'
  }
});
```

#### Python (Requests)
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post(
        'http://localhost:8000/api/v1/documents/comprehensive-analysis/',
        files=files
    )
    result = response.json()
```

## Response Format

### Success Response (200)

```json
{
  "filename": "document.pdf",
  "parsed_content": "--- Page 1 ---\n\nExtracted content from page 1...\n\n--- Page 2 ---\n\nExtracted content from page 2...",
  "gemini_analysis": "Comprehensive analysis including:\n\n**Document Summary:** This document is a...\n\n**Document Type:** Technical specification...",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Name of the uploaded file |
| `parsed_content` | string | Raw text extracted from PDF with page markers |
| `gemini_analysis` | string | AI-generated comprehensive analysis |
| `timestamp` | string | ISO timestamp when analysis was completed |

### Error Response (500)

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Analysis Structure

The `gemini_analysis` field contains a comprehensive analysis with the following sections:

1. **Document Summary** - 3-4 sentence overview
2. **Document Type** - Classification (patent, research paper, etc.)
3. **Key Technical Concepts** - Main technical terms and concepts
4. **Main Claims/Findings** - Important conclusions or claims
5. **Technical Details** - Specific implementation details
6. **Innovation Assessment** - Evaluation of novelty and innovation
7. **Potential Applications** - Real-world use cases
8. **Document Structure** - Organization and sections

## Frontend Implementation Guide

### File Upload Component

```javascript
function DocumentAnalysisUpload({ onAnalysisComplete }) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (file) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/documents/comprehensive-analysis/', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      onAnalysisComplete(result);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => handleFileUpload(e.target.files[0])}
        disabled={isAnalyzing}
      />
      {isAnalyzing && <div>Analyzing document... (this may take 60-120 seconds)</div>}
      {error && <div className="error">Error: {error}</div>}
    </div>
  );
}
```

### Analysis Results Component

```javascript
function AnalysisResults({ analysis }) {
  if (!analysis) return null;

  return (
    <div className="analysis-results">
      <h2>Analysis Results</h2>
      
      <div className="metadata">
        <p><strong>File:</strong> {analysis.filename}</p>
        <p><strong>Processed:</strong> {new Date(analysis.timestamp).toLocaleString()}</p>
        <p><strong>Content Length:</strong> {analysis.parsed_content.length.toLocaleString()} characters</p>
      </div>

      <div className="analysis-content">
        <h3>AI Analysis</h3>
        <div className="formatted-analysis">
          {analysis.gemini_analysis.split('\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>
      </div>

      <details>
        <summary>Raw Parsed Content</summary>
        <pre className="parsed-content">
          {analysis.parsed_content}
        </pre>
      </details>
    </div>
  );
}
```

## Performance Considerations

- **Processing Time**: Typically 60-120 seconds per document
- **File Size Limits**: Recommended max 50MB PDF files
- **Rate Limiting**: No current limits, but consider implementing client-side throttling
- **Timeout**: Default 2-minute timeout for requests

## Error Handling

### Common Error Scenarios

1. **Invalid File Format**: Non-PDF files will cause processing errors
2. **Corrupted PDF**: Damaged files may fail during parsing
3. **Large Files**: Very large PDFs may timeout
4. **API Limits**: LLaMA Parse or Gemini API limits may be reached

### Recommended Error Handling

```javascript
const handleAnalysis = async (file) => {
  try {
    // Validate file type
    if (file.type !== 'application/pdf') {
      throw new Error('Please upload a PDF file');
    }
    
    // Check file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      throw new Error('File size must be less than 50MB');
    }

    const result = await uploadForAnalysis(file);
    return result;
    
  } catch (error) {
    if (error.message.includes('500')) {
      return { error: 'Server error processing document. Please try again.' };
    }
    return { error: error.message };
  }
};
```

## Testing

### Test Files Available
- `experiments/sample_patents/COVID-19 NEUTRALIZING ANTIBODY DETE.pdf`
- `experiments/sample_patents/test file.pdf`

### Example Test Response
The analysis typically returns 2,000-5,000 characters of structured analysis for patent documents.

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Support

For technical issues or questions, refer to the main API documentation or contact the backend team.