# AIPatent API - Feature Development Plan

## Project Overview
AIPatent API is a FastAPI-based microservice for processing and managing patent-related documents. It provides intelligent patent document analysis, embodiment extraction, glossary term identification, and patent section generation using AI models.

## System Architecture

### Core Technology Stack
- **Framework**: FastAPI with async/await support
- **Storage Systems**:
  - **LanceDB**: Vector storage for document embeddings and RAG operations (S3-backed)
  - **DynamoDB**: Metadata storage for patent projects and synthetic embodiments
  - **Supabase**: Document storage and relational data (embodiments, glossary terms, sections)
- **AI Integration**:
  - Multiple LLM providers (OpenAI, Google GenAI, Cohere)
  - Instructor library for structured output
  - Langfuse for LLM observability and tracing
- **Document Processing**:
  - OCR with Tesseract
  - Unstructured API for document parsing
  - PDF processing with page-level content storage

### Key Components

1. **Main Application** (`src/main.py`)
   - FastAPI app with CORS middleware
   - AWS connection management with automatic credential refresh
   - Lifespan management for LanceDB connections
   - Core API endpoints

2. **Patent Processing Pipeline**
   - OCR-based patent document processing (`src/utils/ocr.py`)
   - Section segmentation and extraction
   - Embodiment extraction (`src/embodiment_extraction.py`)
   - Glossary term extraction with definitions
   - Abstract extraction with page detection (`src/utils/abstract_extractor.py`)
   - Page-based content storage for source tracing

3. **AI Services**
   - Patent section generation (`src/patent_sections.py`, `src/patent_sections_async.py`)
   - Embodiment generation (`src/embodiment_generation.py`)
   - Comprehensive document analysis (`src/comprehensive_analysis.py`)
   - RAG operations (`src/rag.py`, `src/rag_workflow.py`)

4. **Data Models** (`src/models/`)
   - `api_schemas.py`: API request/response models
   - `ocr_schemas.py`: OCR and patent processing models
   - `rag_schemas.py`: RAG operation models
   - `sections_schemas.py`: Patent section generation models
   - `llm.py`: LLM configuration models

## Current API Endpoints

### Document Management
- `POST /api/v1/documents/` - Upload and process documents for vector storage
- `GET /api/v1/documents/` - List processed documents available for querying
- `DELETE /api/v1/documents/{filename}` - Delete specific document
- `DELETE /api/v1/documents/` - Delete all documents

### Patent Processing
- `POST /api/v1/patent/{patent_id}/` - Process patent document with OCR
- `GET /api/v1/raw-sections/{patent_id}` - Get raw patent sections
- `GET /api/v1/pages/{patent_id}` - Get page-based patent sections
- `GET /api/v1/patent-files/` - List all patent files
- `GET /api/v1/source-embodiments/{patent_id}` - Get source embodiments

### Patent Section Generation (`/api/v1/sections/`)
- `POST /background` - Generate background section
- `POST /summary` - Generate summary section
- `POST /field_of_invention` - Generate field of invention
- `POST /target_overview` - Generate target overview
- `POST /disease_overview` - Generate disease overview
- `POST /underlying_mechanism` - Generate underlying mechanism
- `POST /high_level_concept` - Generate high-level concept
- `POST /claims` - Generate patent claims
- `POST /abstract` - Generate abstract
- `POST /key_terms` - Generate key terms

### RAG and Search
- `POST /api/v1/rag/multiquery-search/` - Multi-query vector search with judgment
- `POST /api/v1/rag/retrieval/` - Direct retrieval from vector store

### Project Management
- `POST /api/v1/project/` - Create new patent project
- `GET /api/v1/projects/` - List all projects

### Embodiment Management
- `POST /api/v1/embodiment/` - Generate synthetic embodiment
- `POST /api/v1/embodiment/approve/` - Approve and store embodiment

### Knowledge Management
- `POST /api/v1/knowledge/approach/` - Store approach knowledge
- `POST /api/v1/knowledge/innovation/` - Store innovation knowledge
- `POST /api/v1/knowledge/technology/` - Store technology knowledge
- `POST /api/v1/knowledge/research-note/` - Store research notes
- `GET /api/v1/knowledge/{type}/{patent_id}` - Retrieve knowledge by type

### Comprehensive Analysis
- `POST /api/v1/documents/comprehensive-analysis/lancedb/{table_name}`
- `POST /api/v1/documents/comprehensive-analysis/filename/{filename}`
- `POST /api/v1/documents/comprehensive-analysis/storage-id/{file_id}`

### LanceDB Management
- `DELETE /api/v1/lancedb/tables/` - Drop all tables
- `DELETE /api/v1/lancedb/tables/{table_name}` - Drop specific table

## Environment Configuration

### Required Environment Variables
```env
# OpenAI Configuration
OPENAI_API_KEY=<your-key>

# AWS Configuration
ACCESS_KEY_ID=<aws-access-key>
SECRET_ACCESS_KEY=<aws-secret-key>
AWS_REGION=us-east-1
LANCEDB_URI=s3://<bucket>/lancedb

# Supabase Configuration
SUPABASE_URL=<supabase-url>
SUPABASE_SECRET_KEY=<supabase-key>

# DynamoDB Configuration
DYNAMODB_TABLE=patents

# Unstructured API
UNSTRUCTURED_API_KEY=<api-key>
UNSTRUCTURED_API_URL=<api-url>

# Langfuse Configuration (Optional)
LANGFUSE_PUBLIC_KEY=<public-key>
LANGFUSE_SECRET_KEY=<secret-key>
LANGFUSE_HOST=<host-url>
```

## Feature Development Guidelines

### 1. Adding New API Endpoints
- Define Pydantic models in `src/models/` for request/response validation
- Create endpoint in appropriate router or main.py
- Use async/await for all I/O operations
- Include proper error handling with HTTPException
- Add appropriate tags for API documentation

### 2. Integrating New AI Models
- Use the existing `src/utils/ai.py` module for LLM operations
- Leverage Instructor library for structured output
- Add Langfuse tracing for observability
- Handle rate limiting and retries appropriately

### 3. Storage Operations
- **LanceDB**: Use for vector embeddings and similarity search
- **DynamoDB**: Use for metadata and structured data
- **Supabase**: Use for file storage and relational data
- Always use the connection from `db_connection["db"]` for LanceDB

### 4. Document Processing
- Process documents page-by-page for better traceability
- Store page numbers with extracted content
- Use OCR for scanned documents
- Implement chunking strategies for large documents

### 5. Error Handling
```python
try:
    # Operation
    result = await some_operation()
except SpecificError as e:
    logging.error(f"Specific error: {str(e)}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logging.error(f"Unexpected error: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### 6. Testing Strategy
- Unit tests in `tests/unit/` with mocked dependencies
- Integration tests in `tests/integration/` for API endpoints
- Use pytest fixtures in `conftest.py` for shared test setup
- Mock external services (AWS, Supabase, LLMs) in unit tests

## Common Development Tasks

### Running the Application
```bash
# Development with auto-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/unit/test_ocr.py
```

### Code Quality
```bash
# Type checking (if configured)
uv run mypy src/

# Format code (if configured)
uv run black src/

# Lint (if configured)
uv run ruff check src/
```

## Implementation Patterns

### 1. Async Endpoint Pattern
```python
@app.post("/api/v1/endpoint", response_model=ResponseModel, tags=["Category"])
async def endpoint_name(request: RequestModel):
    """
    Endpoint description.
    
    Args:
        request: Request parameters
    
    Returns:
        ResponseModel: Structured response
    
    Raises:
        HTTPException: On error
    """
    try:
        # Process request
        result = await process_function(request.data)
        
        # Return response
        return ResponseModel(
            status="success",
            data=result
        )
    except Exception as e:
        logging.error(f"Error in endpoint_name: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. LanceDB Operation Pattern
```python
async def vector_operation(query: str, db):
    """Perform vector search operation."""
    try:
        # Ensure connection is valid
        if db is None:
            raise ValueError("Database connection not available")
        
        # Get or create table
        table = await db.open_table("table_name")
        
        # Perform operation
        results = await table.search(query).limit(10).to_list()
        
        return results
    except Exception as e:
        logging.error(f"Vector operation error: {str(e)}")
        raise
```

### 3. DynamoDB Operation Pattern
```python
def dynamo_operation(patent_id: str):
    """Perform DynamoDB operation."""
    try:
        table = dynamodb.Table("patents")
        
        # Put item
        table.put_item(
            Item={
                'patent_id': patent_id,
                'timestamp': datetime.now().isoformat(),
                'data': {...}
            }
        )
        
        # Get item
        response = table.get_item(
            Key={'patent_id': patent_id}
        )
        
        return response.get('Item')
    except ClientError as e:
        logging.error(f"DynamoDB error: {str(e)}")
        raise
```

## Key Considerations for New Features

1. **Authentication & Authorization**: Currently no auth middleware - consider adding for production
2. **Rate Limiting**: Implement rate limiting for LLM-intensive endpoints
3. **Caching**: Consider Redis for caching frequently accessed data
4. **Monitoring**: Expand Langfuse integration for comprehensive observability
5. **Data Validation**: Always use Pydantic models for input/output validation
6. **Error Recovery**: Implement retry logic for transient failures
7. **Scalability**: Design with horizontal scaling in mind
8. **Documentation**: Update OpenAPI docs with clear descriptions

## File Structure
```
aipatent-api/
├── src/
│   ├── main.py                    # Main FastAPI application
│   ├── models/                    # Pydantic models
│   │   ├── api_schemas.py
│   │   ├── ocr_schemas.py
│   │   ├── rag_schemas.py
│   │   └── sections_schemas.py
│   ├── routers/                   # API routers
│   │   └── sections.py
│   ├── services/                  # External service integrations
│   │   └── supabase.py
│   ├── utils/                     # Utility modules
│   │   ├── ai.py
│   │   ├── ocr.py
│   │   ├── langfuse_client.py
│   │   └── abstract_extractor.py
│   ├── comprehensive_analysis.py  # Document analysis service
│   ├── embodiment_generation.py   # Embodiment generation
│   ├── patent_sections.py         # Patent section generation
│   ├── pdf_processing.py          # PDF processing utilities
│   └── rag.py                     # RAG operations
├── tests/
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── .env                           # Environment variables
└── requirements.txt               # Python dependencies
```

## Next Steps for New Features

When implementing a new feature:

1. **Define Requirements**: Clearly specify what the feature should do
2. **Design API Contract**: Create Pydantic models for request/response
3. **Plan Storage Strategy**: Determine which storage system(s) to use
4. **Implement Core Logic**: Write the business logic with proper error handling
5. **Create API Endpoint**: Add endpoint with proper documentation
6. **Write Tests**: Create unit and integration tests
7. **Update Documentation**: Update this plan and API docs
8. **Deploy & Monitor**: Deploy with proper monitoring and logging

## Common Troubleshooting

1. **LanceDB Connection Issues**: Check AWS credentials and S3 permissions
2. **DynamoDB Errors**: Verify table exists and IAM permissions
3. **LLM Rate Limits**: Implement exponential backoff and retry logic
4. **Memory Issues**: Use streaming for large documents
5. **Timeout Errors**: Increase timeout or implement background processing

This plan provides a comprehensive overview of the AIPatent API system, making it easier for any LLM or developer to understand the codebase and implement new features effectively.