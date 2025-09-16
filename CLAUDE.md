# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commit Guidelines

**NEVER use co-signing in commits.** Do not add "Co-Authored-By: Claude" or similar co-signing tags to commit messages. Keep commit messages clean and focused on the actual changes.

## Project Overview

AIPatent API is a FastAPI-based microservice for processing and managing patent-related documents. It provides intelligent patent document analysis, embodiment extraction, glossary term identification, and patent section generation using AI models.

## Architecture

### Core Components

1. **Main API Application** (`src/main.py`)
   - FastAPI application with CORS middleware
   - Manages connections to AWS services (LanceDB, DynamoDB, S3)
   - Handles document uploads, patent processing, and knowledge management
   - Includes automatic AWS credential refresh for LanceDB connections

2. **Patent Document Processing Pipeline**
   - OCR-based patent document processing with Tesseract and Unstructured API
   - Intelligent section segmentation and embodiment extraction
   - Glossary term extraction with definitions
   - Abstract extraction with page detection
   - Page-based content storage for source tracing

3. **Storage Systems**
   - **LanceDB**: Vector storage for document embeddings and RAG operations
   - **DynamoDB**: Metadata storage for patent projects and synthetic embodiments
   - **Supabase**: Document storage and relational data (embodiments, glossary terms, sections)

4. **AI Integration**
   - Multiple LLM providers (OpenAI, Google GenAI, Cohere)
   - Instructor library for structured output
   - Langfuse for LLM observability and tracing
   - Patent section generation endpoints for various patent components

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies using uv (preferred)
uv sync

# Or using pip
pip install -r requirements.txt

# Create and configure .env file with required environment variables
```

### Running the Application
```bash
# Development server with auto-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production server
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

# Run integration tests only
uv run pytest tests/integration/

# Run with verbose output
uv run pytest -v
```

### Type Checking and Linting
```bash
# Type checking (if mypy is configured)
uv run mypy src/

# Format code with black (if configured)
uv run black src/

# Lint with ruff (if configured)
uv run ruff check src/
```
js
## Key API Endpoints

### Document Processing
- `POST /api/v1/documents/` - Upload and process documents for vector storage
- `GET /api/v1/documents/` - List processed documents available for querying
- `POST /api/v1/patent/{patent_id}/` - Process patent document with OCR

### Patent Sections Generation
- `POST /api/v1/sections/background` - Generate background section
- `POST /api/v1/sections/claims` - Generate patent claims
- `POST /api/v1/sections/abstract` - Generate abstract
- `POST /api/v1/sections/field_of_invention` - Generate field of invention
- `POST /api/v1/sections/summary` - Generate summary section

### RAG and Search
- `POST /api/v1/rag/multiquery-search/` - Perform multi-query vector search with judgment

### Knowledge Management
- `POST /api/v1/project/` - Create new patent project
- `POST /api/v1/embodiment/` - Generate synthetic embodiment
- `POST /api/v1/embodiment/approve/` - Approve and store embodiment
- Various knowledge endpoints for approach, innovation, technology knowledge

## Environment Variables

Critical environment variables that must be configured:
- `OPENAI_API_KEY` - OpenAI API key for LLM operations
- `LANCEDB_CLOUD_KEY` - LanceDB Cloud API key for vector storage
- `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY` - AWS credentials (for DynamoDB)
- `SUPABASE_URL`, `SUPABASE_SECRET_KEY` - Supabase connection
- `LANGFUSE_*` - Langfuse observability configuration
- `LLAMA_CLOUD_API_KEY` - LlamaParse API key for document processing
- `DYNAMODB_TABLE` - DynamoDB table name (default: "patents")

## Testing Approach

The project uses pytest with both unit and integration tests:
- Unit tests mock external dependencies
- Integration tests may require live API connections
- Tests are organized in `tests/unit/` and `tests/integration/`
- Use `conftest.py` for shared fixtures

## Important Patterns

1. **Async Operations**: Most operations are async, especially document processing and LLM calls
2. **Error Handling**: HTTPExceptions with appropriate status codes for API errors
3. **Pydantic Models**: Extensive use of Pydantic for request/response validation (`src/models/`)
4. **Tracing**: Langfuse integration for LLM call tracing and debugging
5. **Connection Management**: Automatic refresh of AWS credentials for long-running connections