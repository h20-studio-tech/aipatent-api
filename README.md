# AIPatent API

A FastAPI-based microservice for managing and processing patent-related documents. This project is under active development.

## Overview

**AIPatent API** provides endpoints to:
- Upload and process PDF documents (extracting data, storing them in LanceDB, etc.).
- Perform multi-query searches for retrieval-augmented generation (RAG).
- Store and retrieve patent project metadata in DynamoDB.


1. **Clone the repository**:

```bash
git clone https://github.com/your-username/aipatent-api.git
cd aipatent-api

python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

pip install -r requirements.txt
```
2. **Create a virtual environment (optional but recommended)**:
```bash

python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
    - Create a .env file in the project root.

    - Add necessary variables (e.g., AWS credentials, LanceDB URI, etc.).

```bash

OPENAI_API_KEY
EMBEDDING_MODEL_NAME
LANGFUSE_SECRET_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_HOST
UNSTRUCTURED_API_KEY
UNSTRUCTURED_API_URL
COHERE_API_KEY
ENV
MODEL
LANCEDB_URI
LANCEDB_TABLE
SUPABASE_URL
SUPABASE_BUCKET_NAME
SUPABASE_DB_PASSWORD
SUPABASE_PUBLIC_KEY
SUPABASE_SECRET_KEY
ACCESS_KEY_ID
SECRET_ACCESS_KEY

```