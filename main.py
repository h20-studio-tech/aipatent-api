import os
import base64
import logging
import lancedb
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.pdf_workflow import FileProcessedError
from models.rag_typing import Chunk
from rag import multiquery_search, create_table_from_file, chunks_summary
from contextlib import asynccontextmanager
from lancedb.db import AsyncConnection
from pdf_processing import partition_request, supabase_upload, process_file, supabase_files
from utils.normalize_filename import normalize_filename

class FileUploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    message: str = Field(..., description="Status message for the upload operation")
    status_code: int = Field(..., description="HTTP status code indicating the result of the operation")
class MultiQueryResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: List[Chunk]

class Metadata(BaseModel):
    eTag: str
    size: int
    mimetype: str
    cacheControl: str
    lastModified: datetime
    contentLength: int
    httpStatusCode: int

class Document(BaseModel):
    name: str
    id: str
    updated_at: datetime
    created_at: datetime
    last_accessed_at: datetime
    metadata: Metadata

class FilesResponse(BaseModel):
    status: str
    response: List[Document]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

db_connection = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_connection["db"] = await lancedb.connect_async(os.getenv("LANCEDB_URI")) # type: ignore
    yield
    db_connection.clear()


app = FastAPI(title="aipatent", version="0.1.0", lifespan=lifespan)

origins = [
    "http://192.168.0.236:3000",
    "http://localhost:3000",
    "https://aipatent.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=None,
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/v1/documents/", response_model=FileUploadResponse, status_code=200 )
async def upload_file(file: UploadFile):
    """
    Upload and process a document file.

    This endpoint accepts an uploaded file and performs the following steps:
    
    1. Reads the file content.
    2. Uploads the file to Supabase storage.
    3. Processes the file to extract its data.
    4. Checks whether the file already exists in the vector store. If it does, it returns an
       appropriate message instructing the client to request a search instead.
    5. Creates a database table from the processed file data if the file is new.

    Parameters:
    - file (UploadFile): The file to be uploaded.

    Returns:
    - FileUploadResponse: A structured JSON response containing the filename and a status message.

    Raises:
    - HTTPException: Returns a 500 status code if an error occurs during file processing.
    """
    # Read file content asynchronously
    content = await file.read()
    filename = file.filename
    
    # normalized filename
    filename = normalize_filename(filename)
    try:
        # Step 1: Upload the file content to Supabase storage
        supabase_upload(content, filename, partition=False)

        # Step 2: Process the file (asynchronous operation)
        res = await process_file(content, filename, db = db_connection["db"])

        # Step 3: Check if the file already exists in the vector store
        if isinstance(res, FileProcessedError):
            return FileUploadResponse(
                filename=filename,
                message="File exists in vectorstore, request a search instead",
                status_code=401
            )
        
        # Step 4: Create a database table from the processed file data
        # The table name is derived from the filename (removing the ".pdf" extension)
        await create_table_from_file(res["filename"].replace(".pdf", ""), res["data"], db = db_connection["db"])

        # Return a successful response
        return FileUploadResponse(
            filename=filename,
            message="File uploaded successfully",
            status_code=200
        )
    except Exception as e:
        logging.info(f"upload_file error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")
    
@app.get("/api/v1/documents/", response_model=FilesResponse)
async def get_files():
    """
    Endpoint to retrieve a list of documents.
    
    Returns:
        FilesResponse: A Pydantic model containing the status and list of Document instances.
    """
    files = supabase_files()
    return FilesResponse(status="success", response=files)

@app.post("/api/v1/rag/multiquery-search/", response_model=MultiQueryResponse)
async def query_search(query: str, target_files: list[str]):
    """
    Performs a search query on the specified target files and returns the formatted chunks and a summary.
    Args:
        query (str): The search query string.
        target_files (list[str]): A list of target file names to search within.
    Returns:
        MultiQueryResponse: A Pydantic model containing the status, a summary message, and data (formatted chunks).
    Raises:
        HTTPException: If an error occurs during the multiquery search process.
    """
    
    table_names = [file.replace(".pdf", "") for file in target_files]
    try:
        formatted_chunks = await multiquery_search(query, table_names = table_names, db = db_connection["db"])
        summary = await chunks_summary(formatted_chunks, query)
        return MultiQueryResponse(status="success", message=summary, data=formatted_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during multiquery-search: {e}")

