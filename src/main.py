import os
import base64
import logging
import boto3
import asyncio
import boto3.dynamodb
import boto3.dynamodb.table
import lancedb
import uuid
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.pdf_workflow import FileProcessedError
from models.rag_typing import Chunk
from src.rag import multiquery_search, create_table_from_file, chunks_summary
from contextlib import asynccontextmanager
from lancedb.db import AsyncConnection
from src.pdf_processing import (
    partition_request,
    supabase_upload,
    process_file,
    supabase_files,
    lancedb_tables,
)
from src.utils.normalize_filename import normalize_filename
from dotenv import load_dotenv
from src.utils.ocr import Embodiment, process_patent_document

load_dotenv(".env")


class FileUploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    message: str = Field(..., description="Status message for the upload operation")
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )


class PatentProject(BaseModel):
    name: str
    antigen: str
    disease: str


class PatentProjectResponse(BaseModel):
    patent_id: uuid.UUID
    message: str = Field(..., description="Status message for the upload operation")
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )


class PatentUploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    message: str = Field(..., description="Status message for the upload operation")
    data: list[Embodiment] = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )
    status_code: int = Field(
        ..., description="HTTP status code indicating the result of the operation"
    )


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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

db_connection = {}


def get_temporary_credentials(duration=3600):
    """Retrieve temporary AWS credentials using STS."""
    # Load AWS credentials from environment variables
    aws_access_key = os.getenv("ACCESS_KEY_ID")
    aws_secret_key = os.getenv("SECRET_ACCESS_KEY")

    # Ensure credentials exist before making an API call
    if not aws_access_key or not aws_secret_key:
        logging.error(
            "Missing AWS credentials. Ensure they are set in Heroku config vars."
        )
        raise ValueError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )

    # Initialize a session explicitly with access keys
    session = boto3.Session(
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )

    # Use STS to get temporary session credentials
    sts_client = session.client("sts")

    try:
        response = sts_client.get_session_token(DurationSeconds=duration)
        logging.info("Temporary AWS credentials obtained successfully.")
        return response["Credentials"]
    except Exception as e:
        logging.error(f"Failed to obtain AWS credentials: {e}")
        raise


async def refresh_lancedb_connection(lancedb_uri: str, refresh_interval: int = 3000):
    """Periodically refresh the LanceDB connection using new AWS credentials."""
    while True:
        try:
            creds = get_temporary_credentials()

            # Reinitialize the LanceDB connection with the new credentials
            new_connection = await lancedb.connect_async(
                lancedb_uri,
                storage_options={
                    "aws_access_key_id": creds["AccessKeyId"],
                    "aws_secret_access_key": creds["SecretAccessKey"],
                    "aws_session_token": creds["SessionToken"],
                },
            )
            db_connection["db"] = new_connection
            logging.info("LanceDB connection refreshed successfully.")
        except Exception as e:
            logging.error(f"Error refreshing LanceDB connection: {e}")

        # Wait for the refresh interval before updating again
        await asyncio.sleep(refresh_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    lancedb_uri = os.getenv("LANCEDB_URI")

    refresh_task = None  # ✅ Initialize refresh_task to None
    # Validate that the LanceDB URI is set
    if not lancedb_uri:
        raise ValueError("LANCEDB_URI environment variable is missing.")

    try:
        # Retrieve initial temporary credentials
        creds = get_temporary_credentials()

        # Initialize the initial LanceDB connection with storage options
        db_connection["db"] = await lancedb.connect_async(
            lancedb_uri,
            storage_options={
                "aws_access_key_id": creds["AccessKeyId"],
                "aws_secret_access_key": creds["SecretAccessKey"],
                "aws_session_token": creds["SessionToken"],
            },
        )

        # Test connection
        tables = await db_connection["db"].table_names()
        logging.info(f"Connected to LanceDB at {lancedb_uri} with tables: {tables}")

        # Start background task to refresh credentials and connection
        refresh_task = asyncio.create_task(refresh_lancedb_connection(lancedb_uri))

        yield  # ✅ Keep the app running during its lifespan

    except Exception as e:
        logging.error(f"Failed to initialize LanceDB connection: {e}")
        raise

    finally:
        if refresh_task:  # ✅ Only cancel the task if it was started
            refresh_task.cancel()
        db_connection.clear()


app = FastAPI(title="aipatent", version="0.1.0", lifespan=lifespan)

origins = ["*"]

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


@app.post("/api/v1/documents/", response_model=FileUploadResponse, status_code=200)
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
        res = await process_file(content, filename, db=db_connection["db"])

        # Step 3: Check if the file already exists in the vector store
        if isinstance(res, FileProcessedError):
            return FileUploadResponse(
                filename=filename,
                message="File exists in vectorstore, request a search instead",
                status_code=401,
            )

        # Step 4: Create a database table from the processed file data
        # The table name is derived from the filename (removing the ".pdf" extension)
        await create_table_from_file(
            res["filename"].replace(".pdf", ""), res["data"], db=db_connection["db"]
        )

        # Return a successful response
        return FileUploadResponse(
            filename=filename, message="File uploaded successfully", status_code=200
        )
    except Exception as e:
        logging.info(f"upload_file error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")


@app.get("/api/v1/documents/", response_model=FilesResponse)
async def get_files():
    """
    Endpoint to retrieve a list of documents in lancedb.

    Returns:
        FilesResponse: A Pydantic model containing the status and list of Document instances.
    """
    files = (
        supabase_files()
    )  # TODO: Filter Supabase files by lancedb tables for consistency
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
        formatted_chunks = await multiquery_search(
            query, table_names=table_names, db=db_connection["db"]
        )
        summary = await chunks_summary(formatted_chunks, query)
        return MultiQueryResponse(
            status="success", message=summary, data=formatted_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during multiquery-search: {e}"
        )


@app.post("/api/v1/patent/", response_model=PatentUploadResponse, status_code=200)
async def patent(file: UploadFile):
    """
    Endpoint to process a patent document and extract embodiments.

    Args:
        file (UploadFile): The patent document file to process.

    Returns:
        PatentResponse: A Pydantic model containing the filename, message, data (list of embodiments), and status code.

    Raises:
        HTTPException: If an error occurs during the patent document processing.
    """
    content = await file.read()
    filename = file.filename

    filename = normalize_filename(filename)
    try:
        patent_embodiments = await process_patent_document(content, filename)
        return PatentUploadResponse(
            filename=filename,
            message="Patent document processed successfully",
            data=patent_embodiments,
            status_code=200,
        )
    except Exception as e:
        logging.info(f"patent error during processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during patent processing: {e}"
        )


# Create DynamoDB client at module level for reuse
dynamodb = boto3.resource(
    "dynamodb",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


@app.post("/api/v1/project/", response_model=PatentProjectResponse, status_code=200)
async def patent_project(patent: PatentProject):
    try:
        # Get the table
        table = dynamodb.Table("patents")

        # Generate new UUID for the patent
        patent_id = str(uuid.uuid4())

        # Create the item to be stored
        patent_item = {
            "patent_id": patent_id,
            **patent.model_dump(),  # Unpack all the validated fields from the PatentProject model
            "created_at": datetime.now().isoformat(),
        }

        # Put the item in DynamoDB
        response = table.put_item(Item=patent_item)

        # Check if the put was successful
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            return PatentProjectResponse(
                patent_id=patent_id, message="Patent project created successfully", status_code=200
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to create patent project"
            )

    except ClientError as e:
        # Handle specific DynamoDB errors
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise HTTPException(status_code=500, detail=f"Database error: {error_message}")

    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
