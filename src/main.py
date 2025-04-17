import os
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
from typing import Union
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.models.pdf_workflow import FileProcessedError
from src.rag import multiquery_search, create_table_from_file, chunks_summary
from contextlib import asynccontextmanager
from src.pdf_processing import (
    supabase_upload,
    process_file,
    supabase_files,
    lancedb_tables,
)
from src.utils.normalize_filename import normalize_filename
from dotenv import load_dotenv
from src.utils.ocr import process_patent_document
from src.embodiment_generation import generate_embodiment
from src.models.api_schemas import (
     FileUploadResponse,
     PatentProject,
     PatentProjectItem,
     PatentProjectListResponse,
     PatentProjectResponse,
     PatentUploadResponse,
     MultiQueryResponse,
     FilesResponse,
     SyntheticEmbodimentRequest,
     EmbodimentApproveSuccessResponse,
     EmbodimentApproveErrorResponse,
     ApprovedEmbodimentRequest,
     ApproachKnowledge,
     InnovationKnowledge,
     TechnologyKnowledge,
     ResearchNote,
     ApproachKnowledgeListResponse,
     InnovationKnowledgeListResponse,
     TechnologyKnowledgeListResponse,
     ResearchNoteListResponse,
)

load_dotenv(".env")


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
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    logging.info(f"Using key ID that starts with: {aws_access_key[:4] if aws_access_key else 'None'}")
    
    # Create session - if running on EC2 with instance profile, explicit credentials not needed
    if aws_access_key and aws_secret_key:
        logging.info("Using explicit AWS credentials from environment variables")
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )
    else:
        logging.info("No explicit credentials found, using instance profile if available")
        session = boto3.Session(region_name=aws_region)
        
    # Test if we have permissions before proceeding
    try:
        sts_client = session.client("sts")
        identity = sts_client.get_caller_identity()
        logging.info(f"Successfully authenticated as: {identity['Arn']}")
    except Exception as e:
        logging.error(f"AWS authentication error: {e}")
        raise ValueError("Failed to authenticate with AWS. Check credentials or instance profile.")

    # Use STS to get temporary session credentials
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

    refresh_task = None  # Initialize refresh_task to None
    # Validate that the LanceDB URI is set
    if not lancedb_uri:
        raise ValueError("LANCEDB_URI environment variable is missing.")

    try:
        # Use direct AWS credentials instead of temporary ones
        aws_access_key = os.getenv("ACCESS_KEY_ID")
        aws_secret_key = os.getenv("SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        if not aws_access_key or not aws_secret_key:
            logging.error("AWS credentials are missing. LanceDB connection will likely fail.")
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")
            
        logging.info(f"DynamoDB using key ID that starts with: {aws_access_key[:4] if aws_access_key else 'None'}")
        
        # Test AWS credentials
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            sts = session.client('sts')
            account_id = sts.get_caller_identity().get('Account')
            logging.info(f"DynamoDB connected to AWS account: {account_id}")
        except Exception as e:
            logging.error(f"AWS credential test failed: {str(e)}")
            raise

        # Initialize the LanceDB connection with direct AWS credentials
        db_connection["db"] = await lancedb.connect_async(
            lancedb_uri,
            storage_options={
                "aws_access_key_id": aws_access_key,
                "aws_secret_access_key": aws_secret_key,
                "region_name": aws_region
            },
        )

        # Test connection
        tables = await db_connection["db"].table_names()
        logging.info(f"Connected to LanceDB at {lancedb_uri} with tables: {tables}")

        # Start background task to refresh credentials and connection periodically
        # Modify to use direct credentials instead of temporary ones
        async def keep_alive_connection():
            while True:
                try:
                    # Just verify connection is still working
                    tables = await db_connection["db"].table_names()
                    logging.debug(f"LanceDB connection verified with {len(tables)} tables")
                except Exception as e:
                    logging.error(f"Connection error, attempting to reconnect: {e}")
                    # Reconnect with direct credentials
                    try:
                        db_connection["db"] = await lancedb.connect_async(
                            lancedb_uri,
                            storage_options={
                                "aws_access_key_id": aws_access_key,
                                "aws_secret_access_key": aws_secret_key,
                                "region_name": aws_region
                            },
                        )
                        logging.info("LanceDB connection refreshed successfully.")
                    except Exception as conn_error:
                        logging.error(f"Failed to reconnect: {conn_error}")
                
                # Wait before checking again
                await asyncio.sleep(3000)  # 5 minutes
                
        refresh_task = asyncio.create_task(keep_alive_connection())

        yield  # Keep the app running during its lifespan

    except Exception as e:
        logging.error(f"Failed to initialize LanceDB connection: {e}")
        raise

    finally:
        if refresh_task:  # Only cancel the task if it was started
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
async def upload_file(file: UploadFile, force_upload: bool = False):
    """
    Upload and process a document file.

    This endpoint accepts an uploaded file and performs the following steps:

    1. Reads the file content.
    2. Processes the file to extract its data.
    3. Checks whether the file already exists in LanceDB. If it does, it returns an
       appropriate message instructing the client to request a search instead,
       unless force_upload is set to True.
    4. Creates a database table from the processed file data if the file is new.
    5. Uploads the file to Supabase storage only after successful processing and LanceDB table creation.

    Parameters:
    - file (UploadFile): The file to be uploaded.
    - force_upload (bool, optional): Override existing file check. Defaults to False.

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
        # Verify AWS credentials and LanceDB connection before proceeding
        try:
            # Log AWS identity for debugging purposes
            aws_access_key = os.getenv("ACCESS_KEY_ID")
            logging.info(f"Using AWS key ID that starts with: {aws_access_key[:4] if aws_access_key else 'None'}")
            
            # Check if LanceDB connection is valid
            if db_connection.get("db") is None:
                logging.error("LanceDB connection is None. Cannot process file.")
                raise HTTPException(status_code=500, detail="Database connection error")
                
            # Verify connection by getting table names
            tables = await db_connection["db"].table_names()
            logging.info(f"LanceDB tables before processing: {tables}")
        except Exception as auth_error:
            logging.error(f"AWS/LanceDB authentication error before processing: {str(auth_error)}")
            raise HTTPException(status_code=500, detail=f"Authentication error: {str(auth_error)}")

        # Step 1: Process the file (asynchronous operation)
        try:
            res = await process_file(content, filename, db=db_connection["db"], force_reprocess=force_upload)
        except Exception as process_error:
            logging.error(f"Error processing file {filename}: {str(process_error)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(process_error)}")

        # Step 2: Check if the file already exists in the vector store
        if isinstance(res, FileProcessedError):
            return FileUploadResponse(
                filename=filename,
                message="File exists in vectorstore, request a search instead",
                status_code=401,
            )

        # Step 3: Create a database table from the processed file data
        # The table name is derived from the filename (removing the ".pdf" extension)
        try:
            await create_table_from_file(
                res["filename"].replace(".pdf", ""), res["data"], db=db_connection["db"]
            )
        except Exception as table_error:
            logging.error(f"Error creating LanceDB table for {filename}: {str(table_error)}")
            raise HTTPException(status_code=500, detail=f"Error creating database table: {str(table_error)}")
        
        # Step 4: Upload the file content to Supabase storage only after successful processing
        try:
            supabase_upload(content, filename, partition=False)
        except Exception as upload_error:
            logging.error(f"Error uploading file {filename} to Supabase: {str(upload_error)}")
            raise HTTPException(status_code=500, detail=f"Error uploading to storage: {str(upload_error)}")

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
    Endpoint to retrieve a list of documents from Supabase and indicate which ones are available for querying.
    
    This endpoint returns all files stored in Supabase storage and marks those that also
    have corresponding tables in LanceDB as queryable.

    Returns:
        FilesResponse: A Pydantic model containing the status, list of Document instances,
        and diagnostic information.
    """
    # Get all files from Supabase
    supabase_files_list = supabase_files()
    
    # Get all table names from LanceDB
    try:
        lancedb_table_names = await lancedb_tables(db_connection["db"])
        logging.info(f"Retrieved {len(lancedb_table_names)} tables from LanceDB")
    except Exception as e:
        logging.error(f"Error retrieving LanceDB tables: {str(e)}")
        lancedb_table_names = []
    
    # Create a set of LanceDB table names for faster lookup
    lancedb_filenames = {table_name for table_name in lancedb_table_names}
    
    # Mark files that are also available in LanceDB as queryable
    for file in supabase_files_list:
        # Check if the file name without .pdf extension exists as a table in LanceDB
        file_base_name = file["name"].replace(".pdf", "")
        file["queryable"] = file_base_name in lancedb_filenames
    
    # Create diagnostic information
    diagnostics = {
        "total_supabase_files": len(supabase_files_list),
        "total_lancedb_tables": len(lancedb_table_names),
        "supabase_file_names": [file["name"] for file in supabase_files_list],
        "lancedb_table_names": list(lancedb_table_names),
        "queryable_files_count": sum(1 for file in supabase_files_list if file.get("queryable", False))
    }
    
    queryable_files = [file for file in supabase_files_list if file['queryable']] 
    
    return FilesResponse(
        status="success", 
        response=queryable_files,
        diagnostics=diagnostics
    )


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
aws_region = os.getenv("AWS_REGION", "us-east-1")
aws_access_key = os.getenv("ACCESS_KEY_ID")
aws_secret_key = os.getenv("SECRET_ACCESS_KEY")

# Log the credentials being used for DynamoDB
logging.info(f"DynamoDB using key ID that starts with: {aws_access_key[:4] if aws_access_key else 'None'}")

# Always use explicit credentials for DynamoDB to ensure correct account access
dynamodb = boto3.resource(
    "dynamodb",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)

# Test DynamoDB connection
try:
    # Get account info for DynamoDB
    sts_client = boto3.client(
        'sts',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )
    dynamodb_account = sts_client.get_caller_identity()['Account']
    logging.info(f"DynamoDB connected to AWS account: {dynamodb_account}")
    
    # Verify table exists
    table = dynamodb.Table("patents")
    table.meta.client.describe_table(TableName="patents")
    logging.info("Successfully verified 'patents' table exists")
except Exception as e:
    logging.error(f"DynamoDB connection test failed: {e}")


@app.post("/api/v1/project/", response_model=PatentProjectResponse, status_code=200)
async def patent_project(patent: PatentProject):
    """
    Endpoint to create a new patent project.
    
    Creates a new patent project in the DynamoDB database with the provided
    information including name, antigen, and disease targets.
    
    Args:
        patent (PatentProject): The patent project details to be created
        
    Returns:
        PatentProjectResponse: Contains the generated patent_id, success message,
                              and HTTP status code
                              
    Raises:
        HTTPException: If there's an error creating the project in DynamoDB
    """
    try:
        # Get the table
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))

        # Generate new UUID for the patent
        patent_id = str(uuid.uuid4())

        # Create the item to be stored
        patent_item = {
            "patent_id": patent_id,
            **patent.model_dump(),  # Unpack all the validated fields from the PatentProject model
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
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
        error_message = e.response["Error"]["Message"]
        raise HTTPException(status_code=500, detail=f"Database error: {error_message}")

    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/api/v1/projects/", response_model=PatentProjectListResponse, status_code=200)
async def list_patent_projects():
    """
    List all patent projects stored in DynamoDB.
    
    This endpoint retrieves all patent projects that have been created in the system.
    Each project contains basic information such as name, antigen, disease, and creation date.
    
    Returns:
        PatentProjectListResponse: A list of patent projects with their details.
        
    Raises:
        HTTPException: If there's an error retrieving projects from DynamoDB.
    """
    try:
        # Get the table
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        
        # Scan the table to get all items
        response = table.scan()
        
        # Extract the items from the response
        items = response.get('Items', [])
        
        # Convert the items to PatentProjectItem objects
        projects = []
        for item in items:
            try:
                # Extract required fields from the DynamoDB item
                project = PatentProjectItem(
                    patent_id=item.get('patent_id'),
                    name=item.get('name'),
                    antigen=item.get('antigen'),
                    disease=item.get('disease'),
                    created_at=item.get('created_at'),
                    updated_at=item.get('updated_at'),
                )
                projects.append(project)
            except Exception as e:
                # Log malformed items but continue processing
                logging.warning(f"Skipping malformed project item: {e}")
        
        # Return the list of projects
        return PatentProjectListResponse(
            status="success",
            projects=projects
        )
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logging.error(f"DynamoDB error listing projects: {error_message}")
        raise HTTPException(status_code=500, detail=f"Database error: {error_message}")
    
    except Exception as e:
        logging.error(f"Unexpected error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@app.post("/api/v1/embodiment/", status_code=200)
async def synthetic_embodiment(request: SyntheticEmbodimentRequest):
    """
    Generate a synthetic embodiment based on the provided parameters.
    
    Args:
        request (SyntheticEmbodimentRequest): The input parameters for generating a synthetic embodiment.
    Returns:
        dict: A dictionary containing the generated embodiment content.
    Raises:
        HTTPException: If an error occurs during embodiment generation.
    """
    try:
        res = await generate_embodiment(
            request.inspiration,
            request.source_embodiment,
            request.patent_title,
            request.disease,
            request.antigen
        )
        content = res.content
        return {"content": content}
    except Exception as e:
        logging.error(f"Error generating synthetic embodiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating synthetic embodiment: {str(e)}")


@app.post(
    "/api/v1/embodiment/approve/",
    response_model=Union[EmbodimentApproveSuccessResponse, EmbodimentApproveErrorResponse]
)
async def embodiment_approve(request: ApprovedEmbodimentRequest):
    """
    Approve and store an embodiment for a given patent.
    
    Args:
        request (ApprovedEmbodimentRequest): The request body containing the patent ID and the embodiment to store.
    Returns:
        Union[EmbodimentApproveSuccessResponse, EmbodimentApproveErrorResponse]: Status and update response from the database.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.update_item(
            Key={
                "patent_id": request.patent_id
            },
            UpdateExpression="SET embodiments = list_append(if_not_exists(embodiments, :empty_list), :embodiment)",
            ExpressionAttributeValues={
                ":embodiment": [request.embodiment.model_dump()],  # Convert Pydantic model to dict
                ":empty_list": []
            },
            ReturnValues="UPDATED_NEW"
        )
        return EmbodimentApproveSuccessResponse(
            message="Embodiment added to patent",
            data=response.get("Attributes")
        )
    except Exception as e:
        return EmbodimentApproveErrorResponse(
            message=str(e)
        )

    
    
@app.post("/api/v1/knowledge/approach/")
async def store_approach_knowledge(request: ApproachKnowledge):
    """
    Store approach knowledge for a given patent and update the patent object in DynamoDB.
    Args:
        request (ApproachKnowledge): The request body containing the patent ID and the approach knowledge to store.
    Returns:
        dict: Status and update response from the database.
    """

    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.update_item(
            Key={
                "patent_id": str(request.patent_id)
            },
            UpdateExpression="SET approach_knowledge = list_append(if_not_exists(approach_knowledge, :empty_list), :knowledge)",
            ExpressionAttributeValues={
                ":knowledge": [request.model_dump()],
                ":empty_list": []
            },
            ReturnValues="UPDATED_NEW"
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Approach knowledge added to patent",
                "data": response.get("Attributes")
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    

@app.post("/api/v1/knowledge/innovation/")
async def store_innovation_knowledge(request: InnovationKnowledge):
    """
    Store innovation knowledge for a given patent and update the patent object in DynamoDB.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.update_item(
            Key={
                "patent_id": str(request.patent_id)
            },
            UpdateExpression="SET innovation_knowledge = list_append(if_not_exists(innovation_knowledge, :empty_list), :knowledge)",
            ExpressionAttributeValues={
                ":knowledge": [request.model_dump()],
                ":empty_list": []
            },
            ReturnValues="UPDATED_NEW"
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Innovation knowledge added to patent",
                "data": response.get("Attributes")
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/api/v1/knowledge/technology/")
async def store_technology_knowledge(request: TechnologyKnowledge):
    """
    Store technology knowledge for a given patent and update the patent object in DynamoDB.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.update_item(
            Key={
                "patent_id": str(request.patent_id)
            },
            UpdateExpression="SET technology_knowledge = list_append(if_not_exists(technology_knowledge, :empty_list), :knowledge)",
            ExpressionAttributeValues={
                ":knowledge": [request.model_dump()],
                ":empty_list": []
            },
            ReturnValues="UPDATED_NEW"
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Technology knowledge added to patent",
                "data": response.get("Attributes")
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
        
@app.post("/api/v1/knowledge/research-note/")
async def store_research_note(request: ResearchNote):
    """
    Store research notes for a given patent and update the patent object in DynamoDB.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.update_item(
            Key={
                "patent_id": str(request.patent_id)
            },
            UpdateExpression="SET research_notes = list_append(if_not_exists(research_notes, :empty_list), :note)",
            ExpressionAttributeValues={
                ":note": [request.model_dump()],
                ":empty_list": []
            },
            ReturnValues="UPDATED_NEW"
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Research note added to patent",
                "data": response.get("Attributes")
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/api/v1/knowledge/approach/{patent_id}", response_model=ApproachKnowledgeListResponse)
async def get_approach_knowledge(patent_id: str):
    """
    Retrieve approach knowledge items for a given patent.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.get_item(Key={"patent_id": patent_id})
        item = response.get("Item", {})
        data = item.get("approach_knowledge", [])
        return ApproachKnowledgeListResponse(status="success", data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/knowledge/innovation/{patent_id}", response_model=InnovationKnowledgeListResponse)
async def get_innovation_knowledge(patent_id: str):
    """
    Retrieve innovation knowledge items for a given patent.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.get_item(Key={"patent_id": patent_id})
        item = response.get("Item", {})
        data = item.get("innovation_knowledge", [])
        return InnovationKnowledgeListResponse(status="success", data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/knowledge/technology/{patent_id}", response_model=TechnologyKnowledgeListResponse)
async def get_technology_knowledge(patent_id: str):
    """
    Retrieve technology knowledge items for a given patent.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.get_item(Key={"patent_id": patent_id})
        item = response.get("Item", {})
        data = item.get("technology_knowledge", [])
        return TechnologyKnowledgeListResponse(status="success", data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/knowledge/research-note/{patent_id}", response_model=ResearchNoteListResponse)
async def get_research_notes(patent_id: str):
    """
    Retrieve research notes for a given patent.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.get_item(Key={"patent_id": patent_id})
        item = response.get("Item", {})
        data = item.get("research_notes", [])
        return ResearchNoteListResponse(status="success", data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))