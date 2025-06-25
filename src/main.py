import os
import logging
import boto3
import asyncio
import boto3.dynamodb
import boto3.dynamodb.table
import lancedb
import uuid
from uuid import uuid4
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
    supabase_delete,
)
from src.utils.normalize_filename import normalize_filename
from dotenv import load_dotenv
from supabase import create_client, Client
from src.utils.ocr import process_patent_document
from src.utils.abstract_extractor import extract_abstract_from_pdf
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
     EmbodimentsListResponse,
     EmbodimentListResponse,
     ApprovedEmbodimentRequest,
     ApproachKnowledge,
     InnovationKnowledge,
     TechnologyKnowledge,
     ResearchNote,
     ApproachKnowledgeListResponse,
     InnovationKnowledgeListResponse,
     TechnologyKnowledgeListResponse,
     ResearchNoteListResponse,
     DropTablesResponse,
     DeleteFileResponse,
     DeleteAllFilesResponse,
     DropTableResponse,
     PatentFilesListResponse,
     RawSectionsResponse
 )
from src.models.ocr_schemas import (
    Embodiment, 
    DetailedDescriptionEmbodiment, 
    Glossary,
    GlossaryDefinition
)
from src.router.sections import router as sections_router
load_dotenv(".env")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

db_connection = {}
url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_SECRET_KEY')
supabase: Client = create_client(url, key)

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

app.include_router(sections_router)

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

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Hello World"}


@app.post("/api/v1/documents/", response_model=FileUploadResponse, status_code=200, tags=["Documents"])
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


@app.get("/api/v1/documents/", response_model=FilesResponse, tags=["Documents"])
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


@app.post("/api/v1/rag/multiquery-search/", response_model=MultiQueryResponse, tags=["RAG"])
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


@app.post("/api/v1/patent/{patent_id}/", response_model=PatentUploadResponse, status_code=200, tags=["Patent"])
async def patent(patent_id: str, file: UploadFile):
    """
    Endpoint to process a patent document and extract embodiments and abstract, it returns the results if the file was previously processed.

    Args:
        patent_id (int): The ID of the patent to update.
        file (UploadFile): The patent document file to process.

    Returns:
        PatentResponse: A Pydantic model containing the filename, message, data (list of embodiments), abstract, and status code.

    Raises:
        HTTPException: If an error occurs during the patent document processing.
    """
    content = await file.read()
    filename = file.filename

    filename = normalize_filename(filename)
    try:
        # check if exists in db
        exist_in_db  = (
            supabase.table("patent_files")
            .select("id, abstract, sections")
            .eq("id", str(patent_id))
            .execute()
        ) 
        if exist_in_db.data:
            logging.info(f"Patent with ID {patent_id} exists.")
            # Get the abstract from the database
            abstract = exist_in_db.data[0].get("abstract")
            abstract_page = None
            abstract_pattern = None
            
            embodiments_response = (
                supabase.table("embodiments")
                .select("*") # Select all fields to get sub_category if it exists
                .eq("file_id", str(patent_id))
                .order("emb_number")
                .execute()
            )
            
            # Process the returned data into Pydantic models
            parsed_embodiments = []
            if embodiments_response.data:
                for record in embodiments_response.data:
                    # Basic check for essential fields, adapt if needed
                    if not all(k in record for k in ('text', 'page_number', 'section', 'emb_number', 'summary')):
                        print(f"Skipping record due to missing essential fields: {record.get('file_id')}")
                        continue

                    if record.get('sub_category'): # Check if sub_category exists and has a value
                        try:
                            record_for_desc_emb = {k: v for k, v in record.items() if k not in ('emb_number', 'file_id')}
                            parsed_embodiments.append(DetailedDescriptionEmbodiment(**record_for_desc_emb, filename=filename))
                        except Exception as e:
                            print(f"Error parsing DetailedDescriptionEmbodiment for record {record.get('file_id', 'N/A')}: {e}")
                    else:
                        # Ensure sub_category isn't passed to Embodiment if it exists but is None/empty
                        record_for_embodiment = {k: v for k, v in record.items() if k not in ('sub_category', 'file_id', 'emb_number')}
                        try:
                            parsed_embodiments.append(Embodiment(**record_for_embodiment, filename=filename))
                        except Exception as e:
                            print(f"Error parsing Embodiment for record {record.get('file_id')}: {e}")
            
            patent_embodiments: list[Embodiment | DetailedDescriptionEmbodiment] = parsed_embodiments
            glossary_rows = (
                supabase.table("glossary_terms")
                .select("*")
                .eq("file_id", str(patent_id))
                .execute()
            )

            if glossary_rows.data:
                glossary_defs = [
                    GlossaryDefinition(
                        term=row["term"],
                        definition=row["definition"],
                        page_number=row["page_number"],
                        filename=filename,
                    )
                    for row in glossary_rows.data
                ]
                glossary_subsection = Glossary(
                    definitions=glossary_defs,
                    filename=filename,
                )
            else:
                glossary_subsection = None
            sections = exist_in_db.data[0].get("sections")
            # Fetch raw sections from Supabase for already-processed patent
            raw_sections_rows = (
                supabase.table("raw_sections").select("section_type, text")
                
                .eq("file_id", str(patent_id))
                .execute()
            )
            raw_sections = {row["section_type"]: row["text"] for row in raw_sections_rows.data} if raw_sections_rows.data else {}
        else:
            logging.info(f"Patent with ID {patent_id} does not exist.")
            # process the doc because it does not exist in db
            glossary_subsection, patent_embodiments, sections, raw_sections = await process_patent_document(content, filename)
            
            # Extract abstract using our enhanced OCR-capable extractor
            abstract_result = await extract_abstract_from_pdf(content)
            abstract = abstract_result["abstract_text"] if abstract_result and abstract_result.get("abstract_text") else None
            abstract_page = abstract_result.get("abstract_page") if abstract_result else None
            abstract_pattern = abstract_result.get("abstract_pattern") if abstract_result else None
            
            logging.info(f"Abstract extraction result: {'Found on page ' + str(abstract_page) if abstract else 'Not found'}")
            
            try:
                # 1️⃣ Insert patent_files row with abstract
                supabase.table("patent_files").insert(
                    {
                        "id": str(patent_id),
                        "filename": filename,
                        "abstract": abstract
                    }
                ).execute()
                logging.info(f"Inserted patent_files id={patent_id} with abstract {bool(abstract)}")

                # 2️⃣ Insert glossary terms (FK satisfied)
                if glossary_subsection and glossary_subsection.definitions:
                    supabase.table("glossary_terms").insert(
                        [
                            {
                                "id": str(uuid4()),
                                "file_id": str(patent_id),
                                "term": d.term.lower(),
                                "definition": d.definition,
                                "page_number": d.page_number,
                            }
                            for d in glossary_subsection.definitions
                        ]
                    ).execute()
                    logging.info(
                        f"Inserted {len(glossary_subsection.definitions)} glossary terms for patent_id={patent_id}"
                    )

                # 3️⃣ Insert embodiments
                supabase.table("embodiments").insert(
                    [
                        {
                            "file_id": str(patent_id),
                            "emb_number": idx,
                            "text": emb.text,
                            **(
                                {"header": emb.header}
                                if isinstance(emb, DetailedDescriptionEmbodiment)
                                else {}
                            ),
                            "page_number": emb.page_number,
                            "section": emb.section,
                            "summary": emb.summary,
                            **(
                                {"sub_category": emb.sub_category}
                                if isinstance(emb, DetailedDescriptionEmbodiment)
                                else {}
                            ),
                        }
                        for idx, emb in enumerate(patent_embodiments, start=1)
                    ]
                ).execute()
                logging.info(f"Inserted {len(patent_embodiments)} embodiments for patent_id={patent_id}")
                # 4️⃣ Insert section hierarchy JSON
                if sections:
                    # Convert Pydantic models to dicts for JSONB insertion
                    sections_data = [sec.model_dump() for sec in sections]
                    supabase.table("patent_files").update(
                        {"sections": sections_data}
                    ).eq("id", str(patent_id)).execute()
                    logging.info(f"Inserted {len(sections_data)} sections for patent_id={patent_id}")
                    # 5️⃣ Upsert raw sections
                    if raw_sections:
                        supabase.table("raw_sections").upsert(
                            [
                                {
                                    "file_id": str(patent_id),
                                    "section_type": sec,
                                    "text": txt,
                                }
                                for sec, txt in raw_sections.items()
                            ]
                        ).execute()
                        logging.info(f"Upserted {len(raw_sections)} raw sections for patent_id={patent_id}")
            except Exception as db_e:
                logging.error(f"Failed to store data for patent_id={patent_id}: {db_e}")
              
        return PatentUploadResponse(
            filename=filename,
            file_id=str(patent_id),
            message="Patent document processed successfully",
            data=patent_embodiments,
            terms=glossary_subsection,
            abstract=abstract,
            abstract_page=abstract_page,
            abstract_pattern=abstract_pattern,
            sections=sections,
            status_code=200,
        )
    except Exception as e:
        logging.info(f"patent error during processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during patent processing: {e}"
        )
        
@app.get("/api/v1/raw-sections/{patent_id}", response_model=RawSectionsResponse, tags=["Patent"])
async def get_raw_sections(patent_id: str):
    """Retrieve the raw text for each patent section previously extracted and stored."""
    try:
        # Fetch filename for friendly response details (ignore errors if not found)
        file_resp = (
            supabase.table("patent_files")
            .select("filename")
            .eq("id", str(patent_id))
            .maybe_single()
            .execute()
        )
        filename = file_resp.data["filename"] if file_resp and file_resp.data else ""

        rows = (
            supabase.table("raw_sections").select("section_type, text")
            
            .eq("file_id", str(patent_id))
            .execute()
        )
        sections = {row["section_type"]: row["text"] for row in rows.data} if rows.data else {}
        return RawSectionsResponse(file_id=str(patent_id), filename=filename, sections=sections)
    except Exception as e:
        logging.error(f"Failed to retrieve raw sections for patent_id={patent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving raw sections: {e}")

@app.get("/api/v1/patent-files/", response_model=PatentFilesListResponse, status_code=200, tags=["Patent"])
async def list_patent_files():
    """
    Endpoint to list all patent files in the database, ordered by upload time (newest first).
    
    Returns:
        PatentFilesListResponse,
      RawSectionsResponse,
      RawSectionsResponse: A Pydantic model containing the list of patent files and metadata.
    
    Raises:
        HTTPException: If an error occurs during database query.
    """
    try:
        response = (
            supabase.table("patent_files")
            .select("*")
            .order("uploaded_at", desc=True)  # Use the index for efficient sorting
            .execute()
        )
        
        if response.data:
            return PatentFilesListResponse(
                data=response.data,
                count=len(response.data),
                message="Patent files retrieved successfully",
                status="success",
                status_code=200
            )
        else:
            return PatentFilesListResponse(
                data=[],
                count=0,
                message="No patent files found",
                status="success",
                status_code=200
            )
    except Exception as e:
        logging.error(f"Error retrieving patent files: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving patent files: {e}"
        )

@app.get("/api/v1/source-embodiments/{patent_id}", response_model=EmbodimentsListResponse, tags=["Patent"])
async def list_source_embodiments(patent_id: str):
    """
    Retrieve the list of source embodiments for a given patent_id.
    Returns an empty list if none are found.
    """
    try:
        response = (
            supabase.table("embodiments")
            .select("*")
            .eq("file_id", str(patent_id))
            .order("emb_number")
            .execute()
        )
        
        terms = supabase.table("glossary_terms").select("*").eq("file_id", str(patent_id)).execute()
        terms = terms.data if terms.data else []
        source_embodiments = response.data if response.data else []
        
        file = (supabase.table("patent_files").select("*").eq("id", str(patent_id)).execute()).data if (supabase.table("patent_files").select("*").eq("id", str(patent_id)).execute()).data else []
        abstract = file[0].get("abstract", "") if file else ""
        sections = file[0].get("sections", []) if file else []
        
        return EmbodimentsListResponse(
            filename=file[0].get("filename", "") if file else "",
            file_id=patent_id,
            status="success",
            message="Source embodiments retrieved successfully",
            data=source_embodiments,
            terms=terms,
            abstract=abstract,
            sections=sections,
            status_code=200
        )
    except Exception as e:
        logging.error(f"Failed to retrieve source embodiments for patent_id={patent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving source embodiments: {e}")    


@app.post("/api/v1/project/", response_model=PatentProjectResponse, status_code=200, tags=["Project"])
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


@app.get("/api/v1/projects/", response_model=PatentProjectListResponse, status_code=200, tags=["Project"])
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
    

@app.post("/api/v1/embodiment/", status_code=200, tags=["Embodiment"])
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
            request.file_id,
            request.inspiration,
            request.knowledge,
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
    response_model=Union[EmbodimentApproveSuccessResponse, EmbodimentApproveErrorResponse],
    tags=["Embodiment"],
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

    
    
@app.post("/api/v1/knowledge/approach/", tags=["Knowledge"])
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
    

@app.post("/api/v1/knowledge/innovation/", tags=["Knowledge"])
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

@app.post("/api/v1/knowledge/technology/", tags=["Knowledge"])
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
        
@app.post("/api/v1/knowledge/research-note/", tags=["Knowledge"])
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


@app.get("/api/v1/knowledge/approach/{patent_id}", response_model=ApproachKnowledgeListResponse, tags=["Knowledge"])
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

@app.get("/api/v1/knowledge/innovation/{patent_id}", response_model=InnovationKnowledgeListResponse, tags=["Knowledge"])
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

@app.get("/api/v1/knowledge/technology/{patent_id}", response_model=TechnologyKnowledgeListResponse, tags=["Knowledge"])
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

@app.get("/api/v1/knowledge/research-note/{patent_id}", response_model=ResearchNoteListResponse, tags=["Knowledge"])
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

@app.get("/api/v1/knowledge/embodiments/{patent_id}", response_model=EmbodimentListResponse, tags=["Knowledge"])
async def get_embodiments(patent_id: str):
    """
    Retrieve stored embodiments for a given patent.
    """
    try:
        table = dynamodb.Table(os.getenv("DYNAMODB_TABLE"))
        response = table.get_item(Key={"patent_id": patent_id})
        item = response.get("Item", {})
        data = item.get("embodiments", [])
        return EmbodimentListResponse(status="success", data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/lancedb/tables/", response_model=DropTablesResponse, tags=["Lancedb"])
async def drop_all_lancedb_tables():
    """
    Drop all tables in LanceDB.
    """
    try:
        db = db_connection.get("db")
        if db is None:
            raise HTTPException(status_code=500, detail="LanceDB connection not initialized")
        tables = await db.table_names()
        dropped = []
        for tbl in tables:
            await db.drop_table(tbl)
            dropped.append(tbl)
        return DropTablesResponse(status="success", tables=dropped)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{filename}", response_model=DeleteFileResponse, tags=["Documents"])
async def delete_file(filename: str):
    """
    Delete a file from Supabase storage.
    """
    try:
        supabase_delete(filename)
        return DeleteFileResponse(status="success", filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/documents/", response_model=DeleteAllFilesResponse, tags=["Documents"])
async def delete_all_files():
    """
    Delete all files from Supabase storage.
    """
    try:
        files = supabase_files()
        deleted = []
        for file in files:
            name = file.get("name")
            supabase_delete(name)
            deleted.append(name)
        return DeleteAllFilesResponse(status="success", filenames=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/lancedb/tables/{table_name}", response_model=DropTableResponse, tags=["Lancedb"])
async def drop_table(table_name: str):
    """
    Drop a specific LanceDB table.
    """
    try:
        db = db_connection.get("db")
        if db is None:
            raise HTTPException(status_code=500, detail="LanceDB connection not initialized")
        await db.drop_table(table_name)
        return DropTableResponse(status="success", table=table_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))