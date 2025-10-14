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
from src.rag import multiquery_search, create_table_from_file, chunks_summary, get_chunks_by_ids
from contextlib import asynccontextmanager
from src.pdf_processing import (
    supabase_upload,
    process_file,
    supabase_files,
    lancedb_tables,
    supabase_delete,
    supabase_download,
)
from src.utils.normalize_filename import normalize_filename
from supabase import create_client, Client
from src.utils.ocr import process_patent_document
from src.utils.abstract_extractor import extract_abstract_from_pdf
from src.embodiment_generation import generate_embodiment
from src.comprehensive_analysis import ComprehensiveAnalysisService
from src.models.rag_schemas import (
    RetrievalRequest,
    RetrievalResponse,
    ChunksByIdsRequest,
    ChunksByIdsResponse
)
from src.models.api_schemas import (
     FileUploadResponse,
     PatentProject,
     PatentProjectItem,
     PatentProjectListResponse,
     PatentProjectResponse,
     PatentUploadResponse,
     MultiQueryResponse,
     FilesResponse,
     EmbodimentApproveSuccessResponse,
     EmbodimentApproveErrorResponse,
     EmbodimentsListResponse,
     EmbodimentListResponse,
     EmbodimentStatusUpdateRequest,
     ApprovedEmbodimentRequest,
     SyntheticEmbodimentRequest,
     ApproachKnowledge,
     InnovationKnowledge,
     TechnologyKnowledge,
     ResearchNote,
     ApproachKnowledgeListResponse,
     InnovationKnowledgeListResponse,
     ComprehensiveAnalysisRequest,
     ComprehensiveAnalysisResponse,
     TechnologyKnowledgeListResponse,
     ResearchNoteListResponse,
     DropTablesResponse,
     DeleteFileResponse,
     DeleteAllFilesResponse,
     DropTableResponse,
     PatentFilesListResponse,
     RawSectionsResponse,
     PageBasedSectionsResponse,
     PageData,
     ComponentUpdateRequest,
     ComponentUpdateResponse,
     PatentDraftSaveRequest,
     PatentDraftSaveResponse,
     PatentDraftResponse,
 )
from src.models.ocr_schemas import (
    Embodiment, 
    DetailedDescriptionEmbodiment, 
    Glossary,
    GlossaryDefinition,
    ProcessedPage
)
from src.routers.sections import router as sections_router
from dotenv import load_dotenv
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




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    lancedb_uri = os.getenv("LANCEDB_URI")

    refresh_task = None  # Initialize refresh_task to None
    # Validate that the LanceDB URI is set
    if not lancedb_uri:
        raise ValueError("LANCEDB_URI environment variable is missing.")

    try:
        # Get LanceDB Cloud credentials
        lancedb_api_key = os.getenv("LANCEDB_CLOUD_KEY")

        if not lancedb_api_key:
            logging.error("LanceDB Cloud API key is missing.")
            raise ValueError("LANCEDB_CLOUD_KEY environment variable is required")

        logging.info("Connecting to LanceDB Cloud...")

        # Initialize the LanceDB Cloud connection
        db_connection["db"] = await lancedb.connect_async(
            uri="db://aipatent-ym7e4b",
            api_key=lancedb_api_key,
            region="us-east-1"
        )

        # Test connection
        tables = await db_connection["db"].table_names()
        logging.info(f"Connected to LanceDB Cloud with tables: {tables}")

        # Simple connection health check (no credential refresh needed for cloud)
        async def keep_alive_connection():
            while True:
                try:
                    tables = await db_connection["db"].table_names()
                    logging.debug(f"LanceDB Cloud connection verified with {len(tables)} tables")
                except Exception as e:
                    logging.error(f"LanceDB Cloud connection error: {e}")
                    # Attempt to reconnect
                    try:
                        db_connection["db"] = await lancedb.connect_async(
                            uri="db://aipatent-ym7e4b",
                            api_key=lancedb_api_key,
                            region="us-east-1"
                        )
                        logging.info("LanceDB Cloud connection restored.")
                    except Exception as conn_error:
                        logging.error(f"Failed to reconnect to LanceDB Cloud: {conn_error}")

                await asyncio.sleep(300)  # Check every 5 minutes

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
    table = dynamodb.Table("aipatent-projects")
    table.meta.client.describe_table(TableName="aipatent-projects")
    logging.info("Successfully verified 'aipatent-projects' table exists")
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
        answer = await chunks_summary(formatted_chunks, query)

        return MultiQueryResponse(
            status="success", message=answer, data=formatted_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during multiquery-search: {e}"
        )
        
@app.post("/api/v1/rag/retrieval/", response_model=RetrievalResponse, tags=["RAG"])
async def retrieval(r: RetrievalRequest):
    """
    Retrieve relevant chunks from the vector store based on the query.
    
    Args:
        query (str): The search query string.
    
    Returns:
        RetrieveResponse: A Pydantic model containing the status, a summary message, and data (formatted chunks).
    Raises:
        HTTPException: If an error occurs during the retrieve process.
    """
    table_names = [file.replace(".pdf", "") for file in r.target_files]
    try:
        formatted_chunks = await multiquery_search(
            query=r.query, table_names=table_names, db=db_connection["db"]
            )
        
        return RetrievalResponse(
            status="success", message="Retrieved chunks successfully", data=formatted_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during retrieve: {e}"
        )

@app.post("/api/v1/rag/chunks-by-ids/", response_model=ChunksByIdsResponse, tags=["RAG"])
async def get_chunks_by_ids_endpoint(request: ChunksByIdsRequest):
    """
    Retrieve specific chunks by their IDs from the vector database.
    
    This endpoint enables traceability from stored knowledge back to original document chunks.
    When a stored knowledge entry references chunk IDs (e.g., [75, 12, 89]), this endpoint
    retrieves the full chunk data including text, page number, and filename.
    
    Args:
        request: ChunksByIdsRequest containing:
            - chunk_ids: List of chunk IDs to retrieve
            - document_names: List of document names (without .pdf extension)
    
    Returns:
        ChunksByIdsResponse: Contains status, message, retrieved chunks, and count
    
    Raises:
        HTTPException: If an error occurs during chunk retrieval
    
    Example:
        POST /api/v1/rag/chunks-by-ids/
        {
            "chunk_ids": [75, 12, 89],
            "document_names": ["vaccine_paper", "antibody_study"]
        }
    """
    try:
        logging.info(f"Retrieving chunks by IDs: {request.chunk_ids} from documents: {request.document_names}")
        
        # Retrieve chunks using the new function
        chunks = await get_chunks_by_ids(
            chunk_ids=request.chunk_ids,
            table_names=request.document_names,
            db=db_connection["db"]
        )
        
        return ChunksByIdsResponse(
            status="success",
            message=f"Successfully retrieved {len(chunks)} chunk(s)",
            chunks=chunks,
            count=len(chunks)
        )
    except Exception as e:
        logging.error(f"Error retrieving chunks by IDs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chunks by IDs: {str(e)}"
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
            glossary_subsection, patent_embodiments, sections, raw_sections, segmented_pages = await process_patent_document(content, filename)
            
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
                            "id": str(uuid4()),  # Generate unique UUID for each embodiment
                            "file_id": str(patent_id),
                            "emb_number": idx,
                            "text": emb.text,
                            "status": "pending",  # Default status for new embodiments
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
                            **(
                                {"start_char": emb.start_char, "end_char": emb.end_char}
                                if hasattr(emb, 'start_char') and hasattr(emb, 'end_char')
                                else {}
                            ),
                        }
                        for idx, emb in enumerate(patent_embodiments, start=1)
                    ]
                ).execute()
                logging.info(f"Inserted {len(patent_embodiments)} embodiments for patent_id={patent_id}")
                
                # 4️⃣ Insert page data for better source tracing
                if segmented_pages:
                    try:
                        supabase.table("pages").insert(
                            [
                                {
                                    "file_id": str(patent_id),
                                    "page_number": page.page_number,
                                    "text": page.text,
                                    "section": page.section,
                                    "filename": page.filename
                                }
                                for page in segmented_pages
                            ]
                        ).execute()
                        logging.info(f"Inserted {len(segmented_pages)} pages for patent_id={patent_id}")
                    except Exception as pages_e:
                        # Pages table might not exist yet, log but don't fail
                        logging.warning(f"Failed to insert pages data (table might not exist): {pages_e}")
                
                # 5️⃣ Insert section hierarchy JSON
                if sections:
                    # Convert Pydantic models to dicts for JSONB insertion
                    sections_data = [sec.model_dump() for sec in sections]
                    supabase.table("patent_files").update(
                        {"sections": sections_data}
                    ).eq("id", str(patent_id)).execute()
                    logging.info(f"Inserted {len(sections_data)} sections for patent_id={patent_id}")
                    # 6️⃣ Upsert raw sections
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
              
        # Convert GlossarySubsectionPage to Glossary for API response
        glossary_terms = None
        if glossary_subsection and hasattr(glossary_subsection, 'definitions'):
            glossary_terms = Glossary(
                definitions=glossary_subsection.definitions,
                filename=glossary_subsection.filename
            )
        
        return PatentUploadResponse(
            filename=filename,
            file_id=str(patent_id),
            message="Patent document processed successfully",
            data=patent_embodiments,
            terms=glossary_terms,
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


@app.get("/api/v1/pages/{patent_id}", response_model=PageBasedSectionsResponse, tags=["Patent"])
async def get_patent_pages(patent_id: str):
    """Retrieve patent content organized by pages for better navigation and source tracing.
    
    This endpoint provides the complete raw text content per page, enabling better
    source tracing when users click on processed embodiments in the UI.
    """
    try:
        # Fetch filename for friendly response details
        file_resp = (
            supabase.table("patent_files")
            .select("filename")
            .eq("id", str(patent_id))
            .maybe_single()
            .execute()
        )
        filename = file_resp.data["filename"] if file_resp and file_resp.data else ""

        # Try to get page data from a dedicated pages table first (if it exists)
        # This would contain the complete raw text per page from segmented_pages
        try:
            pages_resp = (
                supabase.table("pages")
                .select("page_number, text, section, filename")
                .eq("file_id", str(patent_id))
                .order("page_number")
                .execute()
            )
            
            if pages_resp.data:
                # Use complete page data if available
                pages = [
                    PageData(
                        page_number=page["page_number"],
                        text=page["text"],
                        section=page["section"],
                        filename=page["filename"] or filename
                    )
                    for page in pages_resp.data
                ]
                
                return PageBasedSectionsResponse(
                    file_id=str(patent_id),
                    filename=filename,
                    pages=pages,
                    total_pages=len(pages)
                )
        except Exception:
            # Pages table doesn't exist or query failed, fall back to embodiments
            pass

        # Fallback: Reconstruct pages from embodiments (partial content)
        embodiments_resp = (
            supabase.table("embodiments")
            .select("text, page_number, section, filename")
            .eq("file_id", str(patent_id))
            .order("page_number")
            .execute()
        )
        
        if not embodiments_resp.data:
            return PageBasedSectionsResponse(
                file_id=str(patent_id),
                filename=filename,
                pages=[],
                total_pages=0
            )
        
        # Group embodiments by page and aggregate text
        pages_dict = {}
        for emb in embodiments_resp.data:
            page_num = emb["page_number"]
            if page_num not in pages_dict:
                pages_dict[page_num] = {
                    "page_number": page_num,
                    "text": "",
                    "section": emb["section"],
                    "filename": emb["filename"] or filename
                }
            # Concatenate text from all embodiments on this page
            if pages_dict[page_num]["text"]:
                pages_dict[page_num]["text"] += "\n\n" + emb["text"]
            else:
                pages_dict[page_num]["text"] = emb["text"]
        
        # Convert to sorted list of PageData objects
        pages = [PageData(**page_data) for page_data in sorted(pages_dict.values(), key=lambda x: x["page_number"])]
        
        return PageBasedSectionsResponse(
            file_id=str(patent_id),
            filename=filename,
            pages=pages,
            total_pages=len(pages)
        )
        
    except Exception as e:
        logging.error(f"Failed to retrieve pages for patent_id={patent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving pages: {e}")


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

@app.get("/api/v1/source-embodiments/approved/{patent_id}", tags=["Patent"])
async def list_approved_embodiments(patent_id: str):
    """
    Retrieve only approved embodiments for a given patent_id.
    This endpoint is used by generation functions to only include approved content.
    """
    try:
        response = (
            supabase.table("embodiments")
            .select("*")
            .eq("file_id", str(patent_id))
            .eq("status", "approved")
            .order("emb_number")
            .execute()
        )

        approved_embodiments = response.data if response.data else []

        logging.info(f"Retrieved {len(approved_embodiments)} approved embodiments for patent_id={patent_id}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": f"Retrieved {len(approved_embodiments)} approved embodiments",
                "patent_id": patent_id,
                "data": approved_embodiments
            }
        )
    except Exception as e:
        logging.error(f"Failed to retrieve approved embodiments for patent_id={patent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving approved embodiments: {e}")


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

@app.put("/api/v1/embodiment/status/", tags=["Embodiment"])
async def update_embodiment_status(request: EmbodimentStatusUpdateRequest):
    """
    Update the approval status of a specific embodiment.

    Args:
        request (EmbodimentStatusUpdateRequest): Contains embodiment_id (UUID) and new status

    Returns:
        JSONResponse: Success or error response with updated embodiment data
    """
    try:
        # Update the embodiment status in Supabase using the unique ID
        response = (
            supabase.table("embodiments")
            .update({"status": request.status})
            .eq("id", request.embodiment_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            logging.info(f"Updated embodiment {request.embodiment_id} status to {request.status}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "success",
                    "message": f"Embodiment status updated to {request.status}",
                    "data": response.data[0]
                }
            )
        else:
            logging.warning(f"Embodiment {request.embodiment_id} not found")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "message": f"Embodiment with id={request.embodiment_id} not found"
                }
            )
    except Exception as e:
        logging.error(f"Error updating embodiment status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Failed to update embodiment status: {str(e)}"
            }
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


@app.post("/api/v1/documents/comprehensive-analysis/lancedb/{table_name}", tags=["Documents"])
async def comprehensive_analysis_lancedb(table_name: str):
    """
    Comprehensive analysis using existing LanceDB content (bypasses LLaMA Parse).
    Fast path - uses already processed content from LanceDB.
    """
    try:
        # Get database connection
        db = db_connection.get("db")
        if db is None:
            raise HTTPException(status_code=500, detail="LanceDB connection not initialized")
        
        # Check if table exists
        table_names = await lancedb_tables(db)
        if table_name not in table_names:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found in LanceDB")
        
        filename = f"{table_name}.pdf"  # Reconstruct filename
        
        # Initialize analysis service
        analysis_service = ComprehensiveAnalysisService()
        
        # Analyze from LanceDB content
        result = await analysis_service.analyze_from_lancedb(table_name, filename, db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/comprehensive-analysis/filename/{filename}", tags=["Documents"])
async def comprehensive_analysis_filename(filename: str):
    """
    Comprehensive analysis by filename (downloads from Supabase storage).
    Requires exact filename as it appears in storage.
    """
    try:
        # Download file content from Supabase storage
        file_content = supabase_download(filename)
        
        # Initialize analysis service
        analysis_service = ComprehensiveAnalysisService()
        
        # Perform analysis with LLaMA Parse
        result = await analysis_service.analyze_from_file_content(file_content, filename)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/comprehensive-analysis/storage-id/{file_id}", tags=["Documents"])
async def comprehensive_analysis_storage_id(file_id: str):
    """
    Comprehensive analysis by storage file ID (from /api/v1/documents/ response).
    Maps storage ID to filename, downloads from Supabase, analyzes with LLaMA Parse.
    """
    try:
        # Get filename from storage files list
        files = supabase_files()
        target_file = None
        
        for file in files:
            if file["id"] == file_id:
                target_file = file
                break
        
        if not target_file:
            raise HTTPException(status_code=404, detail=f"File with storage ID {file_id} not found")
        
        filename = target_file["name"]
        
        # Download file content from Supabase storage
        file_content = supabase_download(filename)
        
        # Initialize analysis service
        analysis_service = ComprehensiveAnalysisService()

        # Perform analysis with LLaMA Parse
        result = await analysis_service.analyze_from_file_content(file_content, filename)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Patent Content Draft Endpoints - AIP-1
@app.patch("/api/v1/project/patent/{patent_id}/component", response_model=ComponentUpdateResponse, tags=["Patent Draft"])
async def update_patent_component(patent_id: str, request: ComponentUpdateRequest):
    """
    Update or insert a single component in the patent draft.

    This endpoint is called each time a section is generated to incrementally
    build up the patent draft. It performs an upsert operation - if a component
    with the same ID exists, it updates it; otherwise, it appends a new component.

    Args:
        patent_id: Unique identifier for the patent project
        request: Component data including type, title, content, and metadata

    Returns:
        ComponentUpdateResponse: Status, updated component ID, and timestamp

    Raises:
        HTTPException: If there's an error updating the component
    """
    try:
        logging.info(f"PATCH /component: Starting component update for patent_id={patent_id}, component_id={request.component_id}, type={request.type}")

        # Validate patent_id format
        try:
            uuid.UUID(patent_id)
        except ValueError:
            logging.warning(f"PATCH /component: Invalid patent_id format: {patent_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid patent_id format. Expected UUID, got: {patent_id}"
            )

        # Fetch existing draft or prepare new one
        try:
            logging.debug(f"PATCH /component: Querying existing draft for patent_id={patent_id}")
            existing_draft = (
                supabase.table("patent_content_drafts")
                .select("*")
                .eq("patent_id", patent_id)
                .limit(1)
                .execute()
            )
            logging.debug(f"PATCH /component: Query response = {existing_draft}")
        except Exception as db_error:
            logging.error(f"PATCH /component: Supabase query error for patent_id={patent_id}: {db_error}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Database temporarily unavailable. Please try again."
            )

        # Prepare component data
        new_component = {
            "id": request.component_id,
            "type": request.type,
            "title": request.title,
            "content": request.content,
            "order": request.order,
            "generated_at": datetime.now().isoformat(),
        }

        if request.trace_id:
            new_component["trace_id"] = request.trace_id
        if request.metadata:
            new_component["metadata"] = request.metadata

        try:
            # Check if draft exists (data will be empty list if no record, or list with one dict if found)
            if existing_draft.data:
                logging.info(f"PATCH /component: Updating existing draft for patent_id={patent_id}")
                # Update existing draft - data[0] contains the record
                current_components = existing_draft.data[0].get("components", [])

                # Validate components is actually a list
                if not isinstance(current_components, list):
                    logging.error(f"PATCH /component: Invalid components type for patent_id={patent_id}: {type(current_components)}")
                    current_components = []

                # Find and update component if it exists, otherwise append
                component_updated = False
                for i, comp in enumerate(current_components):
                    if not isinstance(comp, dict):
                        continue
                    if comp.get("id") == request.component_id or comp.get("type") == request.type:
                        current_components[i] = new_component
                        component_updated = True
                        logging.debug(f"PATCH /component: Replaced existing component at index {i}")
                        break

                if not component_updated:
                    current_components.append(new_component)
                    logging.debug(f"PATCH /component: Appended new component. Total components: {len(current_components)}")

                # Update the draft
                logging.debug(f"PATCH /component: Executing UPDATE query for patent_id={patent_id}")
                response = (
                    supabase.table("patent_content_drafts")
                    .update({
                        "components": current_components,
                        "last_saved_at": datetime.now().isoformat()
                    })
                    .eq("patent_id", patent_id)
                    .execute()
                )

                # Verify update succeeded
                if not response.data:
                    logging.error(f"PATCH /component: UPDATE returned no data for patent_id={patent_id}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to update component. No data returned from database."
                    )
                logging.debug(f"PATCH /component: UPDATE response = {response}")
                logging.info(f"PATCH /component: Successfully updated component for patent_id={patent_id}")
            else:
                logging.info(f"PATCH /component: Creating new draft for patent_id={patent_id}")
                # Create new draft
                response = (
                    supabase.table("patent_content_drafts")
                    .insert({
                        "patent_id": patent_id,
                        "components": [new_component],
                        "version": 1,
                        "last_saved_at": datetime.now().isoformat()
                    })
                    .execute()
                )

                # Verify insert succeeded
                if not response.data:
                    logging.error(f"PATCH /component: INSERT returned no data for patent_id={patent_id}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to create draft. No data returned from database."
                    )
                logging.debug(f"PATCH /component: INSERT response = {response}")
                logging.info(f"PATCH /component: Successfully created new draft for patent_id={patent_id}")

        except HTTPException:
            raise
        except Exception as db_error:
            logging.error(f"PATCH /component: Supabase write error for patent_id={patent_id}: {db_error}", exc_info=True)
            # Check for common Supabase errors
            error_str = str(db_error).lower()
            if "unique" in error_str or "constraint" in error_str:
                logging.warning(f"PATCH /component: Constraint violation for patent_id={patent_id}")
                raise HTTPException(
                    status_code=409,
                    detail="Conflict: Draft already exists with different version"
                )
            elif "permission" in error_str or "policy" in error_str:
                logging.warning(f"PATCH /component: Permission denied for patent_id={patent_id}")
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied: Cannot modify this draft"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Database write failed. Please try again."
                )

        return ComponentUpdateResponse(
            status="success",
            message="Component updated successfully",
            patent_id=patent_id,
            component_id=request.component_id,
            updated_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error updating patent component for patent_id={patent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error updating patent component: {str(e)}"
        )


@app.post("/api/v1/project/patent/{patent_id}/save", response_model=PatentDraftSaveResponse, tags=["Patent Draft"])
async def save_patent_draft(patent_id: str, request: PatentDraftSaveRequest):
    """
    Save complete patent draft state.

    This endpoint is called when the user clicks "Save Progress" button.
    It replaces the entire components array with the provided data,
    increments the version number, and updates the timestamp.

    Args:
        patent_id: Unique identifier for the patent project
        request: Complete list of draft components

    Returns:
        PatentDraftSaveResponse: Status, version, timestamp, and component count

    Raises:
        HTTPException: If there's an error saving the draft
    """
    try:
        logging.info(f"POST /save: Starting full draft save for patent_id={patent_id}, components_count={len(request.components)}")

        # Validate patent_id format
        try:
            uuid.UUID(patent_id)
        except ValueError:
            logging.warning(f"POST /save: Invalid patent_id format: {patent_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid patent_id format. Expected UUID, got: {patent_id}"
            )

        # Validate request data
        if not isinstance(request.components, list):
            raise HTTPException(
                status_code=400,
                detail="Components must be a list"
            )

        # Validate each component has required fields
        for idx, component in enumerate(request.components):
            if not isinstance(component, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Component at index {idx} must be an object"
                )
            required_fields = ["id", "type", "title", "content", "order"]
            missing_fields = [f for f in required_fields if f not in component]
            if missing_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Component at index {idx} missing required fields: {missing_fields}"
                )

        # Check if draft exists to get current version
        try:
            logging.debug(f"POST /save: Querying existing version for patent_id={patent_id}")
            existing_draft = (
                supabase.table("patent_content_drafts")
                .select("version")
                .eq("patent_id", patent_id)
                .limit(1)
                .execute()
            )
            logging.debug(f"POST /save: Query response = {existing_draft}")
        except Exception as db_error:
            logging.error(f"POST /save: Supabase query error for patent_id={patent_id}: {db_error}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Database temporarily unavailable. Please try again."
            )

        # Calculate new version (data will be empty list if no record exists)
        new_version = (existing_draft.data[0].get("version", 0) + 1) if existing_draft.data else 1
        logging.info(f"POST /save: Calculated new version={new_version} for patent_id={patent_id}")

        # Upsert the complete draft
        try:
            logging.debug(f"POST /save: Executing UPSERT for patent_id={patent_id}")
            response = (
                supabase.table("patent_content_drafts")
                .upsert({
                    "patent_id": patent_id,
                    "components": request.components,
                    "version": new_version,
                    "last_saved_at": datetime.now().isoformat()
                }, on_conflict="patent_id")
                .execute()
            )

            # Verify upsert succeeded
            if not response.data:
                logging.error(f"POST /save: UPSERT returned no data for patent_id={patent_id}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to save draft. No data returned from database."
                )

            logging.debug(f"POST /save: UPSERT response = {response}")
            logging.info(f"POST /save: Successfully saved draft for patent_id={patent_id}, version={new_version}")

        except HTTPException:
            raise
        except Exception as db_error:
            logging.error(f"POST /save: Supabase upsert error for patent_id={patent_id}: {db_error}", exc_info=True)
            error_str = str(db_error).lower()
            if "unique" in error_str or "constraint" in error_str:
                logging.warning(f"POST /save: Constraint violation for patent_id={patent_id}")
                raise HTTPException(
                    status_code=409,
                    detail="Conflict: Failed to save draft due to concurrent modification"
                )
            elif "permission" in error_str or "policy" in error_str:
                logging.warning(f"POST /save: Permission denied for patent_id={patent_id}")
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied: Cannot save this draft"
                )
            elif "size" in error_str or "too large" in error_str:
                logging.warning(f"POST /save: Content too large for patent_id={patent_id}")
                raise HTTPException(
                    status_code=413,
                    detail="Draft content too large. Please reduce component sizes."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Database write failed. Please try again."
                )

        return PatentDraftSaveResponse(
            status="success",
            message="Draft saved successfully",
            patent_id=patent_id,
            version=new_version,
            last_saved_at=datetime.now(),
            components_count=len(request.components)
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error saving patent draft for patent_id={patent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error saving patent draft: {str(e)}"
        )


@app.get("/api/v1/project/patent/{patent_id}", response_model=PatentDraftResponse, tags=["Patent Draft"])
async def get_patent_draft(patent_id: str):
    """
    Retrieve saved patent draft for a given patent project.

    This endpoint fetches the complete draft state including all components,
    version number, and timestamps.

    Args:
        patent_id: Unique identifier for the patent project

    Returns:
        PatentDraftResponse: Complete draft data with components and metadata

    Raises:
        HTTPException: 404 if draft doesn't exist, 500 for other errors
    """
    try:
        logging.info(f"GET /patent: Retrieving draft for patent_id={patent_id}")

        # Validate patent_id format
        try:
            uuid.UUID(patent_id)
        except ValueError:
            logging.warning(f"GET /patent: Invalid patent_id format: {patent_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid patent_id format. Expected UUID, got: {patent_id}"
            )

        # Query draft from database
        try:
            logging.debug(f"GET /patent: Querying draft for patent_id={patent_id}")
            response = (
                supabase.table("patent_content_drafts")
                .select("*")
                .eq("patent_id", patent_id)
                .limit(1)
                .execute()
            )
            logging.debug(f"GET /patent: Query response = {response}")
        except Exception as db_error:
            logging.error(f"GET /patent: Supabase query error for patent_id={patent_id}: {db_error}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Database temporarily unavailable. Please try again."
            )

        # Check if draft exists (data will be empty list if no record found)
        if not response.data:
            logging.info(f"GET /patent: No draft found for patent_id={patent_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No draft found for patent_id={patent_id}"
            )

        # Extract the record from the list
        draft_data = response.data[0]
        logging.debug(f"GET /patent: Retrieved draft with {len(draft_data.get('components', []))} components")

        # Validate data integrity
        components = draft_data.get("components", [])
        if not isinstance(components, list):
            logging.error(f"GET /patent: Invalid components type for patent_id={patent_id}: {type(components)}")
            components = []

        # Ensure timestamps are present
        last_saved_at = draft_data.get("last_saved_at")
        created_at = draft_data.get("created_at")

        if not last_saved_at:
            logging.warning(f"GET /patent: Missing last_saved_at for patent_id={patent_id}")
        if not created_at:
            logging.warning(f"GET /patent: Missing created_at for patent_id={patent_id}")

        logging.info(f"GET /patent: Successfully retrieved draft for patent_id={patent_id}, version={draft_data.get('version', 'unknown')}")

        return PatentDraftResponse(
            status="success",
            patent_id=patent_id,
            components=components,
            version=draft_data.get("version", 1),
            last_saved_at=last_saved_at,
            created_at=created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error retrieving patent draft for patent_id={patent_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving patent draft: {str(e)}"
        )