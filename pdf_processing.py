import io
import os
import logging
import time
import asyncio
import aiofiles
import json
import pandas as pd
import instructor
import unstructured_client

from unstructured_client.models import shared
from unstructured_client.models.operations import PartitionRequest
from supabase import create_client
from models.pdf_workflow import FileProcessedError
# from models.metadata_extraction import Extraction
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from lancedb.db import AsyncConnection


class Extraction(BaseModel):
    method: list[str] = Field(
        default_factory=list, 
        description="method or methods observed/applied in the document. E.g., gene editing, CRISPR, PCR, fermentation, etc")
    hypothetical_questions: list[str] = Field(
        default_factory=list,
        description="Hypothetical questions that this document could answer",
    ) 
    keywords: list[str] = Field(
        default_factory=list, description="Biology related keywords that this document is about or MeSH headings for p apers"
    ) 


openai = instructor.from_openai(OpenAI())
asyncopenai = instructor.from_openai(AsyncOpenAI())

unstructured = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY")
)


unstructured = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

supabase = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_SECRET_KEY"),
)


async def partition_request(filename: str, content: bytes) -> PartitionRequest:
  

    return PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=content,
                file_name=filename,
            ),
            combine_under_n_chars=120,
            chunking_strategy="by_page",
            strategy=shared.Strategy.FAST,
            languages=["eng"],
            split_pdf_page=True,
            split_pdf_allow_failed=True,
            split_pdf_concurrency_level=15,
            max_characters=1000,
            overlap=500,
        ),
    )

async def lancedb_tables(db: AsyncConnection) -> list[str]:
    return await db.table_names()

def supabase_files() -> list[dict[str, str]]:
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME")
    return supabase.storage.from_(bucket_name).list("files")

def supabase_upload(file: bytes, filename: str, partition: bool):
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME")
    
    folder = "partitions" if partition else "files"

    files = supabase.storage.from_(bucket_name).list(folder)

    logging.info(f"files request returned: {[file['name'] for file in files]}")

    if filename in [file["name"] for file in files]:
        logging.info(f"file {filename} already exists in storage")
        return

    try:
        logging.info(f"uploading file to path {folder}/{filename}")
        res = supabase.storage.from_(bucket_name).upload(
            file=file,
            path=f"{folder}/{filename}",
            file_options={"cache-control": "3600", "upsert": "false"},
        )
        logging.info(f"uploaded file to path {res.path}")
    except Exception as e:
        logging.error(f"supabase_upload error during processing: {e}")

async def extract_metadata(text: str, chunk_id: str) -> dict:
    """
    Asynchronously extract metadata (keywords, methods, hypothetical questions)
    from a given text chunk by wrapping the blocking API call.
    """
    try:
        model = "gpt-4o-mini"
        messages = [
            {
                "role": "system",
                "content": "Your role is to extract data from the following document."
            },
            {"role": "user", "content": text},
        ]
        # Wrap the blocking API call in asyncio.to_thread
        extraction = await asyncio.to_thread(
            openai.chat.completions.create,
            model=model,
            response_model=Extraction,
            messages=messages
        )
        extraction_for_debug = dict(extraction)
        extraction_for_debug["chunk_id"] = chunk_id
        
        # logging.info("Raw metadata with chunk id: %s", pprint.pformat(extraction_for_debug))
        return extraction.model_dump() # return the original extraction 
    except Exception as e:
        logging.info(f"extract_metadata error extracting metadata: {e}")
        # Return empty metadata in case of error
        return {"keywords": [], "method": [], "hypothetical_questions": []}

async def process_file(content: bytes, filename: str, db: AsyncConnection) -> dict | FileProcessedError:
    """
    Asynchronously process a PDF file by partitioning its text into chunks,
    extracting metadata concurrently for each chunk, and saving the results
    to both JSON and CSV formats.
    
    :param str = file_path: 
    :param str = filename: 
    :return dict | FileProcessedError: 
    """
    
    
    # Create a table name by removing the .pdf extension
    table_name = filename.replace(".pdf", "")
    if table_name in await db.table_names():
        logging.info("File already exists in database, skipping processing")
        return FileProcessedError(is_processed=True, error="File already processed.")



    logging.info(f"Processing file: {filename}")

    req = await partition_request(filename, content)
    
    try:
        start_time = time.perf_counter()
        # Run the blocking partitioning API call in a separate thread
        res = await asyncio.to_thread(unstructured.general.partition, request=req)
        element_dicts = [element for element in res.elements]
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Partitioning completed in {elapsed_time:.2f} seconds")

        # Asynchronously upload the raw JSON data to supabase
        json_filename = table_name + ".json"
        supabase_upload(json.dumps(element_dicts).encode('utf-8'), json_filename, partition=True)

        # Build rows for our DataFrame
        data = []
        for chunk_counter, element in enumerate(element_dicts, start=1):
            row = {
                "element_id": element.get("element_id"),
                "text": element.get("text"),
                "page_number": element.get("metadata", {}).get("page_number"),
                "filename": element.get("metadata", {}).get("filename"),
                "chunk_id": chunk_counter,
            }
            data.append(row)

        # concurrently extract metadata for each text chunk
        tasks = [extract_metadata(row["text"], row["chunk_id"]) for row in data]
        metadata_results = await asyncio.gather(*tasks)

        # update rows with the extracted metadata
        for row, metadata in zip(data, metadata_results):
            
            metadata_for_chunks = json.loads(json.dumps(metadata))
            
            meta_lines = []
            if metadata_for_chunks.get("keywords"):
                meta_lines.append(f"Keywords: {', '.join(metadata_for_chunks['keywords'])}")
            if metadata_for_chunks.get("method"):
                meta_lines.append(f"Methods: {', '.join(metadata_for_chunks['method'])}")
            if metadata_for_chunks.get("hypothetical_questions"):
                meta_lines.append(f"Hypothetical Questions: {', '.join(metadata_for_chunks['hypothetical_questions'])}")

            # if we actually have metadata to add, append it to the text
            if meta_lines:
                metadata_str = "\n\n--- Extracted Metadata ---\n" + "\n".join(meta_lines)
                # append the metadata string to the original text
                row["text"] = row["text"] + metadata_str
            

        # Create a Pandas DataFrame from the processed data
        df = pd.DataFrame(data)

        total_time = time.perf_counter() - start_time
        logging.info(f"process_file completed successfully in {total_time:.2f} seconds")
        return {"filename": filename, "data": df}

    except Exception as e:
        logging.error(f"process_file error during processing: {e}")
        return FileProcessedError(is_processed=False, error=str(e))