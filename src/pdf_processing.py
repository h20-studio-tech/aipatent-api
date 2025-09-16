import os
import logging
import time
import asyncio
import json
import pandas as pd
import instructor
from llama_cloud_services import LlamaParse
from supabase import create_client
from src.models.pdf_workflow import FileProcessedError
# from models.metadata_extraction import Extraction
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from lancedb.db import AsyncConnection
from dotenv import load_dotenv


load_dotenv('.env')

e_reasoning = os.getenv("e_reasoning")
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

llama_parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en"
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


async def parse_pdf_with_llama(filename: str, content: bytes) -> list:
    """Parse PDF using LlamaParse and return structured data."""
    import tempfile
    import os

    # Save content to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Parse with LlamaParse
        result = await llama_parser.aparse(temp_file_path)

        # Convert to element-like structure with proper chunking
        elements = []
        chunk_size = 1000
        overlap = 200

        for page_num, page in enumerate(result.pages, start=1):
            page_text = page.text or page.md or ""
            if not page_text.strip():
                continue

            # Split page into chunks with overlap
            chunks = chunk_text_with_overlap(page_text, chunk_size, overlap)

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                if len(chunk_text.strip()) > 50:  # Skip very small chunks
                    elements.append({
                        "element_id": f"page_{page_num}_chunk_{chunk_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_num,
                            "filename": filename,
                            "chunk_index": chunk_idx
                        }
                    })

        return elements

    finally:
        # Clean up temp file
        os.unlink(temp_file_path)


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break on sentence or paragraph boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > start + chunk_size * 0.5:  # Don't break too early
                chunk = text[start:break_point + 1]
                end = break_point + 1

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks

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

def supabase_delete(filename: str, partition: bool = False) -> dict:
    """
    Delete a file from Supabase storage.
    """
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME")
    folder = "partitions" if partition else "files"
    path = f"{folder}/{filename}"
    try:
        res = supabase.storage.from_(bucket_name).remove([path])
        logging.info(f"Deleted file {path} from Supabase storage.")
        return res
    except Exception as e:
        logging.error(f"Error deleting file {path} from Supabase storage: {e}")
        raise

def supabase_download(filename: str) -> bytes:
    """
    Download a file from Supabase storage.
    """
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME")
    path = f"files/{filename}"
    try:
        res = supabase.storage.from_(bucket_name).download(path)
        logging.info(f"Downloaded file {path} from Supabase storage.")
        return res
    except Exception as e:
        logging.error(f"Error downloading file {path} from Supabase storage: {e}")
        raise

async def extract_metadata(text: str, chunk_id: str) -> dict:
    """
    Asynchronously extract metadata (keywords, methods, hypothetical questions)
    from a given text chunk by wrapping the blocking API call.
    """
    try:
        model = "gpt-5-nano-2025-08-07",
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
            messages=messages,
            reasoning_effort=e_reasoning
        )
        extraction_for_debug = dict(extraction)
        extraction_for_debug["chunk_id"] = chunk_id
        
        # logging.info("Raw metadata with chunk id: %s", pprint.pformat(extraction_for_debug))
        return extraction.model_dump() # return the original extraction 
    except Exception as e:
        logging.info(f"extract_metadata error extracting metadata: {e}")
        # Return empty metadata in case of error
        return {"keywords": [], "method": [], "hypothetical_questions": []}

async def process_file(content: bytes, filename: str, db: AsyncConnection, force_reprocess: bool = False) -> dict | FileProcessedError:
    """
    Asynchronously process a PDF file by partitioning its text into chunks,
    extracting metadata concurrently for each chunk, and saving the results
    to both JSON and CSV formats.
    
    :param bytes content: The file content to process
    :param str filename: The name of the file
    :param AsyncConnection db: The LanceDB connection
    :param bool force_reprocess: Whether to force reprocessing even if the file exists
    :return dict | FileProcessedError: Processing result or error
    """
    
    try:
        # Create a table name by removing the .pdf extension
        table_name = filename.replace(".pdf", "")
        if not force_reprocess:
            try:
                table_names = await db.table_names()
                if table_name in table_names:
                    logging.info("File already exists in database, skipping processing")
                    return FileProcessedError(is_processed=True, error="File already processed.")
            except Exception as e:
                logging.error(f"Error checking if table exists: {str(e)}")
                return FileProcessedError(is_processed=False, error="Error checking if file exists in database", original_error=e)


        logging.info(f"Processing file: {filename}")
        
        try:
            start_time = time.perf_counter()
            # Parse PDF with LlamaParse
            element_dicts = await parse_pdf_with_llama(filename, content)
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"LlamaParse processing completed in {elapsed_time:.2f} seconds")

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
            return FileProcessedError(is_processed=False, error=f"Error processing file: {str(e)}", original_error=e)
    except Exception as outer_e:
        logging.error(f"Unexpected error in process_file: {outer_e}")
        return FileProcessedError(is_processed=False, error=f"Unexpected error: {str(outer_e)}", original_error=outer_e)