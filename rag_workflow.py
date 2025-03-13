import os
import pandas as pd
import unstructured_client
import lancedb
import json
import uuid
import instructor
import asyncio
import logging
import time
import aiofiles 
import pprint

from unstructured_client.models import operations, shared
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import List, Tuple, Union
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, SecretStr
from utils.langfuse_client import get_langfuse_instance
from models.workflow import FileProcessedError
from models.metadata_extraction import Extraction
from concurrent.futures import ThreadPoolExecutor


langfuse = get_langfuse_instance()
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    
class MultiQueryQuestions(BaseModel):
    questions: List[str]
    
    
class Chunk(BaseModel):
    element_id: str
    text: str
    page_number: int
    filename: str
    chunk_id: int
    
class ChunkReview(BaseModel):
    relevant: bool
    
class ReviewedChunk(BaseModel):
    element_id: str
    text: str
    page_number: int
    filename: str
    chunk_id: int
    relevant: bool
    
class RagWorkflow:
    def __init__(self):
        self.client = unstructured_client.UnstructuredClient(
            api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
            server_url=os.getenv("UNSTRUCTURED_API_URL"),
        )

        self.func = get_registry().get("openai").create(name="text-embedding-3-large")
        self.uri = "app/data/lancedb"
        self.db = lancedb.connect(self.uri)
        self.table_name = "document"
        self.asyncopenai = instructor.from_openai(AsyncOpenAI())
        self.openai = instructor.from_openai(OpenAI())
        self.semaphore = asyncio.Semaphore(20)

        class Schema(LanceModel):
            text: str = self.func.SourceField()
            vector: Vector(self.func.ndims()) = self.func.VectorField()  # type: ignore
            element_id: str
            page_number: int
            filename: str
            chunk_id: int
            # keywords : str
            # hypothetical_questions: str
            # method: str
            

        self.schema = Schema
        
    async def limited_chunk_review(self, chunk):
        async with self.semaphore:
            return await self.chunk_review(chunk)
    
    async def process_all_chunks(self, chunks):
        tasks = [self.limited_chunk_review(chunk) for chunk in chunks]
        reviewed_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any task failed and log the error
        for i, result in enumerate(reviewed_chunks):
            if isinstance(result, Exception):
                logging.info(f"Task {i} failed with exception: {result}")
            else:
                logging.info(f"Task {i} completed successfully")
        
        
            # Filter out any exceptions (failed tasks) before proceeding
        successful_chunks = [rc for rc in reviewed_chunks if not isinstance(rc, Exception)]
    
        return successful_chunks    
        
    async def chunk_review(self, chunk: Chunk)-> ReviewedChunk:
            prompt  = f"""
                You are a data annotator, your task is to review a corpus of data consisting of chunks from a source document.
                Your perfomance is critical as it ensures that bad or irrelevant data doesn't flood the vector database where
                the chunks will be stored
                
                here's the chunk text: <Text>{chunk.text}</Text>
                
                <Guidelines>
                    What makes text irrelevant:
                    - the text is too short and doesn't develop any idea or argument
                    - contains only numbers
                    - text that appears to be a caption for an image
                    - text that appears to only contain people's names
                    - text that is out of context
                    - as a rule of thumb text smaller than 80 tokens is not relevant
                    
                    What makes text relevant: 
                    - it does present findings, ideas, arguments,
                    - it describes something and contains semantic meaning, valuable ideas about a topic can infered from it
                </Guidelines>
            """
            
            res = await self.openai.chat.completions.create(
                model="o3-mini",
                response_model=ChunkReview,
                messages=[
                    {"role": "assistant", "content": prompt}
                ]
            )
            logging.info("API Response:", res)
            return ReviewedChunk(**chunk.model_dump(), relevant=res.relevant)    
        
    async def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['chunk_id'] = range(1, len(df) + 1)
        chunks: List[BaseModel] = []
        for _,row in df.iterrows():
            chunk = Chunk(
                element_id=row["element_id"],
                text=row["text"],
                page_number=row["page_number"],
                filename=row["filename"],
                chunk_id=row["chunk_id"]    
            )
            
            chunks.append(chunk)
            
        reviewed_chunks = await self.process_all_chunks(chunks)
        reviewed_dicts = [rc.model_dump() for rc in reviewed_chunks]
        reviewed_df = pd.DataFrame(reviewed_dicts)
        clean_df = reviewed_df[reviewed_df["relevant"] != False]
        return clean_df
                
    async def extract_metadata(self, text: str, chunk_id: str) -> dict:
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
                self.openai.chat.completions.create,
                model=model,
                response_model=Extraction,
                messages=messages
            )
            extraction_for_debug = dict(extraction)
            extraction_for_debug["chunk_id"] = chunk_id
            
            # logging.info("Raw metadata with chunk id: %s", pprint.pformat(extraction_for_debug))
            return extraction.model_dump() # return the original extraction 
        except Exception as e:
            logging.info(f"Error extracting metadata: {e}")
            # Return empty metadata in case of error
            return {"keywords": [], "method": [], "hypothetical_questions": []}
        
    async def aprocess_file(self, file_path: str, filename: str) -> dict | FileProcessedError:
        """
        Asynchronously process a PDF file by partitioning its text into chunks,
        extracting metadata concurrently for each chunk, and saving the results
        to both JSON and CSV formats.
        
        :param str = file_path: 
        :param str = filename: 
        :return dict | FileProcessedError: 
        """
        # Create a table name by removing the .pdf extension
        self.table_name = filename.replace(".pdf", "")
        if self.table_name in self.db.table_names():
            logging.info("File already exists in database, skipping processing")
            return FileProcessedError(is_processed=True, error="File already processed.")

        if not os.path.exists(file_path):
            logging.info("The file does not exist")
            return

        logging.info(f"Processing file: {file_path}")

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=open(file_path, "rb"),
                    file_name=filename,
                ),
                combine_under_n_chars=120,
                chunking_strategy=shared.ChunkingStrategy.BY_PAGE,
                strategy=shared.Strategy.FAST,
                languages=["eng"],
                split_pdf_page=True,
                split_pdf_allow_failed=True,
                split_pdf_concurrency_level=15,
                max_characters=1000,
                overlap=500
            ),
        )

        try:
            start_time = time.perf_counter()
            # Run the blocking partitioning API call in a separate thread
            res = await asyncio.to_thread(self.client.general.partition, request=req)
            element_dicts = [element for element in res.elements]
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"Partitioning completed in {elapsed_time:.2f} seconds")

            # Asynchronously write the raw JSON data
            json_path = file_path.replace(".pdf", ".json")
            async with aiofiles.open(json_path, "w") as jf:
                await jf.write(json.dumps(element_dicts))

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
            tasks = [self.extract_metadata(row["text"], row["chunk_id"]) for row in data]
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
            df.to_csv("experiments/unstructured/results/result.csv")
            csv_path = file_path.replace(".pdf", ".csv")
            loop = asyncio.get_running_loop()
            # Write CSV using ThreadPoolExecutor to avoid blocking the event loop
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, lambda: df.to_csv(csv_path, index=False))

            total_time = time.perf_counter() - start_time
            logging.info(f"process_file completed successfully in {total_time:.2f} seconds")
            return {"df": df, "filepath": csv_path}

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            return FileProcessedError(is_processed=False, error=str(e))
                
    def process_file(self, file_path: str, filename: str) -> dict | FileProcessedError:
        """
        Synchronously process a PDF file by partitioning its text into chunks,
        and saving the results to both JSON and CSV formats.
        
        **important:** This function doesn't incorporate metadata extraction due to latency considerations.
        for detailed and concurrent metadata extraction use **aprocess_file**
        
        :param str  file_path: 
        :param str  filename:        
        
        
        """
        self.table_name = filename.replace(".pdf", "")
        if self.table_name in self.db.table_names():
            logging.info("File already exists in database, skipping processing")
            return FileProcessedError(is_processed=True)

        if not os.path.exists(file_path):
            logging.info("The file does not exist")
            return
        logging.info(f"Processing file: {file_path}")
        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=open(file_path, "rb"),
                    file_name=filename,
                ),
                combine_under_n_chars=120,
                chunking_strategy=shared.ChunkingStrategy.BY_PAGE,
                strategy=shared.Strategy.FAST,
                languages=["eng"],
                split_pdf_page=True,
                split_pdf_allow_failed=True,
                split_pdf_concurrency_level=15,
                max_characters=1000,
                overlap=500
            ),
        )

        try:
            data = []
            chunk_counter = 0
            start_time = time.perf_counter()
            res = self.client.general.partition(request=req)
            element_dicts = [element for element in res.elements]
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"partition completed with time: {elapsed_time}")
            with open(file_path.replace(".pdf", ".json"), "w") as f:
                json.dump(element_dicts, f)

            for element_dict in element_dicts:
                chunk_counter += 1  # increment only when we add a row

                new_row = {
                    "element_id": element_dict["element_id"],
                    "text": element_dict["text"],
                    "page_number": element_dict["metadata"].get("page_number"),
                    "filename": element_dict["metadata"].get("filename"),
                    "chunk_id": chunk_counter,
                }
                data.append(new_row)

            df = pd.DataFrame(data=data)
            # df = await self.clean_df(df)
            df.to_csv(file_path.replace(".pdf", ".csv"), index=False)
            
            end_time = time.perf_counter() - start_time
            
            logging.info(f"process_file completed successfully with time: {end_time}")
            return {"df": df, "filepath": file_path}

        except Exception as e:
            logging.info(e)
    
    FileTuple = Tuple[str, str]

    async def process_files(self, file_list: List[FileTuple]) -> List[Union[dict, FileProcessedError]]:
        """
        Process a list of files by partitioning each document into chunks.
        For each file, the partitioned data is saved as JSON and CSV.
        Returns a list of results per file processed.
 
        :param file_list: List of tuples (file_path, filename)
        :return: List of dictionaries with keys "df" and "filepath", or a FileProcessedError if processing failed.
        """
        results = []
        
        for file_path, filename in file_list:
            if not os.path.exists(file_path):
                logging.info(f"The file does not exist: {filename}")
                results.append(FileProcessedError(is_processed=False))
                continue

            logging.info(f"Processing file: {filename}")
            
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(file_path, "rb"),
                        file_name=filename,
                    ),
                    combine_under_n_chars=120,
                    chunking_strategy=shared.ChunkingStrategy.BY_PAGE,
                    strategy=shared.Strategy.FAST,
                    languages=["eng"],
                    split_pdf_page=True,
                    split_pdf_allow_failed=True,
                    split_pdf_concurrency_level=15,
                    max_characters=1000,
                    overlap=500
                ),
            )
            try:
                res = await asyncio.to_thread(self.client.general.partition, request=req)
                
                # Extract partitioned elements
                element_dicts = [element for element in res.elements]
                
                # Save the raw partitioned data as JSON for record-keeping
                json_path = file_path.replace(".pdf", ".json")
                async with aiofiles.open(json_path, "w") as jf:
                    await jf.write(json.dumps(element_dicts))
                
                # Process each partition element into a row for the DataFrame
                data = []
                chunk_counter = 0
                for element_dict in element_dicts:
                    chunk_counter += 1  # Increment for every new row
                    new_row = {
                        "element_id": element_dict["element_id"],
                        "text": element_dict["text"],
                        "page_number": element_dict["metadata"].get("page_number"),
                        "filename": element_dict["metadata"].get("filename"),
                        "chunk_id": chunk_counter,
                    }
                    data.append(new_row)
                    
                logging.info(f"Processing file completed successfully for: {filename}")
                df = pd.DataFrame(data=data)
                csv_path = file_path.replace(".pdf", ".csv")
                
                loop = asyncio.get_running_loop()
                
                # Write CSV using ThreadPoolExecutor to avoid blocking the event loop
                with ThreadPoolExecutor() as pool:
                    await loop.run_in_executor(pool, lambda: df.to_csv(csv_path, index=False))
                
                results.append({"df": df, "filepath": file_path})
            except Exception as e:
                logging.info(f"Error processing file {file_path}: {e}")
                
        
        return results

    def create_table_from_file(self, file_path: str):

        if not os.path.exists(file_path):
            logging.info("The file does not exist")
            return

        self.file_path = file_path
        df = pd.read_csv(file_path.replace(".pdf", ".csv"))
        
        df = df[df["text"].str.strip() != ""]
        logging.info(f"checking rows with missing text: {df['text'].isnull().sum()} ")  # How many rows have missing text?
        logging.info(f"create_table_from_file: {file_path} of length: {df.shape[0]}")


        if df.empty:
            logging.info("Warning: The DataFrame is empty, no data will be added.")
            return

        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        
        self.table = self.db.create_table(self.table_name, schema=self.schema, exist_ok=True)
        self.add_df_to_table(df)
        
        table_rows = self.table.count_rows()
        logging.info(f"table {self.table_name} successfully created")
        logging.info(f"Entries added to the table: {table_rows}")
        
    def create_table_from_files(self, file_paths: List[str]) -> None:
        """
        Creates a single table in the database using CSV data generated
        from multiple PDF files. Each CSV is expected to be located at
        the same path as its corresponding PDF, but with a '.csv' extension.
        
        :param file_paths: List of file paths pointing to the PDF files.
        """
        
        aggregated_data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logging.info(f"The file does not exist: {file_path}")
                continue

            # Assume CSV file has the same path but with .csv extension
            csv_path = file_path.replace(".pdf", ".csv")
            if not os.path.exists(csv_path):
                logging.info(f"CSV file does not exist: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            # Remove rows with empty or whitespace-only text
            df = df[df["text"].str.strip() != ""]
            logging.info(f"Checking CSV {csv_path}: missing text rows: {df['text'].isnull().sum()}")
            logging.info(f"File {csv_path} has {df.shape[0]} rows after filtering.")
            
            if df.empty:
                logging.info(f"Warning: The DataFrame for {csv_path} is empty; skipping file.")
                continue
            
            aggregated_data.append(df)
        
        if not aggregated_data:
            logging.info("No valid data found in any of the provided files.")
            return

        # Concatenate all DataFrames into one
        aggregated_df = pd.concat(aggregated_data, ignore_index=True)
        logging.info(f"Aggregated DataFrame contains {aggregated_df.shape[0]} rows.")

        # If a table already exists, drop it to replace with new data
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            logging.info(f"Dropped existing table {self.table_name}.")

        # Create the table using the defined schema
        self.table = self.db.create_table(self.table_name, schema=self.schema, exist_ok=True)
        self.add_df_to_table(aggregated_df)
        
        table_rows = self.table.count_rows()
        logging.info(f"Table {self.table_name} successfully created.")
        logging.info(f"Entries added to the table: {table_rows}")    
        
    def set_table_name(self, filename: str):
        table_name = filename.replace(".pdf", "") # get the filename without the extension
        if table_name in self.db.table_names():
            logging.info(f"setting table: {table_name}")
            self.table_name = table_name
        
    def add_df_to_table(self, df: pd.DataFrame):
        df = df[df["text"].str.strip() != ""]
    
        # for col in ["keywords", "hypothetical_questions", "method"]:
        #     missing_count = df[col].isnull().sum()
        #     logging.info(f"Filling {missing_count} missing values in column '{col}' with empty strings.")
        #     df[col] = df[col].fillna("")        
        
        
        print(f"checking rows with missing text: {df['text'].isnull().sum()} ")
        print(f"add_df_to_table: {df.shape[0]}")

        if df.empty:
            print("Warning: The DataFrame is empty, no data will be added.")
            return
        
        self.table.add(df)

    def format_chunks(self, chunks: List[LanceModel]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"===== Chunk {i} =====")
            lines.append(chunk.text.strip())
            lines.append(f"page number: {chunk.page_number}")
            lines.append(f"filename: {chunk.filename}")
            lines.append(f"chunk ID: {chunk.chunk_id}")
            lines.append("")  # blank line between docs
        logging.info("formatted chunks")
        return "\n".join(lines)

    def search(self, query: str, k_results: int = 4):
        table = self.db.open_table(self.table_name)
        chunks = table.search(query).limit(k_results).to_pydantic(self.schema)
        return chunks

    def formatted_search(self, query: str, k_results: int = 4) -> str:
        chunks = self.search(query, k_results)
        return self.format_chunks(chunks)

    
    def multiquery_search(self, query:str, n_queries: str = 3) -> str:
                
        prompt = f"""
            You are a query understanding system for an AI Patent Generation application your task is to transform the user query and expand it into `{n_queries}` different queries
            in order to maximize retrieval efficiency
            
            
            Generate `{n_queries}` questions based on `{query}`. The questions should be focused on expanding the search of information from a microbiology paper:


            Stylistically the queries should be optimized for matching text chunks in a vectordb, doing so enhances the likelihood of effectively retrieving the relevant chunks
            that contain the answer to the original user query.
            """
        try:
            logging.info(f"Generating MultiQuery questions")
            multiquery = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                response_model=MultiQueryQuestions,
                messages=[{"role": "user", "content": prompt}],
            )
            logging.info(f"MultiQuery questions: \n{chr(10).join(f'- {q}' for q in multiquery.questions)}\n")
            
            retrieved = [self.search(q) for q in multiquery.questions]
          
            chunks = [result for results in retrieved for result in results]
                
            logging.info(f"amount of retrieved chunks: {len(chunks)}")
            logging.info(f"retrieved chunk IDs:\n{[chunk.chunk_id for chunk in chunks]}\n")
            trace_id = str(uuid.uuid4())
            langfuse.trace(
                id=trace_id,
                name=f"multiquery questions",
                input=query,
                output=multiquery
            )
            formatted_chunks = self.format_chunks(chunks)
            return formatted_chunks
        except Exception as e:
            print(f"Error generating MultiQueryQuestions: {str(e)}")
            return []
    
    def delete_file(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        else:
            print("The file does not exist")

    def cleanup(self, delete_file: bool = True):
        self.db.drop_table(self.table_name)
        if delete_file:
            self.delete_file()
            




