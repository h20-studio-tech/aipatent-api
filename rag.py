import os
import logging
import uuid
import instructor
import asyncio
import langfuse
import pandas as pd
from langfuse.decorators import observe
from models.rag_typing import Chunk
from pydantic import BaseModel
from typing import List
from utils.langfuse_client import get_langfuse_instance
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from lancedb.embeddings import get_registry
from lancedb.db import AsyncConnection
from openai import OpenAI, AsyncOpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

langfuse = get_langfuse_instance()
client = OpenAI()
openai = instructor.from_openai(OpenAI())
asyncopenai = instructor.from_openai(AsyncOpenAI())

func = get_registry().get("openai").create(name=os.getenv("EMBEDDING_MODEL_NAME"))


class Schema(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    element_id: str
    page_number: int
    filename: str
    chunk_id: int


class MultiQueryQuestions(BaseModel):
    questions: List[str]


def format_chunks(chunks: List[Chunk]) -> str:
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


def add_df_to_table(df: pd.DataFrame, table: Table):
    df = df[df["text"].str.strip() != ""]

    print(f"checking rows with missing text: {df['text'].isnull().sum()} ")
    print(f"add_df_to_table: {df.shape[0]}")

    if df.empty:
        print("Warning: The DataFrame is empty, no data will be added.")
        return

    table.add(df)


async def create_table_from_file(
    table_name: str, data: pd.DataFrame, db: AsyncConnection, schema: Schema = Schema, 
) -> None:

    logging.info(f"create_table_from_file: {table_name} of length: {data.shape[0]}")

    if data.empty:
        logging.info("Warning: The DataFrame is empty, no data will be added.")
        return

    if table_name in await db.table_names():
        await db.drop_table(table_name)

    table = await db.create_table(table_name, schema=schema, exist_ok=True, data=data)

    table_rows = await table.count_rows()
    logging.info(f"table {table_name} successfully created")
    logging.info(f"Entries added to the table: {table_rows}")


async def search(
    query: str,
    table_name: str,
    db: AsyncConnection,
    schema: Schema = Schema,
    k_results: int = 4,
) -> List[LanceModel]:
    try:
        res = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_NAME"), input=query
        )
        vector = res.data[0].embedding
        table = await db.open_table(table_name)
        # Await the search operation to get the actual search result object.

        search_result = (
            await table.vector_search(query_vector=vector).limit(k_results).to_list()
        )

        # Now chain the limit and to_pydantic calls on the resolved object.
        logging.info(f"search result {search_result}")
        return search_result
    except Exception as e:
        logging.info(f"Error during search: {str(e)}")
        return []


async def multiquery_search(
    query: str, table_names: List[str], n_queries: str = 3, db=AsyncConnection
) -> List[Chunk]:

    prompt = f"""
            You are a query understanding system for an AI Patent Generation application your task is to transform the user query and expand it into `{n_queries}` different queries
            in order to maximize retrieval efficiency
            
            
            Generate `{n_queries}` questions based on `{query}`. The questions should be focused on expanding the search of information from a microbiology paper:


            Stylistically the queries should be optimized for matching text chunks in a vectordb, doing so enhances the likelihood of effectively retrieving the relevant chunks
            that contain the answer to the original user query.
            """
    try:
        logging.info(f"Generating MultiQuery questions")
        multiquery = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_model=MultiQueryQuestions,
            messages=[{"role": "user", "content": prompt}],
        )
        logging.info(
            f"MultiQuery questions: \n{chr(10).join(f'- {q}' for q in multiquery.questions)}\n"
        )
        logging.info(f"Retrieving chunks from tables: {table_names}")

        retrieved = await asyncio.gather(
            *(search(q, table_name=table, db=db) for q in multiquery.questions for table in table_names)
        )

        chunks = [result for results in retrieved for result in results]

        logging.info(f"amount of retrieved chunks: {len(chunks)}")
        logging.info(
            f"retrieved chunk IDs:\n{[chunk['chunk_id'] for chunk in chunks]}\n"
        )
        trace_id = str(uuid.uuid4())
        langfuse.trace(
            id=trace_id, name=f"multiquery questions", input=query, output=multiquery
        )
        
        formatted_chunks = [Chunk(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            page_number=chunk["page_number"],
            filename=chunk["filename"],
        ) for chunk in chunks]
        logging.info(f"returning formatted chunks: {formatted_chunks}")
        return formatted_chunks
    except Exception as e:
        print(f"Error generating MultiQueryQuestions: {str(e)}")
        return []
    
@observe(name="multiquery_message")    
async def chunks_summary(chunks:List[Chunk], prompt: str):
    return client.chat.completions.create(
         model="gpt-4o-mini",
         messages=[
           {"role": "system", "content": "You are a great scientific analyst who is extensively knowledgeable in microbiologics and patent applications."},
           {"role": "assistant", "content": f"reference data to answer questions {format_chunks(chunks)}"},
           {"role": "user", "content": f"provide an answer to the question {prompt} using the above document segments as your reference"}
         ],
     ).choices[0].message.content
