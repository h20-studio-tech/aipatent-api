import os
import logging
import uuid
import instructor
import asyncio
import pandas as pd
try:
    from langfuse.decorators import observe
except Exception:  # pragma: no cover - fallback for test envs
    from src.utils.langfuse_stub import observe
from src.models.rag_schemas import Chunk
from pydantic import BaseModel
from typing import List, Optional
from src.utils.langfuse_client import get_langfuse_instance
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from lancedb.embeddings import get_registry
from lancedb.db import AsyncConnection
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv('.env')


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

class JudgeVerdict(BaseModel):
    passed: bool
    feedback: Optional[str] = None


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
        logging.warning("Warning: The DataFrame is empty, no data will be added.")
        return

    try:
        # Check if db is properly initialized
        if db is None:
            logging.error("LanceDB connection is None. Cannot create table.")
            raise ValueError("LanceDB connection is None")
            
        # Log the connection details
        logging.info(f"Using LanceDB connection: {db}")
        
        # Check available tables before creation
        existing_tables = await db.table_names()
        logging.info(f"Existing tables before operation: {existing_tables}")
        
        # Check if table already exists
        if table_name in existing_tables:
            logging.info(f"Table {table_name} already exists, dropping it first")
            await db.drop_table(table_name)
            
        # Log data sample for debugging
        logging.info(f"Data sample (first row): {data.iloc[0].to_dict() if len(data) > 0 else None}")
        
        # Create the table with more detailed error handling
        logging.info(f"Creating table {table_name} with schema {schema}")
        table = await db.create_table(table_name, schema=schema, exist_ok=True, data=data)
        
        # Verify table creation
        table_rows = await table.count_rows()
        logging.info(f"Table {table_name} successfully created with {table_rows} rows")
        
        # Verify table appears in table list after creation
        updated_tables = await db.table_names()
        logging.info(f"Tables after creation: {updated_tables}")
        if table_name not in updated_tables:
            logging.warning(f"Table {table_name} does not appear in table list after creation!")
    
    except Exception as e:
        logging.error(f"Error creating table {table_name}: {str(e)}")
        logging.exception("Detailed exception information:")
        raise  # Re-raise to propagate the error


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
    query: str,
    table_names: List[str],
    n_queries: str = 3,
    db: AsyncConnection = None,
    feedback: Optional[str] = None,
) -> List[Chunk]:

    prompt = f"""
            You are a query understanding system for an AI Patent Generation application. Your task is to transform the user query and expand it into `{n_queries}` different queries
            in order to maximize retrieval efficiency.

            Generate `{n_queries}` questions based on `{query}`. The questions should be focused on expanding the search of information from a microbiology paper.

            Stylistically the queries should be optimized for matching text chunks in a vector DB, to enhance the likelihood of effectively retrieving the relevant chunks
            that contain the answer to the original user query.
            """

    # If we are retrying due to a failed judge, bias the expansions using the feedback
    if feedback:
        prompt += f"""

            Additional guidance for regeneration:
            The previous answer did not pass the judge. Feedback was: "{feedback}".
            Please bias the expanded queries to cover the gaps highlighted by the feedback while staying faithful to the original intent of the user query.
            """
    try:
        logging.info(f"Generating MultiQuery questions")
        multiquery = openai.chat.completions.create(
            model="gpt-5",
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
         model="gpt-5",
         messages=[
           {"role": "system", "content": "You are a great scientific analyst who is extensively knowledgeable in microbiologics and patent applications."},
           {"role": "assistant", "content": f"reference data to answer questions {format_chunks(chunks)}"},
           {"role": "user", "content": f"""Provide an answer to the question: {prompt}

Use the above document segments as your reference. When you use information from a specific chunk, add an inline citation using the format [X] where X is the chunk ID number.

For example: "IgY antibodies have been shown to reduce bacterial adhesion [57] and improve growth performance in livestock [36]."

Make sure to cite the specific chunk_id(s) that support each claim or piece of information in your response."""}
         ],
     ).choices[0].message.content


@observe(name="judge_answer")
async def judge_answer(question: str, context: List[Chunk], answer: str, label: str = "production") -> JudgeVerdict:
    """Run the judge prompt to evaluate the answer. Returns a structured verdict.

    Falls back to passing the answer if the judge encounters an error to avoid blocking the flow.
    """
    try:
        prompt_obj = langfuse.get_prompt("research_judge", label=label)
        context_text = format_chunks(context)
        compiled = prompt_obj.compile(question=question, context=context_text, answer=answer)

        verdict = openai.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o-mini"),
            response_model=JudgeVerdict,
            messages=[
                {"role": "user", "content": compiled},
            ],
        )
        return verdict
    except Exception as e:
        logging.error(f"Judge step error: {e}")
        # Fail-open: if judge fails, do not block returning an answer
        return JudgeVerdict(passed=True, feedback=None)


@observe(name="regenerate_with_feedback")
async def regenerate_with_feedback(chunks: List[Chunk], question: str, feedback: str) -> str:
    """Regenerate the answer using feedback from the judge, conditioning on the same context."""
    return client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a great scientific analyst who is extensively knowledgeable in microbiologics and patent applications."},
            {"role": "assistant", "content": f"reference data to answer questions {format_chunks(chunks)}"},
            {"role": "user", "content": f"""Question: {question}
Your previous answer did not pass the judge. Feedback: {feedback}

Please provide a single improved answer strictly using the above document segments as reference.

IMPORTANT: When you use information from a specific chunk, add an inline citation using the format [X] where X is the chunk ID number.

For example: "IgY antibodies have been shown to reduce bacterial adhesion [57] and improve growth performance in livestock [36]."

Make sure to cite the specific chunk_id(s) that support each claim or piece of information in your response."""},
        ],
    ).choices[0].message.content
