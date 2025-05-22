from openai import AsyncOpenAI
import instructor 
from pydantic import BaseModel
from langfuse.decorators import observe
import os
import asyncio
from uuid import uuid4
from supabase import create_client
from typing import Optional
from src.utils.langfuse_client import get_langfuse_instance

class SyntheticEmbodiment(BaseModel):
    content: str
    
# Initialize external clients once at import time
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))
instructor_client = instructor.from_openai(AsyncOpenAI())

# Helper -----------------------------------------------------------------
async def fetch_source_embodiments(file_id: str, window_size: int = 20) -> list[str]:
    """Return the *window_size* most recent embodiments for this patent.
    Ordered chronologically (oldest → newest).
    """
    def _query():
        return (
            supabase_client.table("embodiments")
            .select("text, page_number")
            .eq("file_id", file_id)
            .order("emb_number", desc=True)
            .limit(window_size)
            .execute()
        )

    res = await asyncio.to_thread(_query)
    rows = getattr(res, "data", None) or (res.get("data") if isinstance(res, dict) else [])
    # Supabase returns newest-first because of desc=True – reverse so oldest first
    return [row["text"] for row in reversed(rows)]

@observe(name='embodiment_generation')
async def generate_embodiment(file_id: str,
                              inspiration: float,
                              knowledge: str,   
                              patent_title: str = "", 
                              disease: str = "", 
                              antigen: str = "") -> dict:
    source_embodiments = await fetch_source_embodiments(file_id)

    langfuse = get_langfuse_instance()
    prompt = langfuse.get_prompt("generate_embodiment").compile(
        inspiration=inspiration,
        knowledge=knowledge,
        source_embodiments=source_embodiments,
        patent_title=patent_title,
        disease=disease,
        antigen=antigen
    )
    
    response = await instructor_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3",
        response_model=SyntheticEmbodiment
    )
    
    langfuse.trace(
        id=uuid4(),
        name="generate_embodiment",
        session_id=file_id,
        input={
            "inspiration": inspiration,
            "knowledge": knowledge,
            "source_embodiments": source_embodiments,
            "patent_title": patent_title,
            "disease": disease,
            "antigen": antigen
        },
        output=response
    )
    return response


# # Simple test/demo
# if __name__ == "__main__":
#     import asyncio

#     async def test_generate():
#         # Use dummy values; adjust as needed for your schema
#         result = await generate_embodiment(
#             inspiration=0.8,
#             knowledge="Test knowledge",
#             source_embodiments=None,
#             patent_title="Test Patent",
#             disease="Test Disease",
#             antigen="Test Antigen"
#         )
#         print("Generated Embodiment:", result)

#     asyncio.run(test_generate())
