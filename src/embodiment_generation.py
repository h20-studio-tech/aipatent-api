from openai import OpenAI
from utils.langfuse_client import get_langfuse_instance
import instructor 
from pydantic import BaseModel
langfuse = get_langfuse_instance()

class SyntheticEmbodiment(BaseModel):
    content: str
    
instructor_client = instructor.from_openai(OpenAI())
async def generate_embodiment(inspiration: float, 
                              source_embodiment: str, 
                              patent_title, 
                              disease, 
                              antigen) -> dict:
    prompt = langfuse.get_prompt("generate_embodiment").compile(
        inspiration=inspiration,
        source_embodiment=source_embodiment,
        patent_title=patent_title,
        disease=disease,
        antigen=antigen
    )
    
    response = instructor_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3-mini",
        response_model=SyntheticEmbodiment
    )
    
    return response
    