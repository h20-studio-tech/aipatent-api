from openai import OpenAI
from src.utils.langfuse_client import get_langfuse_instance
import instructor 
from pydantic import BaseModel


class SyntheticEmbodiment(BaseModel):
    content: str
    
instructor_client = instructor.from_openai(OpenAI())
async def generate_embodiment(inspiration: float,
                              knowledge: str,  
                              source_embodiment: str, 
                              patent_title: str, 
                              disease: str, 
                              antigen: str) -> dict:
    langfuse = get_langfuse_instance()
    prompt = langfuse.get_prompt("generate_embodiment").compile(
        inspiration=inspiration,
        knowledge=knowledge,
        source_embodiment=source_embodiment,
        patent_title=patent_title,
        disease=disease,
        antigen=antigen
    )
    
    response = instructor_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3",
        response_model=SyntheticEmbodiment
    )
    
    return response
    