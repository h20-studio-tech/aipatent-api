import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

oai = OpenAI() # vanilla openai
async_oai = AsyncOpenAI()

gemini = OpenAI( # no need to explain this 
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

async_gemini = AsyncOpenAI( # no need to explain this 
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)