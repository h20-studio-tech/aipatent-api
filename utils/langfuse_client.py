import langfuse
from dotenv import load_dotenv

load_dotenv('.env')
# Instantiate Langfuse once
langfuse = langfuse.Langfuse()

def get_langfuse_instance():
    return langfuse
