import langfuse
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pprint import pprint

load_dotenv('.env')
# Instantiate Langfuse once
langfuse = langfuse.Langfuse()

# p = langfuse.api.prompts.list()
# pprint([f'\n{pr}\n' for pr in p])
def get_langfuse_instance():
    return langfuse

def get_prompt(prompt_name: str, variables: Optional[Dict[str, Any]] = None):
    """
    Fetch a prompt from Langfuse and optionally compile it with variables.
    
    Args:
        prompt_name: Name of the prompt in Langfuse
        variables: Optional dictionary of variables to compile into the prompt
    
    Returns:
        If variables are provided, returns the compiled prompt string.
        If no variables are provided, returns the prompt object that can be compiled later.
    """
    # Get the langfuse client from the central instance
    client = get_langfuse_instance()
    
    # Fetch the prompt from Langfuse
    prompt = client.get_prompt(prompt_name)
    
    # If variables are provided, compile and return the prompt string
    if variables:
        return prompt.compile(**variables)
    
    # Otherwise return the prompt object for later compilation
    return prompt
