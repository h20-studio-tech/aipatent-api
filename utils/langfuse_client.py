import langfuse

# Instantiate Langfuse once
langfuse = langfuse.Langfuse()

def get_langfuse_instance():
    return langfuse
