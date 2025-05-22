import os
from functools import lru_cache
from supabase import create_client, Client

@lru_cache
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    return create_client(url, key)