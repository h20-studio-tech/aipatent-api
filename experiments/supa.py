<<<<<<< HEAD
import os
import logging
from supabase import create_client, Client

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SECRET_KEY")
supabase: Client = create_client(url, key)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open("experiments/unstructured/docs/gvhd_paper.json", "rb") as f:
    response = (
        supabase.storage
        .from_("aipatent-papers")
        .upload(
            file=f,
            path="data/gvhd_paper.json",
            file_options={"cache-control": "3600", "upsert": "false"}
        )
=======
import os
import logging
from supabase import create_client, Client

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SECRET_KEY")
supabase: Client = create_client(url, key)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open("experiments/unstructured/docs/gvhd_paper.json", "rb") as f:
    response = (
        supabase.storage
        .from_("aipatent-papers")
        .upload(
            file=f,
            path="data/gvhd_paper.json",
            file_options={"cache-control": "3600", "upsert": "false"}
        )
>>>>>>> 5b6e3e1f6bb904635df1f05e870b8aeeed94cf1b
    )