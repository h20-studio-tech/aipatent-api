#!/usr/bin/env python3
"""
Check the current schema of the embodiments table in Supabase.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import json

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SECRET_KEY')
supabase = create_client(url, key)

print("🔍 Checking Supabase embodiments table schema...\n")

try:
    # Get a sample embodiment to see the structure
    result = supabase.table("embodiments").select("*").limit(1).execute()
    
    if result.data and len(result.data) > 0:
        print("✅ Successfully connected to Supabase\n")
        print("📊 Sample embodiment record:")
        print(json.dumps(result.data[0], indent=2))
        
        print("\n🔑 Current fields in embodiments table:")
        for key in result.data[0].keys():
            print(f"  - {key}: {type(result.data[0][key]).__name__}")
    else:
        print("⚠️  No embodiments found in the table")
        
    # Try to get table information using a raw SQL query
    print("\n📋 Attempting to get table schema via SQL...")
    
    # Note: This might not work depending on Supabase permissions
    try:
        schema_query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'embodiments'
        ORDER BY ordinal_position;
        """
        # Supabase client doesn't support raw SQL directly through the Python client
        # We'll need to use the REST API or database connection
        print("ℹ️  Direct SQL queries require database connection or REST API")
    except Exception as e:
        print(f"❌ Could not execute raw SQL: {e}")
        
except Exception as e:
    print(f"❌ Error connecting to Supabase: {e}")
    print("\nPlease check:")
    print("  1. SUPABASE_URL is correct")
    print("  2. SUPABASE_SECRET_KEY is valid")
    print("  3. The embodiments table exists")

print("\n💡 Based on the code analysis, the embodiments table should have:")
print("  - file_id (FK to patent_files)")
print("  - emb_number (integer)")
print("  - text (text)")
print("  - header (text, optional)")
print("  - page_number (integer)")
print("  - section (text)")
print("  - summary (text)")
print("  - sub_category (text, optional)")
print("\n🎯 We need to add:")
print("  - start_char (integer)")
print("  - end_char (integer)")