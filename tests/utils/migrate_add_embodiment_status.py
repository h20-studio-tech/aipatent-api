#!/usr/bin/env python3
"""
Migration script to add 'status' field to embodiments table in Supabase.
This field will track whether an embodiment is 'pending', 'approved', or 'rejected'.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SECRET_KEY')

if not url or not key:
    logging.error("Missing SUPABASE_URL or SUPABASE_SECRET_KEY environment variables")
    sys.exit(1)

supabase = create_client(url, key)

def add_status_field():
    """
    Add status field to embodiments table if it doesn't exist.
    Note: This uses Supabase's SQL editor functionality through their API.
    """

    logging.info("Starting migration to add 'status' field to embodiments table...")

    # First, let's check if the field already exists by fetching a record
    try:
        result = supabase.table("embodiments").select("*").limit(1).execute()

        if result.data and len(result.data) > 0:
            sample_record = result.data[0]

            if 'status' in sample_record:
                logging.info("✅ 'status' field already exists in embodiments table")
                return True
            else:
                logging.info("'status' field not found in embodiments table")
                logging.info("Current fields: " + ", ".join(sample_record.keys()))
        else:
            logging.warning("No records found in embodiments table")

    except Exception as e:
        logging.error(f"Error checking embodiments table: {e}")
        return False

    # Since we can't directly alter the table through the Python client,
    # we'll provide the SQL that needs to be run
    sql_command = """
    -- Add status column to embodiments table
    ALTER TABLE embodiments
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending'
    CHECK (status IN ('pending', 'approved', 'rejected'));

    -- Create an index on status for faster filtering
    CREATE INDEX IF NOT EXISTS idx_embodiments_status
    ON embodiments(status);

    -- Update all existing records to have 'pending' status
    UPDATE embodiments
    SET status = 'pending'
    WHERE status IS NULL;
    """

    logging.info("\n" + "="*60)
    logging.info("⚠️  MANUAL STEP REQUIRED")
    logging.info("="*60)
    logging.info("The Supabase Python client doesn't support direct DDL operations.")
    logging.info("Please run the following SQL in your Supabase SQL editor:")
    logging.info("")
    logging.info("1. Go to your Supabase dashboard")
    logging.info("2. Navigate to SQL Editor")
    logging.info("3. Run this SQL command:")
    logging.info("-"*60)
    print(sql_command)
    logging.info("-"*60)
    logging.info("")
    logging.info("After running the SQL, all embodiments will have status='pending' by default")
    logging.info("="*60)

    return True

def update_existing_records():
    """
    Update all existing embodiment records to have status='pending'
    This function can be run after the schema is updated.
    """
    try:
        # First check if status field exists
        result = supabase.table("embodiments").select("*").limit(1).execute()

        if result.data and len(result.data) > 0:
            if 'status' not in result.data[0]:
                logging.warning("Status field doesn't exist yet. Run the SQL migration first.")
                return False

        # Get all records without a status
        all_embodiments = supabase.table("embodiments").select("file_id, emb_number").execute()

        if all_embodiments.data:
            logging.info(f"Found {len(all_embodiments.data)} embodiments to update")

            # Update each record
            # Note: Batch updates would be more efficient but Supabase Python client
            # doesn't support them directly
            updated_count = 0
            for emb in all_embodiments.data:
                try:
                    supabase.table("embodiments").update(
                        {"status": "pending"}
                    ).eq("file_id", emb["file_id"]).eq("emb_number", emb["emb_number"]).execute()
                    updated_count += 1
                except Exception as e:
                    logging.error(f"Failed to update embodiment {emb['file_id']}-{emb['emb_number']}: {e}")

            logging.info(f"✅ Updated {updated_count} embodiments with status='pending'")
            return True
        else:
            logging.info("No embodiments found to update")
            return True

    except Exception as e:
        logging.error(f"Error updating existing records: {e}")
        return False

if __name__ == "__main__":
    logging.info("Embodiment Status Migration Script")
    logging.info("="*40)

    # Step 1: Check and provide migration SQL
    add_status_field()

    # Step 2: Offer to update existing records
    logging.info("")
    response = input("Have you run the SQL migration? (y/n): ")

    if response.lower() == 'y':
        logging.info("Attempting to update existing records...")
        if update_existing_records():
            logging.info("✅ Migration completed successfully!")
        else:
            logging.info("❌ Failed to update existing records")
    else:
        logging.info("Please run the SQL migration first, then run this script again.")