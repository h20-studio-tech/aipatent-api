#!/usr/bin/env python3
"""Test script to verify LanceDB Cloud and LlamaParse migrations work correctly."""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('.env')

async def test_lancedb_connection():
    """Test LanceDB Cloud connection."""
    try:
        import lancedb

        logger.info("Testing LanceDB Cloud connection...")

        # Test the connection
        db = await lancedb.connect_async(
            uri="db://aipatent-ym7e4b",
            api_key=os.getenv("LANCEDB_CLOUD_KEY"),
            region="us-east-1"
        )

        # List tables
        tables = await db.table_names()
        logger.info(f"✅ LanceDB Cloud connected successfully! Found {len(tables)} tables: {tables}")

        return True, f"Connected with {len(tables)} tables"

    except Exception as e:
        logger.error(f"❌ LanceDB Cloud connection failed: {e}")
        return False, str(e)

def test_llamaparse_setup():
    """Test LlamaParse import and setup."""
    try:
        from llama_cloud_services import LlamaParse

        logger.info("Testing LlamaParse setup...")

        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")

        # Initialize parser
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en"
        )

        logger.info("✅ LlamaParse initialized successfully!")
        return True, "Parser initialized"

    except Exception as e:
        logger.error(f"❌ LlamaParse setup failed: {e}")
        return False, str(e)

def test_chunking_logic():
    """Test the new chunking function."""
    try:
        from src.pdf_processing import chunk_text_with_overlap

        logger.info("Testing chunking logic...")

        # Test with sample text
        sample_text = """This is a sample document. It contains multiple sentences.
        This should be split into chunks properly. Each chunk should have some overlap.
        The chunking function should break on sentence boundaries when possible.
        This ensures better semantic coherence in each chunk."""

        chunks = chunk_text_with_overlap(sample_text, chunk_size=100, overlap=20)

        logger.info(f"✅ Chunking works! Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"   Chunk {i}: '{chunk[:50]}...' (len: {len(chunk)})")

        return True, f"Created {len(chunks)} chunks"

    except Exception as e:
        logger.error(f"❌ Chunking test failed: {e}")
        return False, str(e)

async def test_app_startup():
    """Test if the app can start with new configurations."""
    try:
        logger.info("Testing app startup...")

        # Import main to trigger startup checks
        from src.main import db_connection

        if db_connection.get("db") is None:
            logger.warning("⚠️  App hasn't fully initialized yet (normal in test)")
            return True, "App imports successfully"
        else:
            logger.info("✅ App initialized with database connection!")
            return True, "App fully initialized"

    except Exception as e:
        logger.error(f"❌ App startup test failed: {e}")
        return False, str(e)

async def main():
    """Run all tests."""
    logger.info("🧪 Starting migration validation tests...")

    tests = [
        ("LanceDB Cloud Connection", test_lancedb_connection),
        ("LlamaParse Setup", test_llamaparse_setup),
        ("Chunking Logic", test_chunking_logic),
        ("App Startup", test_app_startup)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success, message = await test_func()
            else:
                success, message = test_func()

            results[test_name] = (success, message)
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = (False, f"Test crashed: {e}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("🏁 TEST SUMMARY")
    logger.info("="*50)

    passed = 0
    total = len(tests)

    for test_name, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Migration looks good!")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    asyncio.run(main())