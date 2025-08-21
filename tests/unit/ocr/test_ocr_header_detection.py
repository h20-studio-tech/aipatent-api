import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic import ValidationError
from src.models.ocr_schemas import ProcessedPage, HeaderDetectionPage
from src.utils.ocr import detect_description_headers





# Define the path to the test data
BASE_DIR = Path(__file__).resolve().parent.parent # This should point to aipatent-api root
PARTITIONS_DIR = BASE_DIR / "experiments" / "partitions"
TEST_PATENT_FILE = PARTITIONS_DIR / "ald_gvhd_provisional_patent_processed.json"

# Define the target section name (case-sensitive)
TARGET_SECTION_NAME = "Detailed Description" 

def load_patent_data(file_path: Path) -> List[Dict[str, Any]]:
    """Loads patent data from a JSON file."""
    if not file_path.exists():
        pytest.fail(f"Test data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        # Assuming the JSON file contains a list of page data objects
        data = json.load(f)
    if not isinstance(data, list):
        pytest.fail(f"Test data file {file_path} is not a JSON list as expected.")
    return data

@pytest.mark.asyncio
async def test_detect_headers_on_detailed_description():
    """Test header detection on pages from the 'Detailed Description' section of a patent."""
    raw_pages_data = load_patent_data(TEST_PATENT_FILE)

    processed_pages_for_test: List[ProcessedPage] = [] # Uses test's local ProcessedPage model
    for i, page_data_from_json in enumerate(raw_pages_data):
        try:
            # Transform page_data_from_json to match the test's local ProcessedPage model structure
            transformed_page_dict = {
                "text": page_data_from_json.get("text"),         # Map 'text' to 'text_content'
                "filename": page_data_from_json.get("filename"),
                "page_number": page_data_from_json.get("page_number"),
                "section": page_data_from_json.get("section"),
                "image": page_data_from_json.get("image"),       # Map 'image' to 'image_base64'
            }

            json_section_name = page_data_from_json.get("section")
            if json_section_name:
                # Create the PatentSectionWithConfidence object as expected by test's ProcessedPage model
            
                page_obj = ProcessedPage(**transformed_page_dict) # Instantiate with transformed data

            # Original filtering logic based on the test's ProcessedPage model structure
            if (
    page_obj.section
    and page_obj.section.strip().lower() == TARGET_SECTION_NAME.strip().lower()
):
                if page_obj.image:
                    processed_pages_for_test.append(page_obj)
                else:
                    print(f"Skipping page {page_obj.page_number} (idx {i}) from {page_obj.filename}: No image for section '{TARGET_SECTION_NAME}'.")
            else:
                section_name_for_log = page_obj.section if page_obj.section else "None"
                print(f"Skipping page {page_obj.page_number} (idx {i}): section '{section_name_for_log}' ('{json_section_name}') != '{TARGET_SECTION_NAME}' or no image")

        except ValidationError as e:
            print(f"Warning: Skipping page data at index {i} ({page_data_from_json.get('filename')}, page {page_data_from_json.get('page_number')}) due to Pydantic validation error: {e}")
        except Exception as e: # Catch other potential errors during transformation
            print(f"Warning: Skipping page data at index {i} ({page_data_from_json.get('filename')}, page {page_data_from_json.get('page_number')}) due to error: {e}")

    if not processed_pages_for_test:
        pytest.skip(f"No suitable pages (with images and section '{TARGET_SECTION_NAME}') found in {TEST_PATENT_FILE.name}. Check TARGET_SECTION_NAME and data structure.")

    print(f"Running header detection on {len(processed_pages_for_test)} filtered pages.")
    
    # Ensure detect_description_headers is imported correctly
    # from src.utils.ocr import detect_description_headers
    try:
        results: List[HeaderDetectionPage] = await detect_description_headers(processed_pages_for_test)
    except ValidationError as e:
        print(f"ValidationError in detect_description_headers: {e}")
        raise

    assert len(results) == len(processed_pages_for_test), \
        f"Number of results ({len(results)}) should match number of input pages ({len(processed_pages_for_test)})"

    for i, result_page in enumerate(results):
        original_page = processed_pages_for_test[i]
        assert isinstance(result_page, HeaderDetectionPage), \
            f"Result item {i} is not a HeaderDetectionPage object"
        assert result_page.filename == original_page.filename, \
            f"Filename mismatch for page {i}"
        assert result_page.page_number == original_page.page_number, \
            f"Page number mismatch for page {i}"
        
        # Check the direct attributes of HeaderDetectionPage that detect_description_header populates
        assert isinstance(result_page.has_header, bool), \
            f"'has_header' should be a boolean for page {i}"
        if result_page.header is not None:
            assert isinstance(result_page.header, str), \
                f"'header' should be a string or None for page {i}"

    print(f"Successfully asserted header detection results for {len(results)} pages.")
    # Optional: Log some specific results for manual inspection if needed
    detected_count = sum(1 for r in results if r.has_header)
    print(f"{detected_count} headers detected out of {len(results)} pages.")
