import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic import ValidationError
from src.models.ocr_schemas import ProcessedPage, HeaderDetectionPage
from src.utils.ocr import detect_description_headers

# Define the path to the test data
# Use direct paths that work better with forward slashes in Python
TEST_PATENT_FILE = Path("C:/Users/vtorr/Work/Projects/aipatent-api/experiments/partitions/ald_gvhd_provisional_patent_processed.json")
RESULTS_FILE = Path("C:/Users/vtorr/Work/Projects/aipatent-api/header_detection_results.json")

# Define the target section name (case-insensitive comparison will be used)
TARGET_SECTION_NAME = "Detailed Description" 

def load_patent_data(file_path: Path) -> List[Dict[str, Any]]:
    """Loads patent data from a JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        # Assuming the JSON file contains a list of page data objects
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Test data file {file_path} is not a JSON list as expected.")
    return data

async def run_header_detection_and_save_results():
    """Run header detection on the Detailed Description section and save results to JSON."""
    print(f"Loading patent data from {TEST_PATENT_FILE}...")
    raw_pages_data = load_patent_data(TEST_PATENT_FILE)

    processed_pages_for_test: List[ProcessedPage] = [] 
    for i, page_data_from_json in enumerate(raw_pages_data):
        try:
            # Transform page_data_from_json to match ProcessedPage model structure
            transformed_page_dict = {
                "text": page_data_from_json.get("text"),
                "filename": page_data_from_json.get("filename"),
                "page_number": page_data_from_json.get("page_number"),
                "section": page_data_from_json.get("section"),
                "image": page_data_from_json.get("image"),
            }

            json_section_name = page_data_from_json.get("section")
            if json_section_name:
                page_obj = ProcessedPage(**transformed_page_dict) # Instantiate with transformed data

                # Case-insensitive comparison for filtering
                if (
                    page_obj.section and 
                    page_obj.section.strip().lower() == TARGET_SECTION_NAME.strip().lower()
                ):
                    if page_obj.image:
                        processed_pages_for_test.append(page_obj)
                        print(f"Including page {page_obj.page_number} from {page_obj.filename}")
                    else:
                        print(f"Skipping page {page_obj.page_number} (idx {i}) from {page_obj.filename}: No image for section '{TARGET_SECTION_NAME}'.")
                else:
                    section_name_for_log = page_obj.section if page_obj.section else "None"
                    print(f"Skipping page {page_obj.page_number} (idx {i}): section '{section_name_for_log}' != '{TARGET_SECTION_NAME}'")

        except ValidationError as e:
            print(f"Warning: Skipping page data at index {i} ({page_data_from_json.get('filename')}, page {page_data_from_json.get('page_number')}) due to Pydantic validation error: {e}")
        except Exception as e: # Catch other potential errors during transformation
            print(f"Warning: Skipping page data at index {i} ({page_data_from_json.get('filename')}, page {page_data_from_json.get('page_number')}) due to error: {e}")

    if not processed_pages_for_test:
        print(f"No suitable pages (with images and section '{TARGET_SECTION_NAME}') found in {TEST_PATENT_FILE.name}.")
        return

    print(f"Running header detection on {len(processed_pages_for_test)} filtered pages.")
    
    try:
        results: List[HeaderDetectionPage] = await detect_description_headers(processed_pages_for_test)
        
        # Create a serializable summary of the results
        json_results = []
        detected_count = 0
        
        for result in results:
            if result.has_header:
                detected_count += 1
                
            # Create a JSON object for each page
            page_result = {
                "filename": result.filename,
                "page_number": result.page_number,
                "has_header": result.has_header,
                "header": result.header,
                "section": result.section,
                "text": result.text[:200] + "..." if result.text and len(result.text) > 200 else result.text
            }
            json_results.append(page_result)
        
        # Save the JSON results to a file
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to {RESULTS_FILE}")
        print(f"{detected_count} headers detected out of {len(results)} pages.")
        
    except ValidationError as e:
        print(f"ValidationError in detect_description_headers: {e}")
        raise
    except Exception as e:
        print(f"Error during header detection: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_header_detection_and_save_results())
