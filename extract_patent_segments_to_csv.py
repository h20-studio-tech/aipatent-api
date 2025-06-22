import os
import asyncio
import csv
import sys
import traceback
from src.utils import ocr

PDF_PATH = r"C:\Users\vtorr\OneDrive\Desktop\Work\Projects\aipatent-api\experiments\sample_patents\ALD_GvHD provisional patent.pdf"
CSV_PATH = "ALD_GvHD_patent_segments.csv"

def extract_segments_to_csv(pdf_path, csv_path):
    try:
        print("Starting extraction...")
        # Load PDF as bytes
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}", file=sys.stderr)
            return

        print(f"Loading PDF: {pdf_path}")
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # Step 1: Load pages
        print("Step 1: Loading pages...")
        pages_tuple = ocr.pdf_pages(pdf_data, os.path.basename(pdf_path))
        
        # Step 2: Process pages
        print("Step 2: Processing pages...")
        processed_pages = ocr.process_pdf_pages(pages_tuple)
        print(f"Processed {len(processed_pages)} pages.")
        
        # Step 3: Segment pages (async)
        print("Step 3: Segmenting pages...")
        segmented_pages = asyncio.run(ocr.segment_pages(processed_pages))
        
        # Step 4: Write to CSV
        print(f"Step 4: Writing {len(segmented_pages)} pages to CSV...")
        with open(csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "page_number", "section", "text"])
            for page in segmented_pages:
                writer.writerow([
                    getattr(page, 'filename', ''),
                    getattr(page, 'page_number', ''),
                    getattr(page, 'section', ''),
                    getattr(page, 'text', '').replace('\r', '').replace('\n', ' ')
                ])
        print(f"Extraction complete. CSV saved to: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    extract_segments_to_csv(PDF_PATH, CSV_PATH)

