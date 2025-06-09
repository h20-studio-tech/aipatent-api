import re
from io import BytesIO
from typing import Dict, Any, Optional

import pdfplumber
from PIL import Image
import pytesseract

from src.utils.logging_helper import create_logger

# Configure logging
logger = create_logger("abstract_extractor.py")


def find_abstract_header(page_text: str) -> Dict[str, Any]:
    """
    Find the ABSTRACT header in patent text and return information about its position
    
    Args:
        page_text: Extracted text of the page
        
    Returns:
        Dictionary with header information:
        {
            "found": bool,
            "start_index": int or None,
            "end_index": int or None,
            "confidence": float,
            "method": str,
            "header_text": str or None
        }
    """
    result = {
        "found": False,
        "start_index": None,
        "end_index": None,
        "confidence": 0.0,
        "method": None,
        "header_text": None
    }
    
    # Method 1: Exact ABSTRACT keyword matching (most reliable)
    # Look for standalone "ABSTRACT" in all caps with padding
    matches = list(re.finditer(r'(?:\n|\s|^)\s*ABSTRACT\s*(?:\n|$)', page_text))
    if matches:
        match = matches[0]  # Take the first match
        result["found"] = True
        result["start_index"] = match.start()
        result["end_index"] = match.end()
        result["confidence"] = 0.95
        result["method"] = "exact_match"
        result["header_text"] = match.group(0).strip()
        return result
    
    # Method 2: Look for variations with case insensitivity
    # This catches "Abstract" or "ABSTRACT" or other variations 
    matches = list(re.finditer(r'(?i)(?:\n|\s|^)\s*abstract(?:\s+(?:of|the|disclosure|invention))?\s*(?:\n|$)', page_text))
    if matches:
        match = matches[0]  # Take the first match
        result["found"] = True
        result["start_index"] = match.start()
        result["end_index"] = match.end()
        result["confidence"] = 0.85
        result["method"] = "case_insensitive_match"
        result["header_text"] = match.group(0).strip()
        return result
    
    # Method 3: Position-based heuristic
    # Check if we have anything that looks like a header in the top portion of the document
    lines = page_text.split('\n')
    top_portion = '\n'.join(lines[:min(10, len(lines))])  # First 10 lines or less
    
    # Look for capitalized standalone words that could be section headers
    header_candidates = re.finditer(r'(?:\n|\s|^)([A-Z]{5,})\s*(?:\n|$)', top_portion)
    for match in header_candidates:
        # If we find capitalized text that might be a header and contains 'ABST' 
        if 'ABST' in match.group(1):
            result["found"] = True
            result["start_index"] = match.start()
            result["end_index"] = match.end()
            result["confidence"] = 0.7
            result["method"] = "capitalized_keyword"
            result["header_text"] = match.group(1).strip()
            return result
            
    # Method 4: Structural analysis - look for common patent structure patterns
    # Abstract often appears after title and before background/description
    if re.search(r'(?i)field\s+of\s+(?:the\s+)?invention', page_text):
        field_pos = re.search(r'(?i)field\s+of\s+(?:the\s+)?invention', page_text).start()
        # Look for a potential header in the text before the field of invention
        text_before = page_text[:field_pos]
        # Try to find something that resembles an abstract section
        abst_candidates = re.finditer(r'(?:\n|\s|^)([A-Z][A-Za-z]{5,})\s*(?:\n|$)', text_before)
        for match in abst_candidates:
            if "ABST" in match.group(1).upper() or "SUMM" in match.group(1).upper():
                result["found"] = True
                result["start_index"] = match.start()
                result["end_index"] = match.end()
                result["confidence"] = 0.6
                result["method"] = "structural_inference"
                result["header_text"] = match.group(1).strip()
                return result
    
    return result


def extract_abstract_text(page_text: str, header_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the abstract text following the detected header
    
    Args:
        page_text: Full text of the page
        header_info: Header detection information from find_abstract_header
        
    Returns:
        Dictionary with abstract information:
        {
            "abstract_text": str or None,
            "confidence": float,
            "extraction_method": str
        }
    """
    if not header_info["found"]:
        return {
            "abstract_text": None,
            "confidence": 0.0,
            "extraction_method": "none"
        }
    
    # Abstract text starts after the header
    text_after_header = page_text[header_info["end_index"]:]
    
    # Method 1: Extract until the next section header or double newline
    # Common section headers that might follow the abstract
    next_section_patterns = [
        r'\n\s*BACKGROUND(?:\s+OF\s+THE\s+INVENTION)?\s*\n',
        r'\n\s*FIELD(?:\s+OF\s+THE\s+INVENTION)?\s*\n',
        r'\n\s*DESCRIPTION(?:\s+OF\s+(?:THE\s+)?(?:INVENTION|EMBODIMENTS))?\s*\n',
        r'\n\s*BRIEF DESCRIPTION\s*\n',
        r'\n\s*DETAILED DESCRIPTION\s*\n',
        r'\n\s*SUMMARY(?:\s+OF\s+THE\s+INVENTION)?\s*\n',
        r'\n\s*CLAIMS\s*\n',
        r'\n\s*DRAWINGS\s*\n',
        r'\n\s*FIGURE\s*\n',
        r'\n\s*FIG\.\s*\n'
    ]
    
    # Find the next section header
    min_pos = len(text_after_header)
    for pattern in next_section_patterns:
        match = re.search(pattern, text_after_header, re.IGNORECASE)
        if match and match.start() < min_pos:
            min_pos = match.start()
    
    # Also look for double newlines which often separate sections
    double_newline = re.search(r'\n\s*\n', text_after_header)
    if double_newline and double_newline.start() < min_pos:
        min_pos = double_newline.start()
    
    # Extract the abstract text
    abstract_text = text_after_header[:min_pos].strip()
    
    # If we get too much text, it's probably not just the abstract
    if len(abstract_text) > 1000:
        # Try to limit to 2-3 paragraphs
        paragraphs = re.split(r'\n\s*\n', abstract_text)
        if len(paragraphs) > 2:
            abstract_text = '\n\n'.join(paragraphs[:2]).strip()
    
    # Check if the abstract text is reasonable
    if len(abstract_text) < 20:
        # Too short to be an abstract
        return {
            "abstract_text": None,
            "confidence": 0.0,
            "extraction_method": "failed_extraction"
        }
    
    # Calculate confidence based on text characteristics
    confidence = min(0.9, header_info["confidence"])
    
    # Higher confidence if the abstract has a reasonable length
    if 50 <= len(abstract_text) <= 500:
        confidence += 0.1
    
    # Lower confidence for very long or short texts
    if len(abstract_text) > 800 or len(abstract_text) < 30:
        confidence -= 0.2
    
    return {
        "abstract_text": abstract_text,
        "confidence": min(1.0, confidence),  # Cap at 1.0
        "extraction_method": f"text_after_{header_info['method']}"
    }


def extract_abstract_from_page(page_text: str) -> Dict[str, Any]:
    """
    Complete abstract extraction process from page text
    
    Args:
        page_text: Full text of the page
        
    Returns:
        Dictionary with abstract extraction results:
        {
            "abstract_text": str or None,
            "abstract_status": str ("available", "not_found", "error", etc.),
            "abstract_confidence": float,
            "abstract_message": str
        }
    """
    try:
        # First, detect the abstract header
        header_info = find_abstract_header(page_text)
        
        if not header_info["found"]:
            return {
                "abstract_text": None,
                "abstract_status": "not_found",
                "abstract_confidence": 0.0,
                "abstract_message": "No abstract header detected in document"
            }
        
        # Extract the abstract text
        extraction_result = extract_abstract_text(page_text, header_info)
        
        if not extraction_result["abstract_text"]:
            return {
                "abstract_text": None,
                "abstract_status": "not_found",
                "abstract_confidence": 0.0,
                "abstract_message": "Abstract header found but could not extract text"
            }
        
        # Determine status based on confidence
        status = "available"
        if extraction_result["confidence"] < 0.6:
            status = "needs_manual"
            message = f"Low confidence extraction ({extraction_result['confidence']:.2f}), may need review"
        else:
            message = f"Abstract extracted successfully with confidence {extraction_result['confidence']:.2f}"
        
        return {
            "abstract_text": extraction_result["abstract_text"],
            "abstract_status": status,
            "abstract_confidence": extraction_result["confidence"],
            "abstract_message": message
        }
    
    except Exception as e:
        logger.error(f"Error extracting abstract: {str(e)}")
        return {
            "abstract_text": None,
            "abstract_status": "error",
            "abstract_confidence": 0.0,
            "abstract_message": f"Error during abstract extraction: {str(e)}"
        }


async def extract_abstract_from_pdf(pdf_data: bytes) -> Dict[str, Any]:
    """
    Extract abstract from a PDF patent document
    
    Args:
        pdf_data: Raw bytes of PDF file
        
    Returns:
        Dictionary with abstract extraction results
    """
    output = {
        'abstract_text': None,
        'abstract_status': 'not_found',
        'abstract_confidence': 0,
        'abstract_message': '',
        'abstract_page': None,
        'abstract_pattern': None,
    }
    try:
        with pdfplumber.open(BytesIO(pdf_data)) as pdf_file:
            if len(pdf_file.pages) == 0:
                return {
                    "abstract_text": None,
                    "abstract_status": "error",
                    "abstract_confidence": 0.0,
                    "abstract_message": "PDF has no pages"
                }
                
            total_pages = len(pdf_file.pages)
            logger.info(f"Patent document has {total_pages} pages")
            
            # Define the pages to check - more extensive for longer documents
            first_pages = range(min(10, total_pages))  # First 10 pages or fewer if document is shorter
            
            # For last pages, ensure we don't check the same pages twice
            if total_pages <= 20:
                # For short documents, we'll already check all pages in first_pages
                last_pages = []
            else:
                last_pages = range(max(10, total_pages - 10), total_pages)  # Last 10 pages
            
            # For very long documents, also check some middle pages that might contain abstract sections
            # This is especially important for provisional patents where abstract might be deep in the document
            middle_pages = []
            if total_pages > 40:
                # For very long documents, check pages at critical positions where abstracts often appear
                # Critical positions often include multiples of 10, especially pages 30, 40, 50
                critical_positions = [30, 40, 50, 60, 70]
                middle_pages = [p for p in critical_positions if p < total_pages and p not in first_pages and p not in last_pages]
                
            logger.info(f"Checking first pages: {list(first_pages)} and last pages: {list(last_pages)}")
            if middle_pages:
                logger.info(f"Also checking middle pages: {middle_pages}")
            
            # Look for a clear abstract header in all the target pages
            abstract_patterns = [
                r'(ABSTRACT|Abstract)\s*[:\r\n]+\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',  # Match until double newlines
                r'\n\s*(ABSTRACT|Abstract)\s*\n+\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',  # Match until double newlines
                r'(?:^|\n)\s*[\[\(]?\s*(ABSTRACT|Abstract)[\s\]\)\:]*[\n\s]*(.*?)(?:(?:\n\s*\n)|$)',
                r'(?:^|\n)\s*(?:ABSTRACT|Abstract)\s*(?:OF THE DISCLOSURE|OF THE INVENTION|OF INVENTION)?\s*[\:\n\r]\s*(.*?(?:\n(?!\s*\n)[^\n]*)*)',  
                r'(?i)\b(?:abstract|ABSTRACT)[\s\:\n\r]+(.*?)(?:\n\s*\n|$)',
                r'(?i)(?:^|\n)\s*(?:(?:\[\s*\]|\(\s*\))\s*)?(?:abstract|ABSTRACT)\s*(?:\[\s*\]|\(\s*\))?\s*[\:\n\r]?\s*(.*?)(?:\n\s*\n|$)'
            ]
            
            for page_num in list(first_pages) + list(middle_pages) + list(last_pages):
                page = pdf_file.pages[page_num]
                page_text = page.extract_text() or ""
                
                # If text extraction returns empty string, try OCR fallback
                if not page_text.strip():
                    logger.info(f"Standard text extraction failed on page {page_num+1}, trying OCR fallback")
                    try:
                        # We need to convert the PDF page to an image for OCR
                        # First render the page to an image
                        img = page.to_image(resolution=300)
                        
                        # Apply OCR directly to the image using pytesseract
                        ocr_text = pytesseract.image_to_string(img.original)
                        
                        if ocr_text and ocr_text.strip():
                            logger.info(f"OCR successfully extracted text from page {page_num+1}")
                            page_text = ocr_text
                        else:
                            logger.info(f"OCR extraction returned empty text for page {page_num+1}")
                            
                    except Exception as e:
                        logger.error(f"OCR fallback failed for page {page_num+1}: {e}")
                
                # Check for abstract header in text
                for pattern_index, pattern in enumerate(abstract_patterns):
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        logger.info(f"Found abstract header on page {page_num+1} with pattern {pattern_index+1}")
                        try:
                            # Extract just the abstract text
                            if len(match.groups()) > 1:
                                abstract_text = match.group(2)
                            else:
                                abstract_text = match.group(1)
                            
                            # Clean up common artifacts
                            # Remove line numbers that might appear at the beginning of lines
                            abstract_text = re.sub(r'^\s*\d+\s+', '', abstract_text, flags=re.MULTILINE)
                            
                            # Remove lines with dates, agent/applicant info, or signature lines
                            abstract_text = re.sub(r'\n.*?(?:dated|agent|applicant|signed|day\s+of).*?\n', '\n', 
                                                  abstract_text, flags=re.IGNORECASE)
                                                  
                            # More aggressive cleaning for common signature patterns
                            abstract_text = re.sub(r'(?:[\s\n]+[A-Z][\w\s,]+(?:PhD|MS|LLB|JD|Esq)[.,]*\s*(?:\([^)]+\))?)', '', 
                                                  abstract_text)
                            
                            # Remove any trailing numbers or digits that might be page numbers
                            abstract_text = re.sub(r'\s+\d+\s*$', '', abstract_text)
                            
                            # Remove any extra whitespace
                            abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
                            
                            output['abstract_text'] = abstract_text
                            output['abstract_status'] = 'available'
                            output['abstract_confidence'] = 0.95
                            output['abstract_message'] = f"Abstract extracted from page {page_num + 1} with explicit header"
                            output['abstract_page'] = page_num + 1
                            output['abstract_pattern'] = pattern_index + 1
                            return output
                        except Exception as e:
                            logger.error(f"Failed to extract abstract from page {page_num+1}: {e}")
            
            # If no explicit abstract found, we'll simplify and focus on 
            # Abstract not found in target pages, for certain files try scanning the entire document
            # Only do this for patents that should have abstracts (i.e., provisionals and large technical documents)
            logger.info("No abstract found in target pages, checking entire document as last resort")
            
            # Check if this is a provisional patent application based on number of pages
            # or if it's a very long document (>40 pages) which might have abstract buried inside
            is_long_document = total_pages > 40
            
            if is_long_document or total_pages > 10:
                logger.info("Document appears to be a long technical document, scanning all pages")
                
                # Scan every page not already checked
                pages_to_check = [p for p in range(total_pages) 
                                if p not in list(first_pages) and p not in list(last_pages) and p not in middle_pages]
                
                # Check pages in batches to avoid memory issues
                batch_size = 5
                for i in range(0, len(pages_to_check), batch_size):
                    batch = pages_to_check[i:i + batch_size]
                    logger.info(f"Checking additional pages: {batch}")
                    
                    for page_num in batch:
                        page = pdf_file.pages[page_num]
                        page_text = page.extract_text() or ""
                        
                        # If text extraction returns empty string, try OCR fallback
                        if not page_text.strip():
                            logger.info(f"Standard text extraction failed on page {page_num+1} during full scan, trying OCR fallback")
                            try:
                                # We need to convert the PDF page to an image for OCR
                                # First render the page to an image
                                img = page.to_image(resolution=300)
                                
                                # Apply OCR directly to the image using pytesseract
                                ocr_text = pytesseract.image_to_string(img.original)
                                
                                if ocr_text and ocr_text.strip():
                                    logger.info(f"OCR successfully extracted text from page {page_num+1} during full scan")
                                    page_text = ocr_text
                                else:
                                    logger.info(f"OCR extraction returned empty text for page {page_num+1} during full scan")
                                    
                            except Exception as e:
                                logger.error(f"OCR fallback failed for page {page_num+1} during full scan: {e}")
                        
                        # Check for abstract in text content 
                        for pattern_index, pattern in enumerate(abstract_patterns):
                            match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                            if match:
                                logger.info(f"Found abstract header during full document scan on page {page_num+1} with pattern {pattern_index+1}")
                                try:
                                    # Extract just the abstract text
                                    if len(match.groups()) > 1:
                                        abstract_text = match.group(2)
                                    else:
                                        abstract_text = match.group(1)
                                    
                                    # Clean the abstract text using the same cleaning function as before
                                    # Remove page numbers, dates, and signatures
                                    abstract_text = re.sub(r'\b(?:Page\s+\d+|\d+\s+of\s+\d+|\d{1,2}/\d{1,2}/\d{2,4})\b', '', abstract_text)
                                    abstract_text = re.sub(r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|PhD|MD|JD|LLB)\b[\s\S]*?(?=\n|$)', '', abstract_text)
                                    abstract_text = re.sub(r'\d+$', '', abstract_text)  # Remove trailing digits (often page numbers)
                                    
                                    # Remove lines that contain dates, agent info, or applicant info
                                    lines = [line for line in abstract_text.splitlines() 
                                            if not re.search(r'(?:Date|Agent|Attorney|Applicant|Application|Filing|Priority)', line)]
                                    abstract_text = '\n'.join(lines)
                                    
                                    # General cleanup of extra whitespace
                                    abstract_text = re.sub(r'\s{2,}', ' ', abstract_text)
                                    abstract_text = abstract_text.strip()
                                    
                                    output['abstract_text'] = abstract_text
                                    output['abstract_status'] = 'available'
                                    output['abstract_confidence'] = 0.85
                                    output['abstract_message'] = f"Abstract extracted from page {page_num+1} during full document scan"
                                    output['abstract_page'] = page_num + 1
                                    output['abstract_pattern'] = pattern_index + 1
                                    return output
                                except Exception as e:
                                    logger.error(f"Failed to extract abstract from page {page_num+1} during full scan: {e}")
            
            # Still no abstract found after extensive search
            logger.info("No explicit abstract header found in patent document after full search")
            return {
                "abstract_status": "not_found",
                "abstract_confidence": 0.0,
                "abstract_message": "This patent does not have a clearly labeled abstract section",
                "abstract_text": None
            }  
            
    except Exception as e:
        logger.error(f"Failed to extract abstract from PDF: {e}")
        return {
            "abstract_text": None,
            "abstract_status": "error",
            "abstract_confidence": 0.0,
            "abstract_message": f"Failed to extract abstract: {str(e)}"
        }


def extract_abstract_by_content_features(page_text: str) -> Optional[str]:
    """
    Extract abstract based on content features rather than explicit headers
    This is useful for patents where the abstract isn't clearly labeled
    
    Args:
        page_text: Text of a patent page
        
    Returns:
        Extracted abstract text or None
    """
    if not page_text.strip():
        return None
    
    # Case 1: Check for text after "DESCRIPTION OF THE INVENTION" section header
    # This is a common pattern in many patents, especially provisionals
    description_match = re.search(r'DESCRIPTION\s+OF\s+THE\s+INVENTION\s*[\r\n]+(.*?)(?:[\r\n]{2,}|DETAILED\s+DESCRIPTION|BRIEF\s+DESCRIPTION|BACKGROUND|SUMMARY)', 
                                page_text, re.IGNORECASE | re.DOTALL)
    if description_match:
        # Extract the first paragraph after the header
        description_text = description_match.group(1).strip()
        paragraphs = re.split(r'\n\s*\n', description_text)
        
        if paragraphs:
            # Get the first substantive paragraph (sometimes there are line numbers or blank lines)
            first_para = paragraphs[0].strip()
            
            # Skip very short text that's likely not a real paragraph
            if len(first_para) > 30:
                return first_para
    
    # Split into paragraphs for other methods
    paragraphs = re.split(r'\n\s*\n', page_text.strip())
    
    # Skip empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    if not paragraphs:
        return None
    
    # Case 2: Look for a paragraph that contains typical abstract indicators
    # Common sections that often contain abstract-like text
    abstract_indicators = [
        r'FIELD\s+OF\s+(THE\s+)?INVENTION', 
        r'BACKGROUND\s+OF\s+(THE\s+)?INVENTION',
        r'SUMMARY\s+OF\s+(THE\s+)?INVENTION',
        r'BRIEF\s+SUMMARY'
    ]
    
    # Check each paragraph to see if it follows one of these headers
    for i, para in enumerate(paragraphs):
        if i < len(paragraphs) - 1:  # Ensure there's a next paragraph
            if any(re.search(pattern, para, re.IGNORECASE) for pattern in abstract_indicators):
                next_para = paragraphs[i+1].strip()
                word_count = len(next_para.split())
                if 20 <= word_count <= 300:
                    return next_para
    
    # Case 3: If first paragraph appears to be a title/header and second looks like abstract
    if len(paragraphs) >= 2 and len(paragraphs[0].strip()) < 100:
        candidate = paragraphs[1]
        
        # Check if it has abstract-like characteristics
        word_count = len(candidate.split())
        if 20 <= word_count <= 300 and (
           "invention" in candidate.lower() or 
           "disclosure" in candidate.lower() or 
           "technology" in candidate.lower() or
           "method" in candidate.lower()):
            return candidate
    
    # Case 4: First paragraph if it has abstract-like features
    first_para = paragraphs[0]
    word_count = len(first_para.split())
    
    # Abstract-like characteristics
    common_phrases = [
        "the invention relates to", 
        "the present invention", 
        "disclosed is",
        "disclosed herein",
        "described is",
        "provided is",
        "the present disclosure"
    ]
    
    if 20 <= word_count <= 300 and (
       any(phrase in first_para.lower() for phrase in common_phrases) or
       ("invention" in first_para.lower() and 
        ("method" in first_para.lower() or "system" in first_para.lower()))):
        return first_para
    
    return None


# Example usage and testing (commented out)
# async def test_extraction(pdf_path):
#     with open(pdf_path, "rb") as f:
#         pdf_data = f.read()
#     
#     result = await extract_abstract_from_pdf(pdf_data)
#     print(json.dumps(result, indent=2))
