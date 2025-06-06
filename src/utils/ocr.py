import instructor
import asyncio
import base64
import re
from io import BytesIO
from pdfplumber.page import Page
from openai import AsyncOpenAI
from src.utils.logging_helper import create_logger
from src.utils.langfuse_client import get_prompt
from typing import Union
from time import time

import pdfplumber
import pytesseract
from dotenv import load_dotenv
from langfuse.decorators import observe
from PIL import Image

from src.models.ocr_schemas import (
    DetailedDescriptionEmbodiment,
    Embodiment,
    Embodiments,
    HeaderDetectionPage,
    HeaderDetection,
    ProcessedPage,
    GlossaryDefinition,
    PatentSectionWithConfidence,
    CategoryResponse,
    EmbodimentSummary,
    EmbodimentSpellCheck,
    Glossary,
    GlossaryPageFlag
    )

# Configure logging
logger = create_logger("ocr.py")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai = AsyncOpenAI(
)

# Add instructor patch for response_model
client = instructor.from_openai(openai)

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<path_to_tesseract>'


def pil_image_to_base64(pil_img: Image.Image) -> str:
    """Convert a PIL Image to a base64 string in the format expected by OpenAI's API.
    
    Args:
        pil_img: PIL Image to convert
        
    Returns:
        Base64-encoded string with data URL prefix
    """
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def pdf_pages(pdf_data: bytes, filename: str) -> tuple[list[Page], str]:
    with pdfplumber.open(BytesIO(pdf_data)) as pdf_file:
        return (pdf_file.pages, filename)


async def segment_pages(pages: list[ProcessedPage]) -> list[ProcessedPage]:
    """
    Scan pages sequentially and assign each a section label based on keywords.
    Expected order is: Summary of the Invention → Detailed Description → Claims.
    If no keyword is detected before a page, default to "Summary of Invention".
    Enforces that each section can only be detected once, in the correct order.
    Handles section headers appearing anywhere on the page, not just at the beginning.
    """
    segmented_pages = []
    current_section = None
    
    # Track which sections have been detected
    # The expected order is: Summary -> Detailed Description -> Claims
    section_order = ["summary of invention", "detailed description", "claims"]
    detected_sections = set()
    
    logger.info("Starting section detection process")
    logger.info(f"Processing {len(pages)} pages")
    
    # Sort pages by page number
    for original_page in sorted(pages, key=lambda p: p.page_number):
        # Create a deep copy to work with, avoiding modification of original objects
        page = original_page.model_copy(deep=True)
        logger.info(f"Processing page {page.page_number}")
        
        # If no section has been detected yet, default to "Summary of Invention"
        if current_section is None:
            current_section = "summary of invention"
            detected_sections.add(current_section)
            # Update the page's section
            page.section = current_section
            logger.info(f"First page default section: {current_section}")
            segmented_pages.append(page)
            continue
            
        text_lower = page.text.lower()
        text_lines = page.text.split('\n')
        
        # Cross-reference patterns that should not be mistaken for section headers
        cross_reference_patterns = [
            r"(?:as|like|according to|mentioned in|described in|refers to|reference to).*(?:the|in|at|on).*(?:summary|detailed description|claims)",
            r"(?:see|per|from).*(?:the|in|at|on).*(?:summary|detailed description|claims)"
        ]
        
        # Limit header keyword search to the first part of the page to avoid
        # matching in-body cross-references. Characters after this limit will be
        # ignored when looking for section headers.
        HEADER_SEARCH_LIMIT = 400
        
        # Find section header candidates - check the entire page, not just the beginning
        section_found = False
        detected_section = None
        detection_method = None
        matched_text = None
        
        logger.info(f"Checking for section headers in all lines of page {page.page_number}")
        
        # Check ALL lines for section headers, not just the first few
        for i, line in enumerate(text_lines):
            line_lower = line.lower().strip()
            
            # Skip if line appears to be a cross-reference
            is_cross_reference = False
            for pattern in cross_reference_patterns:
                if re.search(pattern, line_lower):
                    logger.info(f"Skipping potential cross-reference: '{line}'")
                    is_cross_reference = True
                    break
                
            if is_cross_reference:
                continue
                
            # Check for header patterns: ALL CAPS, roman numerals, section numbers
            is_header = ((line.isupper() and len(line.strip()) > 5) or 
                         re.match(r"^[IVX]+\.\s+", line) or 
                         re.match(r"^\d+\.\s+SECTION", line, re.IGNORECASE))
            
            if is_header:
                logger.info(f"Found potential header (line {i+1}): '{line}'")
                
                # Strict regex for Detailed Description header variants
                detailed_desc_patterns = [
                    r"\bdetailed\s+description\b",
                    r"\bdetailed\s+description\s+of\s+the\s+invention\b",
                    r"\bdetailed\s+description\s+and\s+preferred\s+embodiments\b",
                    r"\bdetailed\s+description\s+of\s+the\s+embodiments\b",
                    r"\bdescription\s+of\s+the\s+invention\b"
                ]
                matched_detailed_header = any(re.search(pat, line_lower) for pat in detailed_desc_patterns)
                if matched_detailed_header and "detailed description" not in detected_sections:
                    # Only consider Detailed Description after Summary has been detected
                    if "summary of invention" in detected_sections:
                        detected_section = "detailed description"
                        section_found = True
                        detection_method = "strong header match"
                        matched_text = line
                        logger.info(f"DETECTED 'Detailed Description' via strong header match: '{line}'")
                        break
                elif "claims" in line_lower and "claims" not in detected_sections:
                    # Only consider Claims after at least Summary and Detailed Description have been detected
                    if "summary of invention" and "detailed description" in detected_sections:
                        detected_section = "claims"
                        section_found = True
                        detection_method = "strong header match"
                        matched_text = line
                        logger.info(f"DETECTED 'Claims' via strong header match: '{line}'")
                        break
        
        # If no strong headers were found, try fallback detection with keyword search
        if not section_found:
            logger.info(f"No strong headers found, trying keyword search for page {page.page_number}")
            
            # Filter out text with cross-references before doing general keyword search
            filtered_text = text_lower
            for pattern in cross_reference_patterns:
                filtered_text = re.sub(pattern, "", filtered_text)
            
            # Look for section keywords in order, respecting the expected sequence
            current_idx = section_order.index(current_section)
            
            # Only look for sections that come after the current one
            for next_idx in range(current_idx + 1, len(section_order)):
                next_section = section_order[next_idx]
                logger.info(f"Checking for '{next_section}' section")
                
                if next_section == "detailed description" and "detailed description" not in detected_sections:
                    # Check for various forms of "detailed description" header
                    detailed_patterns = [
                        r"detailed\s+description",
                        r"detailed\s+description\s+of\s+the\s+invention", 
                        r"description\s+of\s+embodiments", 
                        r"description\s+of\s+the\s+embodiments",
                        r"detailed\s+description\s+of\s+the\s+embodiments",
                        r"detailed\s+description\s+and\s+preferred\s+embodiments"
                    ]
                    
                    # Look for these patterns as standalone headers, checking context
                    for pattern in detailed_patterns:
                        matches = list(re.finditer(pattern, filtered_text))
                        for match in matches:
                            start_pos = match.start()
                            # Ignore matches that are far into the page body
                            if start_pos > HEADER_SEARCH_LIMIT:
                                logger.debug(
                                    f"Ignoring detailed description match beyond limit at position {start_pos}: '{match.group(0)}'"
                                )
                                continue
                            matched_phrase = match.group(0)
                            
                            # Extract the entire line containing the match to examine word count
                            line_start = filtered_text.rfind("\n", 0, start_pos) + 1
                            line_end = filtered_text.find("\n", start_pos)
                            if line_end == -1:
                                line_end = len(filtered_text)
                            line_content = filtered_text[line_start:line_end].strip()
                            word_count = len(line_content.split())
                            # Heuristic: true headers are usually short (<= 8 words)
                            if word_count > 8:
                                logger.debug(
                                    f"Skipping long potential header ({word_count} words): '{line_content}'"
                                )
                                continue
                            
                            # Check if this looks like a header (beginning of text, after newline, or after period)
                            is_potential_header = (
                                start_pos < 50 or filtered_text[start_pos-1:start_pos] in ["\n", "."]
                            )
                            
                            if is_potential_header:
                                logger.info(f"Found potential detailed description header: '{matched_phrase}'")
                                
                                # Check if no trailing text after this pattern on the same line
                                if any(line_content.endswith(suffix) for suffix in ["description", "embodiments", "invention"]):
                                    detected_section = "detailed description"
                                    section_found = True
                                    detection_method = "keyword match"
                                    matched_text = matched_phrase
                                    logger.info(
                                        f"DETECTED 'detailed description' via keyword match (validated): '{matched_phrase}'"
                                    )
                                    break
                        if section_found:
                            break
                            
                elif next_section == "claims" and "claims" not in detected_sections:
                    # Look for "Claims" as a standalone header
                    claim_matches = list(re.finditer(r"claims", filtered_text))
                    for match in claim_matches:
                        start_pos = match.start()
                        # Ignore matches beyond search limit
                        if start_pos > HEADER_SEARCH_LIMIT:
                            logger.debug(
                                f"Ignoring claims match beyond limit at position {start_pos}: '{match.group(0)}'"
                            )
                            continue
                        matched_phrase = match.group(0)
                        
                        # Check if this looks like a header
                        line_start = filtered_text.rfind("\n", 0, start_pos) + 1
                        line_end = filtered_text.find("\n", start_pos)
                        if line_end == -1:
                            line_end = len(filtered_text)
                        line_content = filtered_text[line_start:line_end].strip()
                        if len(line_content.split()) > 4:
                            logger.debug(
                                f"Skipping long potential claims header: '{line_content}'"
                            )
                            continue
                        is_potential_header = (
                            start_pos < 50 or filtered_text[start_pos-1:start_pos] in ["\n", "."]
                        )
                        if is_potential_header:
                            logger.info(f"Found potential claims header: '{matched_phrase}'")
                            
                            # Verify it's actually the Claims section with numbered items
                            if re.search(r"^\s*\d+\.\s+", filtered_text[start_pos:], re.MULTILINE):
                                detected_section = "claims"
                                section_found = True
                                detection_method = "keyword match"
                                matched_text = matched_phrase
                                logger.info(
                                    f"DETECTED 'claims' via keyword match (validated): '{matched_phrase}'"
                                )
                                break
                if section_found:
                    break
        
        # Update current section if a new one was detected
        if section_found:
            logger.info(f"SECTION CHANGE on page {page.page_number}: {current_section} -> {detected_section}")
            logger.info(f"Detection method: {detection_method}")
            logger.info(f"Matched text: '{matched_text}'")
            current_section = detected_section
            detected_sections.add(current_section)
        else:
            # Use language model if no clear section was found, with confidence threshold
            # But only if we haven't seen all sections yet
            if len(detected_sections) < len(section_order):
                logger.info(f"No section detected through patterns, trying language model for page {page.page_number}")
                try:
                    # Get the patent section classifier prompt from Langfuse
                    patent_classifier_prompt = get_prompt("section_detection", variables={
                        "text": page.text
                    })
                except Exception as prompt_error:
                    logger.error(f"Error getting patent section classifier prompt: {prompt_error}")
                    # Make the OpenAI API call using the compiled prompt
                    response = await client.chat.completions.create(
                        model="o4-mini",
                        reasoning_effort="high",    
                        messages=[patent_classifier_prompt],
                        response_model=PatentSectionWithConfidence,
                    )
                    
                    suggested_section = response.section
                    
                    # Only consider detected sections if:
                    # 1. The confidence exceeds the threshold
                    # 2. We haven't seen this section before
                    # 3. This section comes after the current one in the expected order
                    confidence_threshold = 0.8
                    current_idx = section_order.index(current_section)
                    suggested_idx = section_order.index(suggested_section)
                    
                    if (response.confidence > confidence_threshold and 
                        suggested_section not in detected_sections and 
                        suggested_idx > current_idx):
                        
                        logger.info(f"AI classified page {page.page_number} as {suggested_section} with confidence {response.confidence}")
                        current_section = suggested_section
                        detected_sections.add(current_section)
                    else:
                        if suggested_section in detected_sections:
                            logger.info(f"AI suggested {suggested_section} but this section was already detected")
                        elif suggested_idx <= current_idx:
                            logger.info(f"AI suggested {suggested_section} but this would violate section order (current: {current_section})")
                        else:
                            logger.info(f"AI classification had low confidence ({response.confidence}), keeping current section {current_section}")
                except Exception as e:
                    logger.error(f"Error using language model for section classification: {e}")
        
        # Update the page's section
        page.section = current_section
        logger.info(f"Final section for page {page.page_number}: {current_section}")
        segmented_pages.append(page)
        
    logger.info(f"Section detection complete. Sections found: {', '.join(detected_sections)}")
    return segmented_pages


def process_pdf_pages(pdf: tuple[list[Page], str]) -> list[ProcessedPage]:
    processed_pages: list[ProcessedPage] = []
    pdf_pages, pdf_name = pdf

    for page in pdf_pages:
        page_text = page.extract_text()
        if page_text == "":
            logger.info(
                f"PDFPlumber failed to extract text from page {page.page_number}"
            )
            try:
                # Try OCR if PDFPlumber fails
                image = page.to_image(width=1920)
                page_ocr = pytesseract.image_to_string(image.original)
                if page_ocr == "":
                    logger.error(f"OCR failed to extract text from page {page.page_number}")
                else:
                    # Convert PIL image to base64 string in OpenAI-compatible format
                    base64_img = pil_image_to_base64(image.original)
                    processed_pages.append(
                        ProcessedPage(
                            text=page_ocr, 
                            filename=pdf_name, 
                            page_number=page.page_number,
                            section="",  # Empty section to be filled by segment_pages
                            image=base64_img  # Store this as base64 for LLM
                        )
                    )
                    logger.info(
                        f"OCR successfully extracted {len(page_ocr)} characters from page {page.page_number}"
                    )
            except Exception as e:
                logger.error(f"OCR failed with error: {e}")
        else:
            try:
                img_obj = page.to_image(width=1920)
                base64_img = pil_image_to_base64(img_obj.original)
            except Exception as img_err:
                logger.error(f"Failed to render image for page {page.page_number}: {img_err}")
                base64_img = None

            processed_pages.append(
                ProcessedPage(
                    text=page_text, 
                    filename=pdf_name, 
                    page_number=page.page_number,
                    section="",  # Empty section to be filled by segment_pages
                    image=base64_img
                )
            )
            logger.info(
                f"PDFPlumber successfully extracted {len(page_text)} characters from page {page.page_number}"
            )
    return processed_pages

@observe(name='description-subheaders-detection')
async def detect_description_header(segmented_page: ProcessedPage) -> HeaderDetectionPage:
    # Log the start of detection for this page
    logger.info(
        f"Starting header detection for {segmented_page.filename} page {segmented_page.page_number}"
    )
    
    prompt = """
    You are a header detection system for a biological patent ETL application.

    Analyze the image and identify if there is a clearly visible sub-section header.
    Focus on text that appears to be a title or heading, typically at the top of the page or section.
    
    - Ignore any 'example' headers
    - Ignore any step headers like step 1, step 2, etc.
    
    Return:
    - has_header: True if a clear header is found, False otherwise
    - header: The extracted header text if found, otherwise None
    """
    
    if not segmented_page.image:
        logger.warning(
            f"No image available for {segmented_page.filename} page {segmented_page.page_number}; skipping header detection."
        )
        # Using the field names expected in actual application model
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            text=getattr(segmented_page, 'text', None) or getattr(segmented_page, 'text_content', None),
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            section=segmented_page.section,   
            image=None
        )
    
    try:
        response = await client.chat.completions.create(
            model='o4-mini-2025-04-16',
            reasoning_effort='high',
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": segmented_page.image}
                    }
                ]
            }],
            response_model=HeaderDetection
        )
        logger.info(
            f"Header detection finished for {segmented_page.filename} page {segmented_page.page_number}: "
            f"has_header={response.has_header}, header='{response.header}'"
        )
        
        # Using the field names expected in actual application model
        return HeaderDetectionPage(
            header=response.header,
            has_header=response.has_header,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image
        )
    except Exception as e:
        logger.error(
            f"Error in detect_description_header for {segmented_page.filename} page {segmented_page.page_number}: {str(e)}"
        )
        
        # Using the field names expected in actual application model
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image
        )


async def detect_description_headers(segmented_pages: list[ProcessedPage]) -> list[HeaderDetectionPage]:
    """Run header detection over a list of pages with progress logging."""
    total = len(segmented_pages)
    logger.info(f"Beginning header detection over {total} page(s) concurrently")

    # Run header detection concurrently
    tasks = [detect_description_header(page) for page in segmented_pages]
    results = await asyncio.gather(*tasks)

    logger.info("Header detection batch complete")
    return results

@observe(name='summary-subheaders-detection')
async def detect_summary_header(segmented_page: ProcessedPage) -> HeaderDetectionPage:
    """Detect subsection headers on a Summary of the Invention page."""
    logger.info(
        f"Starting header detection for SUMMARY page {segmented_page.filename} page {segmented_page.page_number}"
    )

    prompt = """
    You are a header detection system for a biological patent ETL application.

    Analyze the image and identify if there is a clearly visible sub-section header specific to the SUMMARY OF THE INVENTION section.
    Ignore any 'example' headers or enumeration like step 1, step 2, etc.

    Return:
    - has_header: True if a clear header is found, False otherwise
    - header: The extracted header text if found, otherwise None
    """

    if not segmented_page.image:
        logger.warning(
            f"No image available for {segmented_page.filename} page {segmented_page.page_number}; skipping header detection."
        )
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            text=getattr(segmented_page, 'text', None) or getattr(segmented_page, 'text_content', None),
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            section=segmented_page.section,
            image=None,
        )

    try:
        response = await client.chat.completions.create(
            model="o4-mini-2025-04-16",
            reasoning_effort="high",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": segmented_page.image}},
                ],
            }],
            response_model=HeaderDetection,
        )
        logger.info(
            f"Header detection finished for SUMMARY page {segmented_page.filename} page {segmented_page.page_number}: has_header={response.has_header}, header='{response.header}'"
        )
        return HeaderDetectionPage(
            header=response.header,
            has_header=response.has_header,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image,
        )
    except Exception as e:
        logger.error(
            f"Error in detect_summary_header for {segmented_page.filename} page {segmented_page.page_number}: {str(e)}"
        )
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image,
        )

@observe(name='claims-subheaders-detection')
async def detect_claims_header(segmented_page: ProcessedPage) -> HeaderDetectionPage:
    """Detect subsection headers on a Claims page."""
    logger.info(
        f"Starting header detection for CLAIMS page {segmented_page.filename} page {segmented_page.page_number}"
    )

    prompt = """
    You are a header detection system for a biological patent ETL application.

    Analyze the image and identify if there is a clearly visible sub-section header specific to the CLAIMS section.
    Ignore any 'example' headers or enumeration like step 1, step 2, etc.

    Return:
    - has_header: True if a clear header is found, False otherwise
    - header: The extracted header text if found, otherwise None
    """

    if not segmented_page.image:
        logger.warning(
            f"No image available for {segmented_page.filename} page {segmented_page.page_number}; skipping header detection."
        )
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            text=getattr(segmented_page, 'text', None) or getattr(segmented_page, 'text_content', None),
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            section=segmented_page.section,
            image=None,
        )

    try:
        response = await client.chat.completions.create(
            model="o4-mini-2025-04-16",
            reasoning_effort="high",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": segmented_page.image}},
                ],
            }],
            response_model=HeaderDetection,
        )
        logger.info(
            f"Header detection finished for CLAIMS page {segmented_page.filename} page {segmented_page.page_number}: has_header={response.has_header}, header='{response.header}'"
        )
        return HeaderDetectionPage(
            header=response.header,
            has_header=response.has_header,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image,
        )
    except Exception as e:
        logger.error(
            f"Error in detect_claims_header for {segmented_page.filename} page {segmented_page.page_number}: {str(e)}"
        )
        return HeaderDetectionPage(
            header=None,
            has_header=False,
            section=segmented_page.section,
            text=segmented_page.text,
            filename=segmented_page.filename,
            page_number=segmented_page.page_number,
            image=segmented_page.image,
        )

async def detect_section_headers(segmented_pages: list[ProcessedPage]) -> list[HeaderDetectionPage]:
    """Run header detection concurrently across all major patent sections."""
    section_to_detector = {
        "summary of invention": detect_summary_header,
        "detailed description": detect_description_header,
        "claims": detect_claims_header,
    }

    tasks = []
    for page in segmented_pages:
        detector = section_to_detector.get(page.section.lower())
        if detector:
            tasks.append(detector(page))
        else:
            logger.warning(
                f"No header detector for section '{page.section}' on page {page.page_number}; skipping."
            )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks)
    return results

examples = [
    """
    In certain aspects the disclosure relates to a method for preventing or treating alcoholic liver disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product obtained from an egg‑producing animal, thereby preventing or treating the alcoholic liver disease in the subject, wherein the hyperimmunized egg product comprises a therapeutically effective amount of one or more antibodies to an antigen selected from the group consisting of Enterococcus faecalis and Enterococcus faecalis cytolysin toxin.
    """,
    """
    In certain aspects the disclosure relates to a method for preventing or treating graft‑versus‑host disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product obtained from an egg‑producing animal, thereby preventing or treating the graft‑versus‑host disease in the subject, wherein the hyperimmunized egg product comprises a therapeutically effective amount of one or more antibodies to an antigen selected from the group consisting of Enterococcus faecalis, Enterococcus faecalis cytolysin toxin, and Enterococcus faecium.
    """,
    """
    In certain aspects the disclosure relates to a hyperimmunized egg produced by an animal that has been hyperimmunized with an antigen selected from the group consisting of Enterococcus faecalis, isolated Enterococcus faecalis cytolysin toxin, and Enterococcus faecium, wherein the level of antibodies to the antigen in the hyperimmunized egg is increased relative to an egg from an animal that has not been hyperimmunized. In certain embodiments, the animal has been hyperimmunized with Enterococcus faecalis and isolated Enterococcus faecalis cytolysin toxin, and wherein the level of antibodies to the Enterococcus faecalis and the isolated Enterococcus faecalis cytolysin toxin is increased relative to an egg from an animal that has not been hyperimmunized.
    """,
    """
    In certain aspects the disclosure relates to a hyperimmunized egg product obtained from a hyperimmunized egg described herein. In certain embodiments, the hyperimmunized egg product is whole egg. In certain embodiments, the hyperimmunized egg product is egg yolk. In certain embodiments, the hyperimmunized egg product is purified or partially purified IgY antibody to Enterococcus faecalis cytolysin toxin. In certain embodiments, the hyperimmunized egg product is purified or partially purified IgY antibody to Enterococcus faecalis. In certain embodiments, the hyperimmunized egg product consists of purified or partially purified IgY antibody to Enterococcus faecalis and purified or partially purified IgY antibody to Enterococcus faecalis cytolysin toxin.
    """,
    """
    In certain embodiments, the pharmaceutical composition is formulated for oral administration. In certain embodiments, the hyperimmunized egg product is formulated in nanoparticles or in an emulsion. In certain embodiments, the pharmaceutical composition is formulated for intravenous administration.
    """,
]


async def get_embodiments(page: ProcessedPage) -> list[Embodiment]:
    
    page_section = page.section
    page_text = page.text
    filename = page.filename
    page_number = page.page_number

    logger.info(f"Extracting embodiments from page {page_number} of {filename}")
    completion = await client.chat.completions.create(
        model='o4-mini-2025-04-16',
        reasoning_effort='high',
        messages=[
            {
                "role": "user",
                "content": """
                    You receive a page from a biology patent document.
                    page_number: {{ page_number }}
                    page_section: {{ page_section }}
                    filename: {{ filename }}
                    
                    Use the examples below for understanding of what an Embodiment looks like in a patent document.
                    <EmbodimentExamples>
                    {{ example_embodiments }}
                    </EmbodimentExamples>
                    
                    Extract any Patent Embodiments from the text below
                    
                    Text: 
                    <text>
                    {{ page_text }}
                    </text>
                    
                    
                    Rules
                    - if you can't identify any embodiments, return an empty list
                """,
            }
        ],
        response_model=Embodiments,
        context={
            "page_number": page_number,
            "filename": filename,
            "page_text": page_text,
            "page_section": page_section,
            "example_embodiments": chr(10).join(examples),
        },
    )
    logger.info(f"Embodiments extracted from page {page_number} of {filename}")
    return completion.content


async def find_embodiments(pages: list[ProcessedPage]) -> list[Embodiment]:
    tasks = [asyncio.create_task(get_embodiments(page)) for page in pages]
    results = await asyncio.gather(*tasks)
    patent_embodiments = [
        Embodiment(**embodiment) if isinstance(embodiment, dict) else embodiment
        for embodiments in results
        for embodiment in embodiments
    ]
    return patent_embodiments

async def categorize_embodiment(embodiment: Embodiment) -> DetailedDescriptionEmbodiment: 
    # Get just the category from the API
    response = await client.chat.completions.create(
        model='o4-mini-2025-04-16',
        reasoning_effort='high',
        messages=[
                {
                    "role":"user",
                    "content":"""
                    Analyze the following embodiment of a patent {{ filename }}:
                    
                    this embodiment comes from page {{ page_number }}
                    
                    this embodiment is a {{ section }} embodiment
                    
                    You are tasked with categorizing the embodiment into exactly one of the following categories:
                    - disease rationale
                    - product composition
                    
                    <content>
                    {{ content }}
                    </content> 
                    
                    
                    <rules>
                    - If the embodiment is related to the disease or condition the subject is suffering from, categorize it as "disease rationale".
                    - If the embodiment is related to the composition of the product, categorize it as "product composition".
                    </rules>
                    """
                }
        ],
        response_model=CategoryResponse,
        context={
            "filename": embodiment.filename,
            "content": embodiment.text,
            "section": embodiment.section,
            "page_number": embodiment.page_number,
        }
    )
    
    # Manually create a DetailedDescriptionEmbodiment with all fields from original embodiment plus the sub_category
    return DetailedDescriptionEmbodiment(
        text=embodiment.text,
        filename=embodiment.filename,
        page_number=embodiment.page_number,
        section=embodiment.section,
        sub_category=response.sub_category,
    )


#test categorize embodiment with this text
async def test_categorize_embodiment():
    embodiment_text = """
    In certain embodiments, the composition comprises at least 0.01%, 0.05%, 0.1%, 0.5%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 96%, 97%, 98% or 99% w/w of the hyperimmunized egg product. Any of these values may be used to define a range for the concentration of the hyperimmunized egg product in the composition. For example, in some embodiments, the composition comprises between 0.01% and 50%, between 0.1% and 50%, or between 1% and 50% w/w of the hyperimmunized egg product.
    """
    embodiment = Embodiment(text=embodiment_text, filename="test", page_number=1, section="detailed description")
    result = await categorize_embodiment(embodiment)
    
    # Log detailed information about the result
    logger.info(f"Result type: {type(result).__name__}")
    logger.info(f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'Not a complex object'}")
    logger.info(f"Is result iterable: {hasattr(result, '__iter__') and not isinstance(result, (str, DetailedDescriptionEmbodiment))}")
    
    if hasattr(result, 'model_dump'):
        logger.info(f"Model dump: {result.model_dump()}")
    
    # Print the result for inspection
    print(f"Result: {result}")

    print(result)
    

async def categorize_detailed_description(embodiments: list[Embodiment]) -> list[DetailedDescriptionEmbodiment]:
    """Categorize a list of detailed description embodiments.
    
    This function takes a list of Embodiment objects and returns a list of DetailedDescriptionEmbodiment objects
    with the appropriate sub_category field set.
    """
    # Run categorize_embodiment for each embodiment - ensuring proper typing
    tasks = [asyncio.create_task(categorize_embodiment(embodiment)) for embodiment in embodiments]
    results = await asyncio.gather(*tasks)
    
    # Each result should be a single DetailedDescriptionEmbodiment object
    for i, result in enumerate(results):
        if not isinstance(result, DetailedDescriptionEmbodiment):
            logger.error(f"Expected DetailedDescriptionEmbodiment but got {type(result)} at index {i}")
            raise TypeError(f"categorize_embodiment returned {type(result)} instead of DetailedDescriptionEmbodiment")
    
    return results

embodiment_summarization_prompt = get_prompt('embodiment_summary')
async def summarize_embodiment(embodiment: Union[DetailedDescriptionEmbodiment, Embodiment], embodiments: Union[list[DetailedDescriptionEmbodiment], list[Embodiment]]) -> Union[DetailedDescriptionEmbodiment, Embodiment]:
    prompt = embodiment_summarization_prompt.compile(embodiment=embodiment.text, embodiments=[embodiment.text for embodiment in embodiments])
    res = await client.chat.completions.create(
        model='o4-mini-2025-04-16',
        messages=[{'role': 'system', 'content': prompt}],
        response_model=EmbodimentSummary
    )
    embodiment.summary = res.summary
    return embodiment

async def summarize_embodiments(embodiments: list[Embodiment | DetailedDescriptionEmbodiment]) -> list[Embodiment | DetailedDescriptionEmbodiment]:
    tasks = [asyncio.create_task(summarize_embodiment(embodiment, embodiments)) for embodiment in embodiments]
    results = await asyncio.gather(*tasks)
    return results

embodiment_spell_check_prompt = get_prompt('embodiment_spell_check')

@observe(name='embodiment-spellcheck')
async def embodiment_spell_check(embodiment: Union[Embodiment, DetailedDescriptionEmbodiment]) -> Union[Embodiment, DetailedDescriptionEmbodiment]:
    prompt = embodiment_spell_check_prompt.compile(embodiment=embodiment.text)
    res = await client.chat.completions.create(
        model='o4-mini-2025-04-16',
        reasoning_effort='high',
        messages=[{'role': 'system', 'content': prompt}],
        response_model=EmbodimentSpellCheck
    )
    embodiment.text = res.checked_text
    return embodiment

async def spell_check_embodiments(embodiments: list[Embodiment | DetailedDescriptionEmbodiment]) -> list[Embodiment | DetailedDescriptionEmbodiment]:
    tasks = [asyncio.create_task(embodiment_spell_check(embodiment)) for embodiment in embodiments]
    results = await asyncio.gather(*tasks)
    return results

async def add_headers_to_embodiments(
    dd_embs: list[DetailedDescriptionEmbodiment],
    header_pages: list[HeaderDetectionPage],
) -> list[DetailedDescriptionEmbodiment]:
    """Attach detected page headers to their corresponding detailed-description embodiments.
    
    Each embodiment is matched by `(filename, page_number)` to the result from
    `detect_description_headers`.
    
    Carry-forward rule for orphan pages:
        If a page has *no* detected header, assign the most recent header that
        appeared on an earlier page of the same file. This prevents orphan
        embodiments and mirrors how authors sometimes place a header once and
        continue content on subsequent pages.
    """
    # Direct look-ups for pages with explicit headers
    page_lookup: dict[tuple[str, int], str] = {
        (hp.filename, hp.page_number): hp.header  # type: ignore[arg-type]
        for hp in header_pages
        if hp.has_header and hp.header  # ensure header text present
    }

    # Track last seen header per file for carry-forward
    last_header: dict[str, str] = {}

    # Sort embodiments to ensure forward iteration by page order within each file
    dd_embs_sorted = sorted(dd_embs, key=lambda e: (e.filename, e.page_number))

    for emb in dd_embs_sorted:
        key = (emb.filename, emb.page_number)

        if key in page_lookup:
            # Direct match – use it and update last header cache
            emb.header = page_lookup[key]
            last_header[emb.filename] = page_lookup[key]
        else:
            # No header detected on this page – use carry-forward if available
            if emb.header is None and emb.filename in last_header:
                emb.header = last_header[emb.filename]

    return dd_embs

@observe()
async def extract_glossary_subsection(segmented_pages: list[ProcessedPage]) -> Glossary | None:
    """
    Use the Instructor library to extract glossary definitions
    from the first 40% of detailed description pages.
    Returns a Glossary if definitions are found, otherwise None.
    """
    detailed = [p for p in segmented_pages if p.section.lower() == "detailed description"]
    if not detailed:
        return None

    async def process_page(page: ProcessedPage) -> Glossary | None:
        prompt = f"""
            Extract key terms and their definitions from the following patent text. 
            Respond matching the Glossary Pydantic schema.
            
            Ignore terms that are not directly related to the scientific aspect of the patent
            
            Examples included of terms to be ignored:
            - "or", "a", "an", "the", "about", "herein"
            
            Examples of terms to be included:              
            Glossary definitions typically define terms with patterns like:
            - The term "<TERM>" refers to <DEFINITION>.
            - As used herein, the term "<TERM>" refers to <DEFINITION>.
            
            Examples include:
            - The term "control egg" refers to an egg obtained...
            - The term "antigen" refers to a substance...
            - As used herein, the term "antibody" is a protein...
            - As used herein, the term "hyperimmunization" means...
            
            Text:
            {page.text}
        """
        try:
            res: Glossary = await client.chat.completions.create(
                model="o4-mini-2025-04-16",
                reasoning_effort='high',
                messages=[{"role": "system", "content": prompt}],
                response_model=Glossary
            )
            logger.info(f"Glossary extraction completed on page {page.page_number}, extracted {len(res.definitions)} definitions")
            if res.definitions:
                for d in res.definitions:          # tag each definition
                    d.page_number = page.page_number
                res.filename = page.filename
                return res
        except Exception as e:
            logger.error(f"Glossary extraction error on page {page.page_number}: {e}")
        return None

    # Process pages concurrently to improve performance
    tasks = [process_page(pg) for pg in detailed]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r]
    if not successful:
        return None

    # deduplicate by lowercase term
    unique: dict[str, GlossaryDefinition] = {}
    for g in successful:
        for d in g.definitions:
            key = d.term.strip().lower()
            if key not in unique:
                unique[key] = d

    aggregated = Glossary(
        definitions=list(unique.values()),
        filename=successful[0].filename,
    )

    logger.info(
        "Aggregated %d unique glossary definitions from %d page(s)",
        len(aggregated.definitions),
        len(successful),
    )
    return aggregated

async def detect_glossary_pages(
    segmented_pages: list[ProcessedPage]
) -> list[tuple[ProcessedPage, GlossaryPageFlag]]:
    """
    Flag pages containing glossary definitions in the 'The term ... refers to ...' format.
    Processes pages concurrently for better performance.
    Returns list of (page, flag) tuples.
    """
    async def process_page(page: ProcessedPage) -> tuple[ProcessedPage, GlossaryPageFlag]:
        prompt = f"""
        You are an expert patent document analysis assistant integrated into an automated OCR pipeline.
        After segmenting pages into sections, identify if this Detailed Description page contains glossary-style definitions.
        Glossary definitions typically define terms with patterns like:
        - The term "<TERM>" refers to <DEFINITION>.
        - As used herein, the term "<TERM>" refers to <DEFINITION>.
        Examples include:
        - The term "control egg" refers to an egg obtained...
        - The term "antigen" refers to a substance...
        - As used herein, the term "antibody" is a protein...
        - As used herein, the term "hyperimmunization" means...

        Page Text:
        {page.text}
        """
        flag: GlossaryPageFlag = await client.chat.completions.create(
            model="o4-mini-2025-04-16",
            reasoning_effort="high",
            messages=[{"role": "system", "content": prompt}],
            response_model=GlossaryPageFlag
        )
        return (page, flag)

    # Process all pages concurrently
    tasks = [process_page(page) for page in segmented_pages]
    return await asyncio.gather(*tasks)

async def process_patent_document(
    pdf_data: bytes, filename: str) -> tuple[Glossary, list[Embodiment | DetailedDescriptionEmbodiment]]:
    try:
        # Process PDF pages
        pdf_processing_start = time()
        raw_pages = pdf_pages(pdf_data, filename)
        processed_pages = process_pdf_pages(raw_pages)
        pdf_processing_total = time() - pdf_processing_start
        logger.info(f"PDF processing completed in {pdf_processing_total:.2f} seconds")

        # Segment pages by section
        segmentation_start = time()
        segmented_pages = await segment_pages(processed_pages)
        segmentation_total = time() - segmentation_start
        logger.info(f"Page segmentation completed in {segmentation_total:.2f} seconds")
        
        # Detect glossary pages separately
        detailed_description_pages = [page for page in segmented_pages if page.section == "detailed description"]
        
        # Detect headers across all major patent sections (Summary, Detailed Description, Claims)
        major_section_pages = [p for p in segmented_pages if p.section in ["summary of invention", "detailed description", "claims"]]
        header_detection_pages = await detect_section_headers(major_section_pages)
        logger.info(f"Header detection completed for {len(header_detection_pages)} pages")
        
        glossary_flags = await detect_glossary_pages(detailed_description_pages)
        flagged_pages = [pg for pg, flag in glossary_flags if flag.is_glossary_page]
        logger.info(
            f"Detected {len(flagged_pages)} glossary pages via LLM:"
            f" {[p.page_number for p in flagged_pages]}"
        )

        # Extract glossary definitions via Instructor LLM
        glossary_subsection = await extract_glossary_subsection(flagged_pages)
        if glossary_subsection:
            logger.info(f"Extracted {len(glossary_subsection.definitions)} glossary definitions via LLM")

        # Extract embodiments       
        embodiments_extraction_start = time()
        embodiments = await find_embodiments(segmented_pages)
        embodiments_extraction_total = time() - embodiments_extraction_start
        logger.info(f"Embodiments extraction completed in {embodiments_extraction_total:.2f} seconds")
        
        # Validate that all embodiments are of the correct type
        if embodiments and len(embodiments) > 0:
            logger.info(f"First embodiment type: {type(embodiments[0]).__name__}")
            
            # Enforce strict typing - all items must be Embodiment instances
            for i, embodiment in enumerate(embodiments):
                if not isinstance(embodiment, Embodiment):
                    logger.error(f"Item at index {i} is of type {type(embodiment).__name__}, not Embodiment")
                    raise TypeError(f"Expected all items to be Embodiment instances, found {type(embodiment).__name__}")
        
        # Categorize detailed description embodiments - select only the ones with 'detailed description' section
        detailed_description_embodiments = [embodiment for embodiment in embodiments if embodiment.section == "detailed description"]
        logger.info(f"Found {len(detailed_description_embodiments)} detailed description embodiments to categorize")
        
        # Run categorization to add sub_category field to detailed description embodiments
            
        categorized_detailed_description = await categorize_detailed_description(detailed_description_embodiments)
        logger.info(f"Categorized {len(categorized_detailed_description)} detailed description embodiments")

        # Add headers to the categorized detailed description embodiments
        categorized_detailed_description = await add_headers_to_embodiments(
            categorized_detailed_description, header_detection_pages
        )
        logger.info("Added headers to categorized detailed description embodiments")
        
        #enforce that they are instances of DetailedDescriptionEmbodiment
        for i, embodiment in enumerate(categorized_detailed_description):
            if not isinstance(embodiment, DetailedDescriptionEmbodiment):
                logger.error(f"Item at index {i} is of type {type(embodiment).__name__}, not DetailedDescriptionEmbodiment")
                raise TypeError(f"Expected all items to be DetailedDescriptionEmbodiment instances, found {type(embodiment).__name__}")
            # Replace the original detailed_description embodiments with the categorized ones
        non_detailed_embodiments = [embodiment for embodiment in embodiments if embodiment.section != "detailed description"]
        embodiments_with_detailed_description_categorized = non_detailed_embodiments + categorized_detailed_description
        
        # Summarize all embodiments
        summarized_embodiments = await summarize_embodiments(embodiments_with_detailed_description_categorized)
        logger.info(f"Summarized {len(summarized_embodiments)} embodiments")
        
        # Spell check all embodiments
        spell_checked_embodiments = await spell_check_embodiments(summarized_embodiments)
        logger.info(f"Spell checked {len(spell_checked_embodiments)} embodiments")
        
        return glossary_subsection, spell_checked_embodiments
        
    except Exception as e: 
        raise RuntimeError(f"Error processing patent document: {str(e)}")