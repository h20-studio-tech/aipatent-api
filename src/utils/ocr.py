import json 
import instructor
import asyncio
import pytesseract
import pdfplumber
import sys
from time import time
from io import BytesIO
from pdfplumber.page import Page
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI
import re
import pandas as pd
from src.utils.logging_helper import create_logger
from src.utils.langfuse_client import get_prompt

client = instructor.from_openai(AsyncOpenAI())
logger = create_logger("ocr.py")

class ProcessedPage(BaseModel):
    text: str = Field(
        ..., description="The content of a page that contains embodiments"
    )
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(..., description="The section of the embodiment in the source file")


class Embodiment(BaseModel):
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(..., description="The page number of the embodiment in the source file")
    section: str = Field(..., description="The section of the embodiment in the source file")

class DetailedDescriptionEmbodiment(BaseModel):
    # Define all fields explicitly instead of using inheritance
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page_number: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )
    section: str = Field(..., description="The section of the embodiment in the source file")
    sub_category: str = Field(..., 
                          description="The category of the embodiment",
                          enum=["disease rationale", "product composition"])

class Embodiments(BaseModel):
    content: list[Embodiment] | list = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )


class PatentSection(BaseModel):
    """Classification of a patent document section."""
    section: str = Field(
        ..., 
        description="The section of the patent document",
        enum=["summary of invention", "detailed description", "claims"]
    )   


class PatentSectionWithConfidence(BaseModel):
    section: str = Field(enum=["summary of invention", "detailed description", "claims"])
    confidence: float

    @field_validator('confidence')
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('Confidence must be between 0 and 1')
        return v


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
                
                
                if ("detailed description" in line_lower or
                    "detailed description of the invention" in line_lower or
                    "detailed description and preferred embodiments" in line_lower or
                    "detailed description of the embodiments" in line_lower) and "detailed description" not in detected_sections:
                    # Only consider Detailed Description after Summary has been detected
                    if "summary of invention" in detected_sections:
                        detected_section = "detailed description"
                        section_found = True
                        detection_method = "strong header match"
                        matched_text = line
                        logger.info(f"DETECTED 'Detailed Description' via strong header match: '{line}'")
                        break
                elif "claims" in line_lower and "Claims" not in detected_sections:
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
                        r"detailed\s+description\s+of\s+the\s+embodiments"
                    ]
                    
                    # Look for these patterns as standalone headers, checking context
                    for pattern in detailed_patterns:
                        matches = list(re.finditer(pattern, filtered_text))
                        for match in matches:
                            start_pos = match.start()
                            matched_phrase = match.group(0)
                            
                            # Check if this looks like a header (beginning of text, after newline, or after period)
                            is_potential_header = (start_pos < 50 or filtered_text[start_pos-1:start_pos] in ["\n", "."])
                            
                            if is_potential_header:
                                logger.info(f"Found potential detailed description header: '{matched_phrase}'")
                                
                                # Check if no text after this pattern on the same line
                                line_end = filtered_text[start_pos:].find("\n")
                                if line_end == -1:  # End of text
                                    line_end = len(filtered_text[start_pos:])
                                line_content = filtered_text[start_pos:start_pos+line_end].strip()
                                
                                if any(line_content.endswith(suffix) for suffix in ["description", "embodiments", "invention"]):
                                    detected_section = "detailed description"
                                    section_found = True
                                    detection_method = "keyword match"
                                    matched_text = matched_phrase
                                    logger.info(f"DETECTED 'detailed description' via keyword match: '{matched_phrase}'")
                                    break
                        if section_found:
                            break
                            
                elif next_section == "claims" and "claims" not in detected_sections:
                    # Look for "Claims" as a standalone header
                    claim_matches = list(re.finditer(r"claims", filtered_text))
                    for match in claim_matches:
                        start_pos = match.start()
                        matched_phrase = match.group(0)
                        
                        # Check if this looks like a header
                        is_potential_header = (start_pos < 50 or filtered_text[start_pos-1:start_pos] in ["\n", "."])
                        if is_potential_header:
                            logger.info(f"Found potential claims header: '{matched_phrase}'")
                            
                            # Verify it's actually the Claims section with numbered items
                            if re.search(r"^\s*\d+\.\s+", filtered_text[start_pos:], re.MULTILINE):
                                detected_section = "claims"
                                section_found = True
                                detection_method = "keyword match"
                                matched_text = matched_phrase
                                logger.info(f"DETECTED 'claims' via keyword match: '{matched_phrase}'")
                                break
                
                if section_found:
                    break
        
        # Special case: If this is the last page and "claims" haven't been detected,
        # check more aggressively for claims indicators (numbered paragraphs starting with numbers)
        if not section_found and "claims" not in detected_sections:
            # Check if this page has a structure that looks like claims (numbered paragraphs)
            has_numbered_items = re.search(r"^\s*\d+\.\s+", text_lower, re.MULTILINE)
            if has_numbered_items:
                logger.info(f"Found numbered items on page {page.page_number}, checking if they're claims")
                
                # Additional verification: multiple numbered items and typical claim language
                numbered_items = re.findall(r"^\s*\d+\.\s+", text_lower, re.MULTILINE)
                has_claim_language = any(phrase in text_lower for phrase in ["comprising", "wherein", "consisting of"])
                
                if len(numbered_items) > 1 and has_claim_language:
                    detected_section = "claims"
                    section_found = True
                    detection_method = "structural analysis"
                    matched_text = f"Numbered items ({len(numbered_items)}) with claim language"
                    logger.info(f"DETECTED 'claims' via structural analysis: {len(numbered_items)} numbered items with claim language")
        
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
                    # Make the OpenAI API call using the compiled prompt
                    response = await client.chat.completions.create(
                        model="gpt-4.5-preview-2025-02-27",    
                        messages=[patent_classifier_prompt],
                        temperature=0.2,
                        response_model=PatentSectionWithConfidence,
                    )
                    
                    suggested_section = response.section
                    
                    # Only consider detected sections if:
                    # 1. The confidence exceeds the threshold
                    # 2. We haven't seen this section before
                    # 3. This section comes after the current one in the expected order
                    confidence_threshold = 0.7
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
                image = page.to_image(resolution=400)
                page_ocr = pytesseract.image_to_string(image.original)
                if page_ocr == "":
                    logger.error(f"OCR failed to extract text from page {page.page_number}")
                else:
                    processed_pages.append(
                        ProcessedPage(
                            text=page_ocr, 
                            filename=pdf_name, 
                            page_number=page.page_number,
                            section=""  # Empty section to be filled by segment_pages
                        )
                    )
                    logger.info(
                        f"OCR successfully extracted {len(page_ocr)} characters from page {page.page_number}"
                    )
            except Exception as e:
                logger.error(f"OCR failed with error: {e}")
        else:
            processed_pages.append(
                ProcessedPage(
                    text=page_text, 
                    filename=pdf_name, 
                    page_number=page.page_number,
                    section=""  # Empty section to be filled by segment_pages
                )
            )
            logger.info(
                f"PDFPlumber successfully extracted {len(page_text)} characters from page {page.page_number}"
            )
    return processed_pages


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
        model="o3-mini",
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
                    
                    Text: {{ page_text }}
                    
                    
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
    # Create a custom model for the API response that only needs to provide the sub_category
    class CategoryResponse(BaseModel):
        sub_category: str = Field(..., 
                           description="The category of the embodiment",
                           enum=["disease rationale", "product composition"])
    
    # Get just the category from the API
    response = await client.chat.completions.create(
        model='gpt-4.5-preview',
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
        sub_category=response.sub_category
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

async def process_patent_document(pdf_data: bytes, filename: str) -> list[Embodiment | DetailedDescriptionEmbodiment]:
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

        # Extract embodiments       
        embodiments_extraction_start = time()
        embodiments = await find_embodiments(segmented_pages)
        embodiments_extraction_total = time() - embodiments_extraction_start
        
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
            
        #enforce that they are instances of DetailedDescriptionEmbodiment
        for i, embodiment in enumerate(categorized_detailed_description):
            if not isinstance(embodiment, DetailedDescriptionEmbodiment):
                logger.error(f"Item at index {i} is of type {type(embodiment).__name__}, not DetailedDescriptionEmbodiment")
                raise TypeError(f"Expected all items to be DetailedDescriptionEmbodiment instances, found {type(embodiment).__name__}")
            # Replace the original detailed_description embodiments with the categorized ones
        non_detailed_embodiments = [embodiment for embodiment in embodiments if embodiment.section != "detailed description"]
        embodiments_with_detailed_description_categorized = non_detailed_embodiments + categorized_detailed_description
        
        return embodiments_with_detailed_description_categorized
        
    except Exception as e:
        raise RuntimeError(f"Error processing patent document: {str(e)}")
        