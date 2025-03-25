import os
import instructor
import asyncio
import pytesseract
import pdfplumber
import pprint
import pandas as pd
from time import time
from io import BytesIO
from pdfplumber.page import Page
from pydantic import BaseModel, Field
from src.utils.logging_helper import create_logger
from openai import AsyncOpenAI
from pathlib import Path


client = instructor.from_openai(AsyncOpenAI())
logger = create_logger("ocr.py")


class EmbodimentsPage(BaseModel):
    text: str = Field(
        ..., description="The content of a page that contains embodiments"
    )
    filename: str = Field(..., description="The source file of the embodiment")
    page: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )


class Embodiment(BaseModel):
    text: str = Field(..., description="The embodiment")
    filename: str = Field(..., description="The source file of the embodiment")
    page: int = Field(
        ..., description="The page number of the embodiment in the source file"
    )


class Embodiments(BaseModel):
    content: list[Embodiment] | list = Field(
        ..., description="The list of embodiments in a page that contains embodiments"
    )


def pdf_pages(pdf_data: bytes, filename: str) -> int:
    with pdfplumber.open(BytesIO(pdf_data)) as pdf_file:
        return (pdf_file.pages, filename)


def process_pdf_pages(pdf: tuple[list[Page], str]) -> list[EmbodimentsPage]:
    processed_pages: list[EmbodimentsPage] = []
    pdf_pages, pdf_name = pdf

    for page in pdf_pages:
        page_text = page.extract_text(keep_blank_chars=True)
        if page_text == "":
            logger.info(
                f"PDFPlumber failed to extract text from page {page.page_number}"
            )
            logger.info("Attempting OCR extraction")
            page_image = page.to_image(width=1920)
            page_ocr = pytesseract.image_to_string(page_image.original)
            if page_ocr == "":
                logger.error(f"OCR failed to extract text from page {page.page_number}")
            else:
                processed_pages.append(
                    EmbodimentsPage(
                        text=page_ocr, filename=pdf_name, page=page.page_number
                    )
                )
                logger.info(
                    f"OCR successfully extracted text from page {page.page_number} of {pdf_name}"
                )
        else:
            processed_pages.append(
                EmbodimentsPage(
                    text=page_text, filename=pdf_name, page=page.page_number
                )
            )
            logger.info(
                f"PDFPlumber successfully extracted text from page {page.page_number} of {pdf_name}"
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


async def get_embodiments(page: EmbodimentsPage) -> Embodiments:

    page_text = page.text
    filename = page.filename
    page_number = page.page

    logger.info(f"Extracting embodiments from page {page_number} of {filename}")
    completion = await client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                    You receive a page from a biology patent document.
                    page_number: {page_number}
                    filename: {filename}
                    
                    Use the examples below for understanding of what an Embodiment looks like in a patent document.
                    <EmbodimentExamples>
                    {chr(10).join(example for example in examples)}
                    </EmbodimentExamples>
                    
                    Extract any Patent Embodiments from the text below
                    
                    Text: {page_text}
                    
                    
                    Rules
                    - if you can't identify any embodiments, return an empty list
                """,
            }
        ],
        response_model=Embodiments,
    )
    logger.info(f"Embodiments extracted from page {page_number} of {filename}")
    return completion


async def find_embodiments(pages: list[EmbodimentsPage]) -> list[Embodiments]:
    patent_embodiments: list[Embodiments] = []
    tasks = [asyncio.create_task(get_embodiments(page)) for page in pages]
    patent_embodiments = await asyncio.gather(*tasks)
    return patent_embodiments


async def process_patent_document(pdf_data: bytes, filename: str) -> list[Embodiment]:
    try:
        # Process PDF pages
        pdf_processing_start = time()
        raw_pages = pdf_pages(pdf_data, filename)
        processed_pages = process_pdf_pages(raw_pages)
        pdf_processing_total = time() - pdf_processing_start
        logger.info(f"PDF processing completed in {pdf_processing_total:.2f} seconds")

        # Extract embodiments
        embodiments_extraction_start = time()
        embodiments_collections = await find_embodiments(processed_pages)
        embodiments_extraction_total = time() - embodiments_extraction_start
        logger.info(
            f"Embodiments extraction completed in {embodiments_extraction_total:.2f} seconds"
        )

        results = []
        for embodiment_collection in embodiments_collections:
            for embodiment in embodiment_collection.content:
                results.append(
                    Embodiment(
                        text=embodiment["text"],
                        filename=embodiment["filename"],
                        page=embodiment["page"],
                    )
                )

        return results

    except Exception as e:
        raise RuntimeError(f"Error processing patent document: {str(e)}")


# if __name__ == "__main__":
#     pdf_path = r"C:\Users\vtorr\Work\Projects\aipatent-api\ALD_GvHD_provisional_patent.pdf"
#     embodiments_list = asyncio.run(process_patent_document(pdf_path))

#     df = pd.DataFrame([embodiment.model_dump() for embodiment in embodiments_list])

#     df.to_csv("experiments/ald_gvhd_patent_embodiments.csv", index=False)
