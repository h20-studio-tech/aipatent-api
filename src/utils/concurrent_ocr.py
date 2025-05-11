import instructor
import asyncio
import pytesseract
import pdfplumber
from time import time
from io import BytesIO
from pdfplumber.page import Page
from openai import AsyncOpenAI
import re
from src.utils.logging_helper import create_logger
from src.utils.langfuse_client import get_prompt
from src.models.ocr_schemas import (
    Embodiments, 
    ProcessedPage, 
    Embodiment, 
    DetailedDescriptionEmbodiment,
    PatentSectionWithConfidence,
    CategoryResponse
    )
