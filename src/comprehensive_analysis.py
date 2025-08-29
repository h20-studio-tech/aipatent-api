"""
Simple Comprehensive Analysis Service
Parse with LLaMA Parse -> Analyze with Gemini -> Return result
"""

import os
import asyncio
from typing import Dict, Any
from datetime import datetime

from llama_cloud_services import LlamaParse
from openai import OpenAI
from fastapi import HTTPException


class ComprehensiveAnalysisService:
    """Simple service for document analysis using LLaMA Parse + Gemini."""
    
    def __init__(self):
        # LLaMA Parse setup
        self.llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY required")
        
        # Gemini setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        self.parser = LlamaParse(
            api_key=self.llama_api_key,
            num_workers=4,
            verbose=True,
            language="en"
        )
        
        self.gemini_client = OpenAI(
            api_key=self.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    async def analyze_from_lancedb(self, table_name: str, filename: str, db) -> Dict[str, Any]:
        """Get content from LanceDB and analyze with Gemini (bypasses LLaMA Parse)."""
        
        try:
            # Get all content from LanceDB table
            table = await db.open_table(table_name)
            
            # Use vector_search with a dummy vector to get all rows
            # Since we want all content, we'll create a simple query vector
            query_vector = [0.0] * 3072  # Assuming text-embedding-3-large dimensions
            
            # Get search results as DataFrame - using to_pandas()
            df = await table.vector_search(query_vector).limit(1000).to_pandas()
            
            # Combine all text chunks
            parsed_content = ""
            for idx, row in df.iterrows():
                text_content = row.get('text', '')
                parsed_content += f"\n\n--- Chunk {idx + 1} ---\n\n{text_content}"
            
            # Analyze with Gemini
            gemini_response = await asyncio.to_thread(
                self.gemini_client.chat.completions.create,
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "Analyze this document comprehensively."},
                    {"role": "user", "content": parsed_content}
                ],
                temperature=0.2,
            )
            
            analysis = gemini_response.choices[0].message.content
            
            return {
                "filename": filename,
                "parsed_content": parsed_content,
                "gemini_analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "source": "lancedb"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LanceDB analysis failed: {str(e)}")

    async def analyze_from_file_content(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse document with LLaMA Parse and analyze with Gemini."""
        
        try:
            # Parse with LLaMA Parse
            result = await self.parser.aparse(file_content, extra_info={"file_name": filename})
            
            # Get markdown content
            parsed_content = ""
            for idx, page in enumerate(result.pages):
                page_content = page.md if hasattr(page, 'md') else ""
                parsed_content += f"\n\n--- Page {idx + 1} ---\n\n{page_content}"
            
            # Analyze with Gemini
            gemini_response = await asyncio.to_thread(
                self.gemini_client.chat.completions.create,
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "Analyze this document comprehensively."},
                    {"role": "user", "content": parsed_content}
                ],
                temperature=0.2,
            )
            
            analysis = gemini_response.choices[0].message.content
            
            return {
                "filename": filename,
                "parsed_content": parsed_content,
                "gemini_analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "source": "llama_parse"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLaMA Parse analysis failed: {str(e)}")

    async def analyze_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Legacy method - calls analyze_from_file_content."""
        return await self.analyze_from_file_content(file_content, filename)