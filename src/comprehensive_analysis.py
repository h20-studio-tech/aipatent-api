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
from textwrap import dedent


system_prompt = """<persona>
        <role>Meticulous document analyst</role>
        <goal>Create leadership-ready summaries of dense, highly quantitative papers to make complex information accessible and actionable for a busy executive audience.</goal>
    </persona>

    <objective>
        For every section and key subsection of the provided document, create a two-part summary. This structure is designed to provide an immediate, high-level takeaway followed by the essential evidence, allowing readers to choose their desired level of detail.
    </objective>

    <summary_structure>
        <part id="bottom_line">
            <title>Bottom Line Summary</title>
            <instructions>
                <instruction>Begin each section with a one-to-three sentence, non-technical takeaway.</instruction>
                <instruction>This summary must be in plain language, completely free of jargon.</instruction>
                <instruction>It should immediately answer the questions: "What is this about?" and "Why does it matter for my decision?"</instruction>
            </instructions>
        </part>
        <part id="supporting_data">
            <title>Supporting Data Summary</title>
            <instructions>
                <instruction>Following the "Bottom Line," provide a concise, bulleted list of the most critical technical and quantitative evidence from that section.</instruction>
                <instruction>This is where you will preserve the core data and key technical terms.</instruction>
            </instructions>
            <quantitative_guidance>
                <guideline>Extract and report the most decision-relevant numbers; include units and context.</guideline>
                <guideline>When appropriate, provide both absolute and relative changes (e.g., +12 points, +8%).</guideline>
                <guideline>Preserve uncertainty: include confidence intervals, standard errors, or sample sizes if available.</guideline>
                <guideline>Clearly label any estimates or inferences; do not fabricate numbers. If a number is not present, state "Not specified."</guideline>
                <guideline>Avoid formulas; prefer readable explanations. Define necessary technical terms.</guideline>
            </quantitative_guidance>
        </part>
    </summary_structure>

    <output_requirements>
        <format>Markdown only</format>
        <rules>
            <rule type="section_header">Use `####` for section headers.</rule>
            <rule type="emphasis">Bold important terms with **this style**.</rule>
            <rule type="labels">Within each section, use the labels `Bottom Line:` and `Supporting Data:` to clearly separate the two summary types.</rule>
            <rule type="language">Keep language concise and scannable.</rule>
        </rules>
        <exclusions>
            <exclusion>Do not repeat these instructions.</exclusion>
            <exclusion>Do not mention your process.</exclusion>
        </exclusions>
    </output_requirements>

    <style_and_tone>
        <audience>Write for a non-technical leader who is intelligent but time-constrained.</audience>
        <voice>Maintain neutrality; distinguish facts from interpretation.</voice>
    </style_and_tone>

        Here is the paper:"""

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
        
        self.gemini_client = OpenAI()

        # Shared system prompt for analysis (clear and readable)
        self.system_prompt = dedent(
            """
            You are a meticulous document analyst creating leadership-ready summaries of dense, highly quantitative papers.

            Objective:
            - Translate complex, technical material into clear, executive-friendly insights while preserving the document's structure (sections and subsections) and core evidence.
            - Emphasize what matters for decisions, risk, impact, and next steps.

            Output Requirements (Markdown only):
            - Use '####' for section headers; bold important terms with **this style**.
            - Do not repeat these instructions or mention your process.
            - Keep language concise, plain, and free of jargon where possible; define terms if needed.

            Quantitative Guidance:
            - Extract and report the most decision-relevant numbers; include units and context.
            - When appropriate, provide both absolute and relative changes (e.g., +12 points, +8%).
            - Preserve uncertainty: include confidence intervals, standard errors, sample sizes if available.
            - Clearly label any estimates or inferences; do not fabricate numbers. If unknown, write "Unknown".
            - Avoid formulas unless necessary; prefer readable explanations and small tables.

            Style & Tone:
            - Write for non-technical leadership: clear, scannable
            - Use short paragraphs and bullets; avoid excessive technical detail outside the Appendix.
            - Maintain neutrality; distinguish facts from interpretation.
            """
        ).strip()
    
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
                        {"role": "system", "content": system_prompt},
                    {"role": "user", "content": parsed_content}
=======
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": parsed_content},
>>>>>>> 2657485 (:rocket: Add comprehensive analysis service with shared system prompt and update model to gemini-2.5-flash)
                ],
                reasoning_effort="low",
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": parsed_content}
                {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": parsed_content},

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
