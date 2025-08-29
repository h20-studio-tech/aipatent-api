#!/usr/bin/env python3
"""
LLaMA Parse PDF Document Test with Gemini 2.5 Flash
This script processes PDF documents from the experiments folder using LLaMA Parse and analyzes them with Gemini.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from openai import OpenAI

# Load environment variables
load_dotenv()


class LlamaParseGeminiAnalyzer:
    """Analyzer that uses LLaMA Parse for PDF parsing and Gemini 2.5 Flash for analysis."""
    
    def __init__(self):
        # Set up LLaMA Parse
        self.llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_api_key:
            raise ValueError("LLaMA Parse API key not found. Set LLAMA_CLOUD_API_KEY environment variable.")
        
        # Set up Gemini via OpenAI SDK
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Initialize OpenAI client for Gemini
        self.gemini_client = OpenAI(
            api_key=self.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Initialize LLaMA Parse
        self.parser = LlamaParse(
            api_key=self.llama_api_key,
            num_workers=4,
            verbose=True,
            language="en",
        )
    
    async def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a PDF document using LLaMA Parse."""
        print(f"\n📄 Parsing PDF: {Path(pdf_path).name}")
        print("-" * 50)
        
        try:
            # Parse the document asynchronously
            result = await self.parser.aparse(pdf_path)
            
            # Get markdown documents
            markdown_documents = result.get_markdown_documents(split_by_page=True)
            
            # Extract content from result
            parsed_content = {
                "file_name": Path(pdf_path).name,
                "pages": [],
                "full_markdown": "",
                "num_pages": len(result.pages)
            }
            
            # Process pages from the raw result
            for idx, page in enumerate(result.pages):
                page_data = {
                    "page_number": idx + 1,
                    "markdown": page.md if hasattr(page, 'md') else "",
                }
                parsed_content["pages"].append(page_data)
                
                # Accumulate full markdown
                if page_data["markdown"]:
                    parsed_content["full_markdown"] += f"\n\n--- Page {idx + 1} ---\n\n{page_data['markdown']}"
                
                print(f"✅ Parsed page {idx + 1}: {len(page_data['markdown'])} characters")
            
            print(f"\n✨ Successfully parsed {len(result.pages)} pages")
            return parsed_content
            
        except Exception as e:
            print(f"❌ Error parsing PDF: {e}")
            raise
    
    def analyze_with_gemini(self, content: str, file_name: str) -> str:
        """Analyze parsed content using Gemini 2.5 Flash."""
        print(f"\n🤖 Analyzing {file_name} with Gemini 2.5 Flash...")
        print("-" * 50)
        
        analysis_prompt = f"""
        Analyze this document ({file_name}) and provide a comprehensive analysis:

        1. **Document Summary**: Provide a detailed summary of the document's main purpose and content
        2. **Document Type**: Identify what type of document this is (patent application, research paper, technical specification, etc.)
        3. **Key Technical Concepts**: List and explain the main technical concepts, innovations, or methodologies
        4. **Main Claims/Findings**: Highlight the most important claims, findings, or conclusions
        5. **Technical Details**: Identify specific technical details, processes, or implementations described
        6. **Innovation Assessment**: Assess what makes this document innovative or unique
        7. **Potential Applications**: Suggest potential real-world applications or uses
        8. **Document Structure**: Describe how the document is organized and its main sections

        Document content:
        """
        
        # Limit content for token constraints
        max_content_length = 25000
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content
        
        if len(content) > max_content_length:
            truncated_content += f"\n\n[Content truncated from {len(content)} to {max_content_length} characters]"
        
        full_prompt = f"{analysis_prompt}\n\n{truncated_content}"
        
        try:
            response = self.gemini_client.chat.completions.create(
                model="gemini-2.0-flash-exp",
                messages=[
                    {"role": "system", "content": "You are an expert technical document analyst specializing in patents, research papers, and technical specifications."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=3000,
            )
            
            analysis = response.choices[0].message.content
            print("✅ Analysis complete")
            return analysis
            
        except Exception as e:
            print(f"❌ Error during Gemini analysis: {e}")
            raise
    
    async def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Complete pipeline: Parse PDF and analyze with Gemini."""
        
        # Parse the PDF
        parsed_data = await self.parse_pdf(pdf_path)
        
        # Analyze with Gemini
        analysis = self.analyze_with_gemini(
            parsed_data["full_markdown"], 
            parsed_data["file_name"]
        )
        
        return {
            "file_name": parsed_data["file_name"],
            "pages_parsed": parsed_data["num_pages"],
            "total_characters": len(parsed_data["full_markdown"]),
            "gemini_analysis": analysis,
            "sample_content": parsed_data["full_markdown"][:500] + "..." if len(parsed_data["full_markdown"]) > 500 else parsed_data["full_markdown"]
        }


def print_analysis_results(results: Dict[str, Any]):
    """Print the analysis results to terminal."""
    print("\n" + "=" * 80)
    print(f"📊 ANALYSIS RESULTS: {results['file_name']}")
    print("=" * 80)
    
    print(f"\n📈 Document Statistics:")
    print(f"  • Pages Parsed: {results['pages_parsed']}")
    print(f"  • Total Characters: {results['total_characters']:,}")
    
    print(f"\n🔍 Gemini 2.5 Flash Analysis:")
    print("-" * 60)
    print(results['gemini_analysis'])
    
    print(f"\n📄 Sample Content:")
    print("-" * 60)
    print(results['sample_content'])
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)


async def main():
    """Main function to process PDFs from experiments folder."""
    
    print("\n🚀 LLaMA Parse + Gemini 2.5 Flash Analysis")
    print("=" * 60)
    
    # Find PDF files in experiments folder
    pdf_files = [
        "/workspaces/aipatent-api/experiments/sample_patents/COVID-19 NEUTRALIZING ANTIBODY DETE.pdf"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in pdf_files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No PDF files found in experiments folder")
        return
    
    print(f"Found {len(existing_files)} PDF files to process:")
    for f in existing_files:
        print(f"  • {Path(f).name}")
    
    try:
        analyzer = LlamaParseGeminiAnalyzer()
        
        # Process each PDF
        for pdf_path in existing_files:
            try:
                results = await analyzer.process_pdf(pdf_path)
                print_analysis_results(results)
                print("\n" + "▼" * 40 + " NEXT DOCUMENT " + "▼" * 40 + "\n")
            except Exception as e:
                print(f"❌ Error processing {Path(pdf_path).name}: {e}")
                continue
                
    except Exception as e:
        print(f"\n❌ Error initializing analyzer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())