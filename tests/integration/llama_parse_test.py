#!/usr/bin/env python3
"""
LLaMA Parse PDF Document Test with Gemini via OpenAI SDK
This script tests parsing PDF documents with LLaMA Parse and analyzing them with Gemini.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from openai import OpenAI

load_dotenv()


class LlamaParseGeminiAnalyzer:
    def __init__(self, llama_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.llama_api_key = llama_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_api_key:
            raise ValueError("LLaMA Parse API key not found. Set LLAMA_CLOUD_API_KEY environment variable.")
        
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        self.gemini_client = OpenAI(
            api_key=self.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        self.parser = LlamaParse(
            api_key=self.llama_api_key,
            num_workers=4,
            verbose=True,
            language="en",
        )
    
    async def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        print(f"\n📄 Parsing PDF: {pdf_path}")
        print("-" * 50)
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            result = await self.parser.aparse(pdf_path)
            
            markdown_documents = result.get_markdown_documents(split_by_page=True)
            text_documents = result.get_text_documents(split_by_page=True)
            
            parsed_content = {
                "file_path": pdf_path,
                "file_name": Path(pdf_path).name,
                "pages": [],
                "full_text": "",
                "full_markdown": "",
                "images": []
            }
            
            for idx, page in enumerate(result.pages):
                page_data = {
                    "page_number": idx + 1,
                    "text": page.text if hasattr(page, 'text') else "",
                    "markdown": page.md if hasattr(page, 'md') else "",
                    "images": page.images if hasattr(page, 'images') else [],
                }
                parsed_content["pages"].append(page_data)
                
                if page_data["text"]:
                    parsed_content["full_text"] += f"\n\n--- Page {idx + 1} ---\n\n{page_data['text']}"
                if page_data["markdown"]:
                    parsed_content["full_markdown"] += f"\n\n--- Page {idx + 1} ---\n\n{page_data['markdown']}"
                
                if page_data["images"]:
                    parsed_content["images"].extend(page_data["images"])
                
                print(f"✅ Parsed page {idx + 1}: {len(page_data['text'])} characters")
            
            parsed_content["num_pages"] = len(result.pages)
            
            print(f"\n✨ Successfully parsed {len(result.pages)} pages")
            if parsed_content["images"]:
                print(f"📷 Found {len(parsed_content['images'])} images")
            
            return parsed_content
            
        except Exception as e:
            print(f"❌ Error parsing PDF: {e}")
            raise
    
    def analyze_with_gemini(self, content: str, analysis_prompt: Optional[str] = None) -> str:
        print(f"\n🤖 Analyzing with Gemini (via OpenAI SDK)...")
        print("-" * 50)
        
        if not analysis_prompt:
            analysis_prompt = """
            Please analyze this document and provide:
            1. Document Summary: A comprehensive summary (3-4 sentences)
            2. Document Type: Identify the type of document
            3. Key Topics and Themes: List the main topics discussed
            4. Technical Content: Identify any technical concepts or innovations
            5. Important Findings: Highlight the most important points
            6. Structure Analysis: Describe the document's organization
            7. Potential Applications: If applicable, suggest potential uses
            
            Document content:
            """
        
        max_content_length = 30000
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content
        
        if len(content) > max_content_length:
            truncated_content += f"\n\n[Note: Content truncated from {len(content)} to {max_content_length} characters]"
        
        full_prompt = f"{analysis_prompt}\n\n{truncated_content}"
        
        try:
            response = self.gemini_client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst specializing in technical and patent documents."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            
            analysis = response.choices[0].message.content
            print("✅ Analysis complete")
            return analysis
            
        except Exception as e:
            print(f"❌ Error during Gemini analysis: {e}")
            raise
    
    async def process_pdf(self, pdf_path: str, custom_analysis: Optional[str] = None) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("🚀 Starting LLaMA Parse + Gemini Analysis Pipeline")
        print("=" * 60)
        
        parsed_data = await self.parse_pdf(pdf_path)
        
        content_for_analysis = parsed_data["full_markdown"] if parsed_data["full_markdown"] else parsed_data["full_text"]
        
        analysis = self.analyze_with_gemini(content_for_analysis, custom_analysis)
        
        result = {
            "file_info": {
                "path": parsed_data["file_path"],
                "name": parsed_data["file_name"],
                "pages": parsed_data["num_pages"]
            },
            "parsed_content": parsed_data,
            "gemini_analysis": analysis,
            "statistics": {
                "total_characters": len(parsed_data["full_text"]),
                "total_markdown_characters": len(parsed_data["full_markdown"]),
                "pages_parsed": parsed_data["num_pages"],
                "images_found": len(parsed_data["images"])
            }
        }
        
        return result


def print_results(results: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\n📁 File Information:")
    print(f"  • Name: {results['file_info']['name']}")
    print(f"  • Path: {results['file_info']['path']}")
    print(f"  • Pages: {results['file_info']['pages']}")
    
    print("\n📈 Document Statistics:")
    stats = results['statistics']
    print(f"  • Total Characters: {stats['total_characters']:,}")
    print(f"  • Markdown Characters: {stats['total_markdown_characters']:,}")
    print(f"  • Pages Parsed: {stats['pages_parsed']}")
    print(f"  • Images Found: {stats['images_found']}")
    
    print("\n🔍 Gemini Analysis:")
    print("-" * 40)
    print(results['gemini_analysis'])
    
    if results['parsed_content']['pages']:
        first_page = results['parsed_content']['pages'][0]
        print("\n📄 Sample Content (First Page):")
        print("-" * 40)
        sample_text = first_page.get('markdown', first_page.get('text', ''))
        print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python llama_parse_test.py <path_to_pdf> [custom_prompt]")
        print("\nExample: python llama_parse_test.py sample.pdf")
        print("Example with custom prompt: python llama_parse_test.py sample.pdf 'Analyze for patent claims'")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    custom_prompt = None
    if len(sys.argv) > 2:
        custom_prompt = " ".join(sys.argv[2:])
        print(f"Using custom analysis prompt: {custom_prompt}")
    
    try:
        analyzer = LlamaParseGeminiAnalyzer()
        results = await analyzer.process_pdf(pdf_path, custom_prompt)
        print_results(results)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())