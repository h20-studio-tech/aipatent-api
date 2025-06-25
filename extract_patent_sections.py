#!/usr/bin/env python3
"""
Extract abstract, summary, and claims sections from patent.json for use with embodiment extractor.
"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path


def load_patent_data(file_path: str) -> Dict:
    """Load the patent JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_abstract(patent_data: Dict) -> str:
    """Extract abstract from the first page."""
    pages = patent_data.get('pages', [])
    if not pages:
        return ""
    
    first_page = pages[0]
    md = first_page.get('md', '')
    
    # Look for abstract section
    lines = md.split('\n')
    abstract_lines = []
    in_abstract = False
    
    for line in lines:
        line = line.strip()
        if '(57) Abstract:' in line:
            in_abstract = True
            # Extract text after "Abstract:"
            abstract_text = line.split('(57) Abstract:', 1)
            if len(abstract_text) > 1:
                abstract_lines.append(abstract_text[1].strip())
            continue
        elif in_abstract:
            if line and not line.startswith('('):
                abstract_lines.append(line)
            elif line.startswith('(') and line != line.upper():
                # Stop when we hit another section marker
                break
    
    return ' '.join(abstract_lines).strip()


def extract_summary_section(patent_data: Dict) -> str:
    """Extract the Summary of the Invention section."""
    pages = patent_data.get('pages', [])
    summary_text = []
    in_summary = False
    
    for page in pages:
        md = page.get('md', '')
        if not md:
            continue
            
        lines = md.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check if we're entering the summary section
            if 'SUMMARY OF THE INVENTION' in line.upper():
                in_summary = True
                continue
            
            # Check if we're leaving the summary section
            if in_summary and line.startswith('#') and 'SUMMARY' not in line.upper():
                in_summary = False
                break
                
            # Collect summary content
            if in_summary and line and not line.startswith('#'):
                summary_text.append(line)
    
    return ' '.join(summary_text).strip()


def extract_claims_section(patent_data: Dict) -> List[str]:
    """Extract individual claims from the Claims section."""
    pages = patent_data.get('pages', [])
    claims = []
    in_claims = False
    current_claim = []
    
    for page in pages:
        md = page.get('md', '')
        if not md:
            continue
            
        lines = md.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check if we're entering the claims section
            if 'CLAIMS' in line.upper() and line.startswith('#'):
                in_claims = True
                continue
            
            # Check if we're leaving the claims section
            if in_claims and line.startswith('#') and 'CLAIMS' not in line.upper():
                # Save the last claim if any
                if current_claim:
                    claims.append(' '.join(current_claim).strip())
                break
                
            # Collect claims content
            if in_claims and line:
                # Check if this is a new claim (starts with a number)
                claim_match = re.match(r'^(\d+)\.\s*(.+)', line)
                if claim_match:
                    # Save previous claim if any
                    if current_claim:
                        claims.append(' '.join(current_claim).strip())
                    # Start new claim
                    current_claim = [claim_match.group(2)]
                elif current_claim:
                    # Continue current claim
                    current_claim.append(line)
    
    # Save the last claim if any
    if current_claim:
        claims.append(' '.join(current_claim).strip())
    
    return claims


def extract_all_sections(patent_file: str) -> Dict[str, any]:
    """Extract all required sections from patent data."""
    patent_data = load_patent_data(patent_file)
    
    abstract = extract_abstract(patent_data)
    summary = extract_summary_section(patent_data)
    claims = extract_claims_section(patent_data)
    
    return {
        'abstract': abstract,
        'summary': summary,
        'claims': claims
    }


def main():
    """Main function to extract sections."""
    patent_file = "patent.json"
    
    if not Path(patent_file).exists():
        print(f"Error: {patent_file} not found in current directory")
        return
    
    print("Extracting patent sections...")
    sections = extract_all_sections(patent_file)
    
    print(f"\n=== ABSTRACT ===")
    print(f"Length: {len(sections['abstract'])} characters")
    print(f"Content: {sections['abstract'][:300]}...")
    
    print(f"\n=== SUMMARY OF THE INVENTION ===")
    print(f"Length: {len(sections['summary'])} characters")
    print(f"Content: {sections['summary'][:300]}...")
    
    print(f"\n=== CLAIMS ===")
    print(f"Number of claims: {len(sections['claims'])}")
    for i, claim in enumerate(sections['claims'][:3], 1):
        print(f"Claim {i}: {claim[:200]}...")
    
    # Save sections to JSON file
    output_file = "patent_sections.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved sections to: {output_file}")


if __name__ == "__main__":
    main()
