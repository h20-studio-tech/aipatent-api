#!/usr/bin/env python3
"""
Label ALL chunks in the pages_5_38_chunks.json dataset using the embodiment extractor.
This script will process every single chunk, not just filtered ones.
"""

import json
import asyncio
import re
from typing import Dict, List, Optional
from pathlib import Path

# Import the embodiment extraction functions
import sys
sys.path.append('src')
from embodiment_extraction import (
    extract_principles, 
    classify_chunk, 
    Principle,
    EmbodimentAnnotation
)


def load_chunks_data(file_path: str) -> List[Dict]:
    """Load the chunks JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_patent_data(file_path: str) -> Dict:
    """Load the patent JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


async def process_all_chunks():
    """Main processing function that labels ALL chunks."""
    
    # Define the abstract (provided by user)
    abstract = """In one aspect, the present disclosure is directed to a method for preventing or treating alcoholic liver disease or graft-versus-host disease in a subject in need thereof, comprising administering to the subject a therapeutically effective amount of a hyperimmunized egg product obtained from an egg-producing animal, thereby preventing or treating the alcoholic liver disease or the graft-versus-host disease in the subject, wherein the hyperimmunized egg product comprises a therapeutically effective amount of one or more antibodies to an antigen selected from the group consisting of Enterococcus faecalis, Enterococcus faecalis cytolysin toxin, and Enterococcus faecium. The present disclosure is also directed to hyperimmunized eggs and egg products produced by an animal that has been hyperimmunized with an antigen selected from the group consisting of Enterococcus faecalis, isolated Enterococcus faecalis cytolysin toxin, and Enterococcus faecium. Methods of preparing the hyperimmunized eggs and egg products are also disclosed."""
    
    # Load patent data to extract summary and claims
    print("Loading patent data...")
    patent_data = load_patent_data("patent.json")
    
    print("Extracting summary and claims...")
    summary = extract_summary_section(patent_data)
    claims = extract_claims_section(patent_data)
    
    print("Summary length: {} characters".format(len(summary)))
    print("Number of claims: {}".format(len(claims)))
    
    # Load chunks data
    print("Loading chunks data...")
    chunks_data = load_chunks_data("pages_5_38_chunks.json")
    print("Loaded {} chunks".format(len(chunks_data)))
    
    # Extract principles from abstract, summary, and claims
    print("Extracting invention principles...")
    principles = await extract_principles(abstract, summary, claims)
    print("Extracted {} principles:".format(len(principles)))
    for i, principle in enumerate(principles):
        print("  P{}: {}...".format(i+1, principle.text[:100]))
    
    # Process ALL chunks (no filtering)
    print("Processing ALL chunks for embodiment classification...")
    print("This will process {} total chunks".format(len(chunks_data)))
    
    # Classify each chunk
    print("Starting classification...")
    labeled_chunks = []
    
    for i, chunk_data in enumerate(chunks_data):
        print("Processing chunk {}/{}...".format(i+1, len(chunks_data)))
        
        chunk_text = chunk_data['text']
        
        # Classify the chunk
        annotation = await classify_chunk(chunk_text, principles)
        
        # Create labeled chunk with metadata
        labeled_chunk = {
            # Original chunk metadata
            'chunk_id': chunk_data['chunk_id'],
            'page_number': chunk_data['page_number'],
            'chunk_index': chunk_data['chunk_index'],
            'word_count': chunk_data['word_count'],
            'char_count': chunk_data['char_count'],
            'content_type': chunk_data['content_type'],
            
            # Embodiment classification
            'text': annotation.paragraph,
            'is_embodiment': annotation.is_embodiment,
            'justification': annotation.justification,
            'mapped_principles': annotation.mapped_principles,
            'mapped_claims': annotation.mapped_claims,
            
            # Additional metadata
            'source_section': 'detailed_description',  # Pages 5-38 are typically detailed description
            'processing_timestamp': '2025-06-22T20:13:29-04:00'
        }
        
        labeled_chunks.append(labeled_chunk)
    
    return labeled_chunks, principles


def save_results(labeled_chunks: List[Dict], principles: List[Principle]):
    """Save the labeled dataset and principles."""
    
    # Save labeled chunks
    output_file = "labeled_all_chunks_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_chunks, f, indent=2, ensure_ascii=False)
    
    # Save as JSONL for easier processing
    jsonl_file = "labeled_all_chunks_dataset.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for chunk in labeled_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Save principles separately
    principles_file = "invention_principles.json"
    principles_data = [p.model_dump() for p in principles]
    with open(principles_file, 'w', encoding='utf-8') as f:
        json.dump(principles_data, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved:")
    print("  Labeled chunks (JSON): {}".format(output_file))
    print("  Labeled chunks (JSONL): {}".format(jsonl_file))
    print("  Invention principles: {}".format(principles_file))


def analyze_results(labeled_chunks: List[Dict]):
    """Analyze and print statistics about the labeled dataset."""
    total_chunks = len(labeled_chunks)
    embodiment_chunks = sum(1 for chunk in labeled_chunks if chunk['is_embodiment'])
    non_embodiment_chunks = total_chunks - embodiment_chunks
    
    # Count principle mappings
    principle_counts = {}
    for chunk in labeled_chunks:
        for principle in chunk['mapped_principles']:
            principle_counts[principle] = principle_counts.get(principle, 0) + 1
    
    # Count claim mappings
    claim_counts = {}
    for chunk in labeled_chunks:
        for claim in chunk['mapped_claims']:
            claim_counts[claim] = claim_counts.get(claim, 0) + 1
    
    print("\n=== LABELING RESULTS ===")
    print("Total chunks processed: {}".format(total_chunks))
    print("Embodiment chunks: {} ({:.1f}%)".format(embodiment_chunks, embodiment_chunks/total_chunks*100))
    print("Non-embodiment chunks: {} ({:.1f}%)".format(non_embodiment_chunks, non_embodiment_chunks/total_chunks*100))
    
    print("\nPrinciple mapping frequency:")
    for principle, count in sorted(principle_counts.items()):
        print("  {}: {} chunks".format(principle, count))
    
    print("\nClaim mapping frequency:")
    for claim, count in sorted(claim_counts.items()):
        print("  Claim {}: {} chunks".format(claim, count))
    
    # Show sample embodiments
    print("\n=== SAMPLE EMBODIMENTS ===")
    embodiments = [chunk for chunk in labeled_chunks if chunk['is_embodiment']]
    for i, chunk in enumerate(embodiments[:3]):
        print("\nEmbodiment {}:".format(i+1))
        print("  Page: {}".format(chunk['page_number']))
        print("  Principles: {}".format(chunk['mapped_principles']))
        print("  Claims: {}".format(chunk['mapped_claims']))
        print("  Text: {}...".format(chunk['text'][:200]))
        print("  Justification: {}".format(chunk['justification']))


async def main():
    """Main function."""
    print("Starting embodiment extraction pipeline for ALL chunks...")
    
    # Check required files
    if not Path("pages_5_38_chunks.json").exists():
        print("Error: pages_5_38_chunks.json not found")
        return
    
    if not Path("patent.json").exists():
        print("Error: patent.json not found")
        return
    
    try:
        # Process the dataset
        labeled_chunks, principles = await process_all_chunks()
        
        # Save results
        save_results(labeled_chunks, principles)
        
        # Analyze results
        analyze_results(labeled_chunks)
        
        print("\n✅ Embodiment extraction completed successfully!")
        print("All {} chunks have been labeled!".format(len(labeled_chunks)))
        
    except Exception as e:
        print("❌ Error during processing: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
