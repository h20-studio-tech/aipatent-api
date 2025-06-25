#!/usr/bin/env python3
"""
Extract pages 5-38 from patent.json and parse them into punctuation-aware chunks.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path


def load_patent_data(file_path: str) -> Dict[str, Any]:
    """Load the patent JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_pages_5_to_38(patent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract pages 5 through 38 from the patent data."""
    pages = patent_data.get('pages', [])
    target_pages = []
    
    for page in pages:
        page_num = page.get('page', 0)
        if 5 <= page_num <= 38:
            target_pages.append(page)
    
    return target_pages


def punctuation_aware_chunk(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks that respect punctuation boundaries.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum number of characters per chunk
        
    Returns:
        List of text chunks that end at natural punctuation boundaries
    """
    if not text.strip():
        return []
    
    # Split into sentences using multiple punctuation marks
    sentence_endings = r'[.!?;]\s+'
    sentences = re.split(sentence_endings, text.strip())
    
    # Handle the case where the last sentence doesn't end with punctuation
    if sentences and not re.search(r'[.!?;]\s*$', text.strip()):
        # Add back the punctuation that was removed by split
        matches = list(re.finditer(sentence_endings, text))
        for i, match in enumerate(matches):
            if i < len(sentences) - 1:
                sentences[i] += match.group().strip()
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max_chunk_size
        if current_length + sentence_length > max_chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
    
    # Add the final chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if chunk.strip()]


def extract_text_content(page: Dict[str, Any]) -> Dict[str, str]:
    """Extract both raw text and markdown content from a page."""
    return {
        'text': page.get('text', ''),
        'md': page.get('md', ''),
        'page_number': page.get('page', 0)
    }


def process_pages_to_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process pages into punctuation-aware chunks."""
    all_chunks = []
    
    for page in pages:
        page_content = extract_text_content(page)
        page_num = page_content['page_number']
        
        # Prefer markdown content if available, fallback to raw text
        content = page_content['md'] if page_content['md'].strip() else page_content['text']
        
        if not content.strip():
            continue
            
        # Create punctuation-aware chunks
        chunks = punctuation_aware_chunk(content, max_chunk_size=500)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'page_number': page_num,
                'chunk_index': i,
                'chunk_id': f"page_{page_num}_chunk_{i}",
                'text': chunk,
                'content_type': 'markdown' if page_content['md'].strip() else 'raw_text',
                'word_count': len(chunk.split()),
                'char_count': len(chunk)
            }
            all_chunks.append(chunk_data)
    
    return all_chunks


def save_chunks_to_files(chunks: List[Dict[str, Any]], output_dir: str = "."):
    """Save chunks to JSON and CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = output_path / "pages_5_38_chunks.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print("Saved {} chunks to:".format(len(chunks)))
    print("  JSON: {}".format(json_file))
    
    # Save as CSV if pandas is available
    try:
        import pandas as pd
        csv_file = output_path / "pages_5_38_chunks.csv"
        df = pd.DataFrame(chunks)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print("  CSV: {}".format(csv_file))
    except ImportError:
        print("  CSV: Skipped (pandas not available)")
        # Create a simple CSV manually
        csv_file = output_path / "pages_5_38_chunks.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if chunks:
                # Write header
                headers = list(chunks[0].keys())
                f.write(','.join(headers) + '\n')
                
                # Write data
                for chunk in chunks:
                    row = []
                    for header in headers:
                        value = str(chunk.get(header, '')).replace(',', ';').replace('\n', ' ')
                        row.append('"{}"'.format(value))
                    f.write(','.join(row) + '\n')
        print("  CSV: {}".format(csv_file) + " (manual format)")


def analyze_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Print analysis of the chunks."""
    if not chunks:
        print("No chunks found.")
        return
    
    total_chunks = len(chunks)
    total_words = sum(chunk['word_count'] for chunk in chunks)
    total_chars = sum(chunk['char_count'] for chunk in chunks)
    
    pages_covered = set(chunk['page_number'] for chunk in chunks)
    
    avg_words = total_words / total_chunks if total_chunks > 0 else 0
    avg_chars = total_chars / total_chunks if total_chunks > 0 else 0
    
    print("\n=== Chunk Analysis ===")
    print("Total chunks: {}".format(total_chunks))
    print("Pages covered: {}".format(sorted(pages_covered)))
    print("Page range: {} - {}".format(min(pages_covered), max(pages_covered)))
    print("Total words: {:,}".format(total_words))
    print("Total characters: {:,}".format(total_chars))
    print("Average words per chunk: {:.1f}".format(avg_words))
    print("Average characters per chunk: {:.1f}".format(avg_chars))
    
    # Show content type distribution
    content_types = {}
    for chunk in chunks:
        ct = chunk['content_type']
        content_types[ct] = content_types.get(ct, 0) + 1
    
    print("\nContent type distribution:")
    for ct, count in content_types.items():
        print("  {}: {} chunks ({:.1f}%)".format(ct, count, count/total_chunks*100))


def main():
    """Main function to process pages 5-38."""
    patent_file = "patent.json"
    
    if not Path(patent_file).exists():
        print("Error: {} not found in current directory".format(patent_file))
        return
    
    print("Loading patent data...")
    patent_data = load_patent_data(patent_file)
    
    print("Extracting pages 5-38...")
    target_pages = extract_pages_5_to_38(patent_data)
    
    if not target_pages:
        print("No pages found in range 5-38")
        return
    
    print("Found {} pages in range 5-38".format(len(target_pages)))
    
    print("Processing pages into punctuation-aware chunks...")
    chunks = process_pages_to_chunks(target_pages)
    
    if not chunks:
        print("No chunks generated")
        return
    
    # Analyze the chunks
    analyze_chunks(chunks)
    
    # Save to files
    save_chunks_to_files(chunks)
    
    # Show sample chunks
    print("\n=== Sample Chunks ===")
    for i, chunk in enumerate(chunks[:3]):
        print("\nChunk {} (Page {}):".format(i+1, chunk['page_number']))
        print("ID: {}".format(chunk['chunk_id']))
        print("Words: {}, Chars: {}".format(chunk['word_count'], chunk['char_count']))
        print("Content: {}...".format(chunk['text'][:200]))


if __name__ == "__main__":
    main()
