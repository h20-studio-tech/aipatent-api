"""Helper utilities for debugging character position extraction."""

import re
from typing import Optional, Tuple


def show_text_with_positions(text: str, highlight_start: Optional[int] = None, 
                           highlight_end: Optional[int] = None, context_chars: int = 50) -> str:
    """
    Display text with character positions for debugging.
    
    Args:
        text: The source text
        highlight_start: Start position to highlight
        highlight_end: End position to highlight
        context_chars: Number of characters to show before/after highlight
    
    Returns:
        Formatted string showing positions and highlighted section
    """
    lines = []
    
    if highlight_start is not None and highlight_end is not None:
        # Show context around the highlighted region
        start_context = max(0, highlight_start - context_chars)
        end_context = min(len(text), highlight_end + context_chars)
        
        lines.append(f"\n--- Text excerpt [{start_context}:{end_context}] ---")
        
        # Build position ruler
        ruler = []
        text_line = []
        
        for i in range(start_context, end_context):
            char = text[i] if i < len(text) else ''
            
            # Handle newlines and tabs
            if char == '\n':
                char = '↵'
            elif char == '\t':
                char = '→'
            
            text_line.append(char)
            
            # Add position markers
            if i == highlight_start:
                ruler.append('[')
            elif i == highlight_end - 1:
                ruler.append(']')
            elif i % 10 == 0:
                ruler.append('|')
            else:
                ruler.append(' ')
        
        # Position numbers every 10 characters
        pos_line = []
        for i in range(start_context, end_context):
            if i % 10 == 0:
                pos_str = str(i)
                pos_line.extend(pos_str)
                # Skip ahead
                for _ in range(len(pos_str) - 1):
                    if i + len(pos_str) - 1 < end_context:
                        i += 1
            else:
                pos_line.append(' ')
        
        lines.append('Pos: ' + ''.join(pos_line[:len(text_line)]))
        lines.append('     ' + ''.join(ruler))
        lines.append('Text: ' + ''.join(text_line))
        
        # Show the extracted text
        if highlight_start < len(text) and highlight_end <= len(text):
            extracted = text[highlight_start:highlight_end]
            lines.append(f"\nExtracted [{highlight_start}:{highlight_end}]: '{extracted}'")
    else:
        # Show full text with position markers
        lines.append("\n--- Full text with positions ---")
        for i in range(0, len(text), 100):
            chunk = text[i:i+100]
            lines.append(f"[{i:4d}] {repr(chunk)}")
    
    return '\n'.join(lines)


def find_text_position(source_text: str, target_text: str, 
                      case_sensitive: bool = True) -> Tuple[int, int]:
    """
    Find the character positions of target text within source text.
    
    Args:
        source_text: The text to search in
        target_text: The text to find
        case_sensitive: Whether to match case
    
    Returns:
        Tuple of (start_position, end_position) or (-1, -1) if not found
    """
    if not case_sensitive:
        source_lower = source_text.lower()
        target_lower = target_text.lower()
        pos = source_lower.find(target_lower)
    else:
        pos = source_text.find(target_text)
    
    if pos != -1:
        return pos, pos + len(target_text)
    
    return -1, -1


def validate_position_extraction(page_text: str, embodiment_text: str, 
                               start_char: int, end_char: int) -> dict:
    """
    Validate that the character positions correctly extract the embodiment text.
    
    Returns:
        Dictionary with validation results and debugging info
    """
    result = {
        'valid': False,
        'extracted_text': '',
        'expected_text': embodiment_text,
        'error': None,
        'debug_view': ''
    }
    
    try:
        # Extract text using positions
        extracted = page_text[start_char:end_char]
        result['extracted_text'] = extracted
        
        # Check exact match
        if extracted == embodiment_text:
            result['valid'] = True
        else:
            # Check with whitespace normalization
            if extracted.strip() == embodiment_text.strip():
                result['valid'] = True
                result['error'] = 'Minor whitespace difference'
            else:
                result['error'] = 'Text mismatch'
                
                # Try to find the correct position
                correct_start, correct_end = find_text_position(page_text, embodiment_text)
                if correct_start != -1:
                    result['error'] += f' - Correct position: [{correct_start}:{correct_end}]'
        
        # Add debug view
        result['debug_view'] = show_text_with_positions(
            page_text, start_char, end_char, context_chars=30
        )
        
    except Exception as e:
        result['error'] = f'Exception: {str(e)}'
    
    return result


def create_position_test_cases(page_text: str) -> list:
    """
    Generate test cases for position extraction from a page of text.
    
    Looks for common embodiment patterns and creates test cases.
    """
    test_cases = []
    
    # Common embodiment patterns
    patterns = [
        r'In certain aspects[^.]+\.',
        r'In certain embodiments[^.]+\.',
        r'In some embodiments[^.]+\.',
        r'The method[^.]+\.',
        r'The composition[^.]+\.',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, page_text, re.IGNORECASE)
        for match in matches:
            test_cases.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'pattern': pattern
            })
    
    return test_cases