import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab/english.pickle")
except LookupError:  # pragma: no cover - only run on first execution
    nltk.download("punkt")
    nltk.download("punkt_tab")

MAX_WORDS_PER_CHUNK = 60  # chunks should not exceed this size

EMBODIMENT_PATTERNS = [
    r"in (?:one|another|certain|some|particular) embodiment",
    r"according to (?:the|one) (?:embodiment|implementation)",
    r"the (?:method|composition|device) comprises",
]
EMBODIMENT_RE = re.compile("|".join(EMBODIMENT_PATTERNS), re.IGNORECASE)

PRINCIPLE_KEYWORDS = {
    "P1": ["alcoholic liver disease", "graft-versus-host", "gvhd"],
    "P2": ["hyperimmunized egg product"],
    "P3": ["immunized", "hyperimmunized", "antigen"],
    "P4": ["extracted", "formulated", "composition"],
    "P5": ["administer"],
    "P6": ["excipient", "stabilizer", "carrier"],
    "P7": ["dose", "intravenous", "oral", "injection", "route"],
    "P8": ["igy", "egg yolk", "whole egg", "egg white"],
}

SECTIONS_TO_LABEL = {"summary of invention", "detailed description", "claims"}

def chunk_text(text: str) -> List[str]:
    """Split text into chunks not exceeding ``MAX_WORDS_PER_CHUNK`` words."""
    sentences = nltk.sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    for sent in sentences:
        words = sent.split()
        while words:
            space_left = MAX_WORDS_PER_CHUNK - current_len
            if len(words) <= space_left:
                current.extend(words)
                current_len += len(words)
                words = []
            else:
                current.extend(words[:space_left])
                words = words[space_left:]
                current_len += space_left
                flush_current()
    flush_current()
    return [c for c in chunks if c.strip()]

def map_principles(chunk: str) -> List[str]:
    chunk_lower = chunk.lower()
    matched: List[str] = []
    for pid, keywords in PRINCIPLE_KEYWORDS.items():
        for kw in keywords:
            if kw in chunk_lower:
                matched.append(pid)
                break
    return matched

def label_chunk(chunk: str) -> Dict:
    lower = chunk.lower()
    is_embodiment = bool(EMBODIMENT_RE.search(lower))
    justification = (
        "contains embodiment keywords" if is_embodiment else "no embodiment keywords"
    )
    principles = map_principles(chunk)
    return {
        "chunk": chunk,
        "is_embodiment": is_embodiment,
        "justification": justification,
        "mapped_principles": principles,
        "mapped_claims": [],
    }

def process_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    chunks_col: List[List[str]] = []
    labels_col: List[List[Dict]] = []
    for _, row in df.iterrows():
        if str(row.get("section", "")).strip().lower() in SECTIONS_TO_LABEL:
            chunks = chunk_text(str(row.get("text", "")))
            labels = [label_chunk(c) for c in chunks]
            chunks_col.append(chunks)
            labels_col.append(labels)
        else:
            chunks_col.append([])
            labels_col.append([])
    df["chunks"] = chunks_col
    df["embodiment_labels"] = [json.dumps(l, ensure_ascii=False) for l in labels_col]
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Label patent segments with embodiment information")
    parser.add_argument("csv", type=Path, help="Path to ALD_GvHD_patent_segments.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    df = process_csv(args.csv)
    out_path = args.output or args.csv
    df.to_csv(out_path, index=False)
    print(f"Wrote labeled CSV to {out_path}")

if __name__ == "__main__":
    main()
