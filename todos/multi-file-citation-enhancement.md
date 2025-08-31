# Multi-File Citation Enhancement

## Current Issue
When users select multiple files in the RAG query interface, chunk_ids can collide between different documents. For example:
- File A might have chunk_id: 44
- File B might also have chunk_id: 44
- The citation `[chunk_id: 44]` becomes ambiguous

## Current Implementation (Phase 1)
Simple inline citations using format: `[chunk_id: X]`
- Works well for single file queries
- Chunk_ids are unique within a single document
- Clean, low output without filename clutter

## Future Enhancement Options

### Option 1: Compound Citations
Include both chunk_id and filename when multiple files are queried:
```
Format: [chunk_id: 44, file: document.pdf]
```

**Pros:**
- Unambiguous citations
- Clear source attribution

**Cons:**
- Cluttered output with repeated filenames
- Longer citations interrupt reading flow

### Option 2: Unique Chunk Identifiers
Generate globally unique chunk identifiers during processing:
```
Format: [chunk_id: doc1_44] or [chunk_id: hash_44]
```

**Pros:**
- Still concise citations
- No ambiguity

**Cons:**
- Requires modifying chunk storage/retrieval
- Backward compatibility concerns

### Option 3: Post-Processing Approach
Keep simple `[chunk_id: X]` format but add filename only when ambiguous:
- Backend or frontend detects duplicate chunk_ids across files
- Automatically expands ambiguous citations to include filename
- Non-ambiguous citations remain simple

**Pros:**
- Clean output for single-file queries
- Handles multi-file gracefully
- No changes to existing single-file workflows

**Cons:**
- Requires post-processing logic
- Additional complexity in citation parsing

## Recommended Approach
**Option 3** with backend post-processing:

1. LLM continues to generate simple `[chunk_id: X]` citations
2. Backend checks for chunk_id collisions in the returned chunks
3. If collision detected:
   - Expand citation to `[chunk_id: X, file: shortened_name]`
   - Or map to unique identifiers in response metadata
4. Return enhanced citation mapping in API response

## Implementation Considerations

### Backend Changes Needed:
1. Add collision detection in `multiquery_search` or API endpoint
2. Create citation mapping in response:
```python
{
  "message": "...[chunk_id: 44]...",
  "data": [...chunks...],
  "citation_map": {
    "44": {
      "files": ["doc1.pdf", "doc2.pdf"],
      "disambiguated": true
    }
  }
}
```

### Frontend Changes Needed:
1. Parse citation_map to handle ambiguous references
2. Show filename in tooltip/footnote when disambiguated
3. Optional: Different UI treatment for multi-source citations

## Testing Scenarios
1. Single file query - ensure no regression
2. Multiple files with unique chunk_ids - clean citations
3. Multiple files with colliding chunk_ids - proper disambiguation
4. Edge case: Same chunk_id appears multiple times in single file