# Patent Content Draft Storage Endpoints (AIP-1)

## Overview

This module provides flexible JSONB-based storage for patent draft components with three API endpoints that enable incremental section generation and full draft persistence.

## Features

- **Flexible Schema**: JSONB storage allows components to have varying structures without migrations
- **Independent Module**: No dependencies on `patent_files` table - works with any `patent_id`
- **Version Tracking**: Automatic version incrementing on full saves
- **Incremental Updates**: Add/update components one at a time during generation
- **Full State Persistence**: Save complete draft state with "Save Progress" functionality

## Database Schema

### Table: `patent_content_drafts`

```sql
CREATE TABLE patent_content_drafts (
    id UUID PRIMARY KEY,
    patent_id UUID NOT NULL UNIQUE,
    components JSONB NOT NULL DEFAULT '[]'::jsonb,
    version INTEGER NOT NULL DEFAULT 1,
    last_saved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Component Structure

Each component in the JSONB array:

```json
{
  "id": "uuid",
  "type": "background|claims|abstract|field_of_invention|summary|...",
  "title": "Section Title",
  "content": "Generated text content...",
  "order": 1,
  "generated_at": "2025-10-14T12:00:00Z",
  "trace_id": "langfuse_trace_id",
  "metadata": {
    "model": "gpt-4",
    "prompt_version": "v2"
  }
}
```

## API Endpoints

### 1. PATCH `/api/v1/project/patent/{patent_id}/component`

**Purpose:** Update or insert a single component (called during section generation)

**Request:**
```json
{
  "component_id": "uuid",
  "type": "background",
  "title": "Background Section",
  "content": "Generated content...",
  "order": 1,
  "trace_id": "optional_trace_id",
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Component updated successfully",
  "patent_id": "uuid",
  "component_id": "uuid",
  "updated_at": "2025-10-14T12:00:00Z"
}
```

**Behavior:**
- If component with same `id` or `type` exists → updates it
- Otherwise → appends new component
- Creates draft if doesn't exist for this `patent_id`
- Updates `last_saved_at` timestamp

### 2. POST `/api/v1/project/patent/{patent_id}/save`

**Purpose:** Save complete draft state (called on "Save Progress")

**Request:**
```json
{
  "components": [
    {
      "id": "uuid_1",
      "type": "background",
      "title": "Background",
      "content": "...",
      "order": 1
    },
    {
      "id": "uuid_2",
      "type": "claims",
      "title": "Claims",
      "content": "...",
      "order": 2
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Draft saved successfully",
  "patent_id": "uuid",
  "version": 2,
  "last_saved_at": "2025-10-14T12:00:00Z",
  "components_count": 5
}
```

**Behavior:**
- Replaces entire `components` array
- Increments `version` number
- Upsert operation (creates if not exists)

### 3. GET `/api/v1/project/patent/{patent_id}`

**Purpose:** Retrieve saved draft progress

**Response:**
```json
{
  "status": "success",
  "patent_id": "uuid",
  "components": [...],
  "version": 2,
  "last_saved_at": "2025-10-14T12:00:00Z",
  "created_at": "2025-10-13T10:00:00Z"
}
```

**Error Responses:**
- `404`: No draft found for `patent_id`
- `500`: Server error

## Usage Examples

### Python Client Example

```python
import httpx
import uuid

BASE_URL = "http://localhost:8000"
patent_id = "your-patent-uuid"

# 1. Add component incrementally
async with httpx.AsyncClient() as client:
    response = await client.patch(
        f"{BASE_URL}/api/v1/project/patent/{patent_id}/component",
        json={
            "component_id": str(uuid.uuid4()),
            "type": "background",
            "title": "Background",
            "content": "Generated background text...",
            "order": 1,
            "trace_id": "langfuse_trace_123"
        }
    )

# 2. Save complete draft
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{BASE_URL}/api/v1/project/patent/{patent_id}/save",
        json={
            "components": [
                {"id": "uuid1", "type": "abstract", "title": "Abstract", "content": "...", "order": 0},
                {"id": "uuid2", "type": "background", "title": "Background", "content": "...", "order": 1}
            ]
        }
    )

# 3. Retrieve draft
async with httpx.AsyncClient() as client:
    response = await client.get(
        f"{BASE_URL}/api/v1/project/patent/{patent_id}"
    )
    draft = response.json()
    print(f"Version: {draft['version']}")
    print(f"Components: {len(draft['components'])}")
```

### cURL Examples

```bash
# Add component
curl -X PATCH "http://localhost:8000/api/v1/project/patent/{patent_id}/component" \
  -H "Content-Type: application/json" \
  -d '{
    "component_id": "uuid",
    "type": "background",
    "title": "Background",
    "content": "Generated content...",
    "order": 1
  }'

# Save draft
curl -X POST "http://localhost:8000/api/v1/project/patent/{patent_id}/save" \
  -H "Content-Type: application/json" \
  -d '{
    "components": [...]
  }'

# Get draft
curl "http://localhost:8000/api/v1/project/patent/{patent_id}"
```

## Testing

Run the manual test suite:

```bash
# Start the API server
uv run uvicorn src.main:app --reload

# In another terminal, run tests
uv run python tests/manual/test_patent_draft_endpoints.py
```

The test suite covers:
1. Incremental component updates (PATCH)
2. Multiple component additions
3. Draft retrieval (GET)
4. Full draft save (POST)
5. Version tracking
6. 404 handling for non-existent drafts

## Design Decisions

### Why JSONB?

1. **Schema Flexibility**: Component types and structures can evolve without migrations
2. **Simple Updates**: Entire array replacement is straightforward
3. **Query Capability**: PostgreSQL GIN indexes enable efficient JSONB queries
4. **Atomic Operations**: Single-row updates ensure consistency
5. **Future-Proof**: Easy to add new component fields without breaking changes

### Why Independent Module?

1. **No Coupling**: Works with `patent_id` from any source (DynamoDB, Supabase, etc.)
2. **Simple Maintenance**: No cascade delete concerns or foreign key constraints
3. **Flexible Usage**: Can be reused across different patent management flows

## Performance Considerations

- GIN index on `components` enables efficient JSONB queries
- B-tree index on `patent_id` for fast lookups
- `UNIQUE` constraint on `patent_id` ensures one draft per patent

## Future Enhancements

Potential improvements:
- Add draft deletion endpoint
- Implement draft history/snapshots
- Add component-level timestamps
- Support draft templates
- Add collaborative editing with conflict resolution
