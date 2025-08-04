# Database Architecture

This document describes the Supabase (PostgreSQL) schema for the aipatent-api project. It covers tables, columns, relationships, and usage notes inferred from the diagram.

## Overview

Entities:
- patent_files: One record per uploaded patent file.
- pages: OCR/text content for each page in a patent file.
- raw_sections: Detected raw sections (e.g., claims, description) across page ranges.
- glossary_terms: Glossary entries extracted from a file.
- embodiments: Parsed “embodiments” segments within a file.

Core relationship:
- patent_files (1) ──< pages, raw_sections, glossary_terms, embodiments (many)
- All children reference patent_files.id via file_id.

---

## Tables

### 1) patent_files
Stores metadata about uploaded patent documents.

Columns:
- id (uuid, PK)
- filename (text, not null): Original file name.
- mime_type (text, not null): Detected content type (e.g., application/pdf).
- uploaded_at (timestamptz, default now()): Upload timestamp.
- abstract (text, nullable): Extracted abstract text if available.
- sections (jsonb, nullable): Normalized section map, e.g. { "claims": {...}, "description": {...} }.

Indexes and Constraints:
- PK on (id).
- Consider index on (uploaded_at) for recent uploads listing.

Typical operations:
- Insert new file on upload.
- Store parsed abstract and structured sections after processing.

Example:
```sql
-- Create
insert into patent_files (id, filename, mime_type)
values (gen_random_uuid(), 'US_1234567_A1.pdf', 'application/pdf')
returning *;

-- List recent
select id, filename, uploaded_at
from patent_files
order by uploaded_at desc
limit 20;
```

---

### 2) pages
Stores per-page text for each patent file.

Columns:
- id (int8/bigint, PK)
- file_id (uuid, not null, FK → patent_files.id)
- page_number (int4, not null): 1-based page index.
- text (text, nullable): Page OCR or extracted text.
- section (text, nullable): Optional coarse section label for the page.
- filename (text, nullable): Source page image or derived file reference.
- created_at (timestamptz, default now())

Indexes and Constraints:
- Unique(file_id, page_number) recommended.
- Foreign key with on delete cascade.

Queries:
```sql
-- Get all pages in order
select page_number, text
from pages
where file_id = $1
order by page_number;

-- Backfill index
create unique index if not exists pages_file_page_uq
on pages(file_id, page_number);
```

---

### 3) raw_sections
Represents contiguous section spans identified in a document.

Columns:
- id (int8/bigint, PK)
- file_id (uuid, not null, FK → patent_files.id)
- section_type (text, not null): e.g., 'claims', 'description', 'summary', 'figures'.
- text (text, nullable): Concatenated text across pages in the span.
- page_start (int4, not null)
- page_end (int4, not null)
- created_at (timestamptz, default now())
- updated_at (timestamptz, default now())

Indexes and Constraints:
- Check(page_start <= page_end).
- Index(file_id, section_type).
- Consider ensuring no overlapping ranges per section_type.

Queries:
```sql
-- Sections for a file
select section_type, page_start, page_end
from raw_sections
where file_id = $1
order by page_start;

-- Claims block
select text
from raw_sections
where file_id = $1 and section_type = 'claims'
limit 1;
```

---

### 4) glossary_terms
Glossary items detected in the patent.

Columns:
- id (uuid, PK)
- file_id (uuid, not null, FK → patent_files.id)
- term (text, not null)
- definition (text, nullable)
- page_number (int4, nullable): Source page where the term/definition occurs.
- created_at (timestamptz, default now())

Indexes and Constraints:
- Unique(file_id, term) recommended (case-insensitive if desired).
- Index(file_id, page_number).

Queries:
```sql
-- Find a definition
select term, definition
from glossary_terms
where file_id = $1 and term ilike $2;

-- All glossary for a file
select term, definition, page_number
from glossary_terms
where file_id = $1
order by term;
```

---

### 5) embodiments
Captures “embodiment” sections, often enumerated in specifications.

Columns:
- id (int8/bigint, PK)
- file_id (uuid, not null, FK → patent_files.id)
- emb_number (int4, nullable): Enumeration index if parsed.
- text (text, not null): Embodiment body text.
- page_number (int4, nullable): Page where the embodiment starts.
- section (text, nullable): Parent section label.
- sub_category (text, nullable): Optional subtype or tag.
- summary (text, nullable): Extracted one- or multi-sentence summary.
- header (text, nullable): Heading text if present.

Indexes and Constraints:
- Index(file_id, emb_number).
- Index(file_id, page_number).

Queries:
```sql
-- List embodiments with summaries
select emb_number, header, summary, page_number
from embodiments
where file_id = $1
order by coalesce(emb_number, 999999), page_number;

-- Example full text search (if configured)
-- create index embodiments_text_idx on embodiments using gin (to_tsvector('english', text));
```

---

## Relationships

- patent_files.id → pages.file_id (1:N, cascade delete recommended)
- patent_files.id → raw_sections.file_id (1:N)
- patent_files.id → glossary_terms.file_id (1:N)
- patent_files.id → embodiments.file_id (1:N)

Cardinality:
- A single patent_files row has many pages, raw_sections, glossary_terms, and embodiments.
- Children rows belong to exactly one patent_file via file_id.

Example join:
```sql
-- Gather a file with its abstract, sections, and counts
select pf.id,
       pf.filename,
       pf.abstract,
       jsonb_pretty(pf.sections) as sections,
       (select count(*) from pages p where p.file_id = pf.id) as page_count,
       (select count(*) from glossary_terms g where g.file_id = pf.id) as glossary_count,
       (select count(*) from embodiments e where e.file_id = pf.id) as embodiment_count
from patent_files pf
where pf.id = $1;
```

---

## Suggested Constraints and Policies

- On delete cascade from patent_files to children:
```sql
alter table pages
  add constraint pages_file_fk
  foreign key (file_id) references patent_files(id) on delete cascade;

alter table raw_sections
  add constraint raw_sections_file_fk
  foreign key (file_id) references patent_files(id) on delete cascade;

alter table glossary_terms
  add constraint glossary_terms_file_fk
  foreign key (file_id) references patent_files(id) on delete cascade;

alter table embodiments
  add constraint embodiments_file_fk
  foreign key (file_id) references patent_files(id) on delete cascade;
```

- Recommended uniqueness:
```sql
create unique index if not exists pages_file_page_uq
  on pages(file_id, page_number);

create unique index if not exists glossary_terms_file_term_uq
  on glossary_terms(file_id, lower(term));
```

- Common performance indexes:
```sql
create index if not exists pages_file_idx on pages(file_id);
create index if not exists raw_sections_file_type_idx on raw_sections(file_id, section_type);
create index if not exists embodiments_file_page_idx on embodiments(file_id, page_number);
```

---

## Ingestion and Processing Flow

1) Insert into patent_files upon upload.
2) Extract pages → insert into pages with page_number and text.
3) Detect raw section spans → insert into raw_sections with page ranges and text.
4) Parse glossary terms and definitions → insert into glossary_terms.
5) Parse numbered embodiments → insert into embodiments, with summaries if available.
6) Optionally store normalized sections summary into patent_files.sections and abstract.

---

## Data Quality Notes

- Ensure page_number starts at 1 and is continuous for a file.
- raw_sections page ranges should not overlap for the same section_type unless intentional.
- Use text normalization (lower/strip punctuation) when enforcing unique glossary terms.
- Consider full-text search indexes on pages.text, raw_sections.text, embodiments.text for retrieval.