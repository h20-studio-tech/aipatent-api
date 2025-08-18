# Frontend Citation Implementation Guide

## Overview
The API now returns responses with inline citations in the format `[44]` where the number corresponds to a `chunk_id` in the response data array.

## API Response Structure
```json
{
  "status": "success",
  "message": "Streptococcus mutans causes tooth decay [58]. IgY antibodies reduce bacterial adhesion [57] and improve growth in livestock [36].",
  "data": [
    {
      "chunk_id": 58,
      "text": "...",
      "page_number": 10,
      "filename": "document.pdf"
    },
    // ... more chunks
  ]
}
```

## Frontend Implementation Steps

### 1. Parse Citations from Message

```javascript
function parseCitations(message) {
  const citationRegex = /\[(\d+)\]/g;
  const citations = [];
  let match;
  
  while ((match = citationRegex.exec(message)) !== null) {
    citations.push({
      fullMatch: match[0],     // "[44]"
      chunkId: parseInt(match[1]), // 44
      position: match.index     // position in string
    });
  }
  
  return citations;
}
```

### 2. Create Citation Links/Tooltips

```javascript
function renderMessageWithCitations(message, chunks) {
  // Create a map for quick chunk lookup
  const chunkMap = {};
  chunks.forEach(chunk => {
    chunkMap[chunk.chunk_id] = chunk;
  });
  
  // Replace citations with interactive elements
  return message.replace(/\[(\d+)\]/g, (match, chunkId) => {
    const chunk = chunkMap[parseInt(chunkId)];
    if (!chunk) return match; // Keep original if chunk not found
    
    return `<sup class="citation" 
                 data-chunk-id="${chunkId}"
                 data-filename="${chunk.filename}"
                 data-page="${chunk.page_number}"
                 title="Source: ${chunk.filename}, Page ${chunk.page_number}">
              [${chunkId}]
            </sup>`;
  });
}
```

### 3. Handle Citation Clicks

```javascript
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('citation')) {
    const chunkId = parseInt(e.target.dataset.chunkId);
    const chunk = findChunkById(chunkId);
    
    // Options:
    // 1. Show modal with full chunk text
    showChunkModal(chunk);
    
    // 2. Scroll to chunk in sidebar
    scrollToChunk(chunkId);
    
    // 3. Highlight relevant text
    highlightChunkText(chunk);
  }
});
```

### 4. Create Footnotes Section

```javascript
function generateFootnotes(message, chunks) {
  const citations = parseCitations(message);
  const uniqueChunkIds = [...new Set(citations.map(c => c.chunkId))];
  
  const footnotes = uniqueChunkIds.map(chunkId => {
    const chunk = chunks.find(c => c.chunk_id === chunkId);
    return {
      id: chunkId,
      filename: chunk.filename,
      page: chunk.page_number,
      preview: chunk.text.substring(0, 200) + '...'
    };
  });
  
  return footnotes;
}
```

### 5. React/Vue/Angular Examples

#### React Component
```jsx
function CitedMessage({ message, chunks }) {
  const renderMessage = () => {
    const parts = message.split(/(\[\d+\])/g);
    
    return parts.map((part, index) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const chunkId = parseInt(match[1]);
        const chunk = chunks.find(c => c.chunk_id === chunkId);
        
        return (
          <Citation 
            key={index}
            chunkId={chunkId}
            chunk={chunk}
            onClick={() => handleCitationClick(chunk)}
          />
        );
      }
      return <span key={index}>{part}</span>;
    });
  };
  
  return <div>{renderMessage()}</div>;
}
```

#### Vue Component
```vue
<template>
  <div v-html="formattedMessage"></div>
</template>

<script>
export default {
  props: ['message', 'chunks'],
  computed: {
    formattedMessage() {
      return this.message.replace(/\[(\d+)\]/g, (match, chunkId) => {
        const chunk = this.chunks.find(c => c.chunk_id === parseInt(chunkId));
        return `<span class="citation" @click="showChunk(${chunkId})">[${chunkId}]</span>`;
      });
    }
  }
}
</script>
```

## Styling Suggestions

```css
.citation {
  color: #0066cc;
  cursor: pointer;
  font-size: 0.85em;
  padding: 0 2px;
  text-decoration: none;
  transition: all 0.2s;
}

.citation:hover {
  background-color: #e6f2ff;
  border-radius: 3px;
}

.citation-tooltip {
  position: absolute;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  max-width: 300px;
  z-index: 1000;
}

.footnote-section {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #ddd;
}

.footnote-item {
  margin-bottom: 0.5rem;
  font-size: 0.9em;
}
```

## Advanced Features

### 1. Highlight Source Text on Hover
```javascript
function addCitationHoverEffect() {
  const citations = document.querySelectorAll('.citation');
  
  citations.forEach(citation => {
    citation.addEventListener('mouseenter', (e) => {
      const chunkId = e.target.dataset.chunkId;
      highlightChunkInSidebar(chunkId);
    });
    
    citation.addEventListener('mouseleave', () => {
      clearHighlights();
    });
  });
}
```

### 2. Citation Summary Panel
```javascript
function createCitationSummary(message, chunks) {
  const citations = parseCitations(message);
  const citationGroups = {};
  
  // Group citations by source file
  citations.forEach(citation => {
    const chunk = chunks.find(c => c.chunk_id === citation.chunkId);
    if (!citationGroups[chunk.filename]) {
      citationGroups[chunk.filename] = [];
    }
    citationGroups[chunk.filename].push(chunk);
  });
  
  return citationGroups;
}
```

### 3. Handle Multiple Files (Future Enhancement)
When multiple files have the same chunk_id:
```javascript
function handleDuplicateChunkIds(chunks) {
  const duplicates = {};
  
  chunks.forEach(chunk => {
    if (!duplicates[chunk.chunk_id]) {
      duplicates[chunk.chunk_id] = [];
    }
    duplicates[chunk.chunk_id].push(chunk);
  });
  
  // If duplicates exist, show disambiguation UI
  Object.entries(duplicates).forEach(([chunkId, chunks]) => {
    if (chunks.length > 1) {
      // Add filename to citation display
      // e.g., [44:doc1.pdf] or show modal to select
    }
  });
}
```

## Testing Checklist

- [ ] Citations are correctly parsed from message
- [ ] Each citation links to the correct chunk
- [ ] Tooltips show on hover with source info
- [ ] Click handling works (modal/scroll/highlight)
- [ ] Footnotes section displays correctly
- [ ] Multiple citations to same chunk handled
- [ ] Missing chunk_ids handled gracefully
- [ ] Mobile touch interactions work
- [ ] Accessibility: keyboard navigation works
- [ ] Performance: handles 100+ citations efficiently

## Example Integration

```html
<div class="rag-response">
  <div class="message-content" id="message">
    <!-- Rendered message with citations -->
  </div>
  
  <div class="source-panel" id="sources">
    <!-- Chunk details panel -->
  </div>
  
  <div class="footnotes" id="footnotes">
    <!-- Footnotes section -->
  </div>
</div>

<script>
// On API response
fetch('/api/v1/rag/multiquery-search/')
  .then(res => res.json())
  .then(data => {
    const messageEl = document.getElementById('message');
    messageEl.innerHTML = renderMessageWithCitations(data.message, data.data);
    
    const footnotesEl = document.getElementById('footnotes');
    footnotesEl.innerHTML = renderFootnotes(data.message, data.data);
    
    // Add event listeners
    addCitationHoverEffect();
  });
</script>
```

## Troubleshooting

### Citations Not Appearing
- Check if LLM is generating citations in responses
- Verify regex pattern matches citation format
- Ensure chunk_ids in citations exist in data array

### Performance Issues
- For large responses, consider virtual scrolling for chunks
- Debounce hover effects
- Lazy load chunk details on demand

### Accessibility
- Add ARIA labels to citations
- Ensure keyboard navigation between citations
- Provide text-only fallback option