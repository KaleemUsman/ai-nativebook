# Implementation Plan: RAG Chatbot

**Spec**: [spec.md](./spec.md) | **Date**: 2025-12-18 | **Status**: Draft

---

## Summary

Implement a Retrieval-Augmented Generation (RAG) chatbot that answers questions from the AI-Native Book content. The system uses FastAPI for the backend, Qdrant for vector storage, and OpenAI for embeddings and completions.

---

## Technical Context

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.10+ |
| **Framework** | FastAPI |
| **Vector DB** | Qdrant (local or cloud) |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **LLM** | OpenAI GPT-3.5-turbo / GPT-4 |
| **Frontend** | React component (Docusaurus plugin) |

---

## Constitution Check

| Principle | Compliance |
|-----------|------------|
| Technical Accuracy | ✅ Grounded responses only |
| Clean/Modular Docusaurus | ✅ Integrates with existing docs |
| Code Correctness | ✅ Type hints, validation |
| AI-Native Creation | ✅ RAG pattern implementation |
| Verification Standards | ✅ Unit + integration tests |

---

## Project Structure

### Documentation
```
specs/005-rag-chatbot/
├── spec.md              ✅ Created
├── plan.md              📝 This file
├── research.md          📋 TODO
├── data-model.md        📋 TODO
├── quickstart.md        📋 TODO
├── tasks.md             📋 TODO
└── contracts/
    └── api.yaml         📋 TODO
```

### Source Code
```
src/rag/
├── __init__.py
├── config.py            # Configuration management
├── embeddings.py        # OpenAI embedding client
├── vectorstore.py       # Qdrant integration
├── retriever.py         # Semantic search
├── generator.py         # Answer generation
├── ingestion.py         # MDX content processor
└── api/
    ├── __init__.py
    ├── main.py          # FastAPI application
    ├── routes.py        # API endpoints
    ├── models.py        # Pydantic schemas
    └── dependencies.py  # Dependency injection

examples/rag-chatbot/
├── config/
│   └── rag_config.yaml  # Default configuration
├── scripts/
│   ├── ingest.py        # Ingestion CLI
│   └── query.py         # Query CLI
└── frontend/
    └── ChatWidget.tsx   # Docusaurus component
```

---

## Proposed Changes

### Phase 1: Core Infrastructure

#### [NEW] `src/rag/config.py`
Configuration management with Pydantic settings:
- Qdrant connection settings
- OpenAI API configuration
- Chunk size and overlap settings

#### [NEW] `src/rag/embeddings.py`
OpenAI embedding client:
- Batch embedding generation
- Retry logic and rate limiting
- Token counting utilities

#### [NEW] `src/rag/vectorstore.py`
Qdrant vector database integration:
- Collection management
- Vector upsert operations
- Similarity search with filters

---

### Phase 2: Content Ingestion

#### [NEW] `src/rag/ingestion.py`
MDX content processor:
- Parse MDX files with frontmatter
- Chunk documents by paragraphs/sections
- Generate embeddings for chunks
- Store in Qdrant with metadata

#### [NEW] `examples/rag-chatbot/scripts/ingest.py`
CLI script for running ingestion:
```bash
python examples/rag-chatbot/scripts/ingest.py \
    --docs-path ./docs \
    --collection ai-native-book
```

---

### Phase 3: Query & Retrieval

#### [NEW] `src/rag/retriever.py`
Semantic search implementation:
- Query embedding generation
- Top-k similarity search
- Result ranking and filtering

#### [NEW] `src/rag/generator.py`
Answer generation with OpenAI:
- System prompt with grounding instructions
- Context injection from retrieved documents
- Source citation extraction
- "I don't know" handling

---

### Phase 4: FastAPI Backend

#### [NEW] `src/rag/api/main.py`
FastAPI application setup:
- CORS middleware
- Health check endpoint
- Error handling

#### [NEW] `src/rag/api/routes.py`
API endpoints:
- `POST /api/chat` - Query the chatbot
- `GET /api/health` - Health check
- `POST /api/ingest` - Trigger ingestion (admin)

#### [NEW] `src/rag/api/models.py`
Pydantic request/response schemas:
- `ChatRequest`, `ChatResponse`
- `Source` with document metadata

---

### Phase 5: Frontend Integration

#### [NEW] `examples/rag-chatbot/frontend/ChatWidget.tsx`
React chat component:
- Floating chat button
- Message input and display
- Source citations with links
- Loading states

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Docusaurus Frontend                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   ChatWidget.tsx                         ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /api/chat
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   routes.py  │──│  retriever   │──│    generator      │  │
│  │              │  │              │  │                   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└──────────────────────────┬─────────────────────┬────────────┘
                           │                     │
            ┌──────────────┴──────────────┐     │
            ▼                             ▼     ▼
┌─────────────────────┐        ┌─────────────────────┐
│      Qdrant         │        │      OpenAI API     │
│  (Vector Database)  │        │  (Embeddings + LLM) │
└─────────────────────┘        └─────────────────────┘
```

---

## Verification Plan

### Automated Tests

#### Unit Tests
```bash
# Run all RAG tests
pytest tests/rag/ -v

# Test specific components
pytest tests/rag/test_embeddings.py -v
pytest tests/rag/test_retriever.py -v
pytest tests/rag/test_generator.py -v
```

#### Integration Tests
```bash
# Start Qdrant locally first
docker run -p 6333:6333 qdrant/qdrant

# Run integration tests
pytest tests/rag/test_integration.py -v
```

### Manual Verification

1. **Ingestion Test**:
   ```bash
   # Ingest all documentation
   python examples/rag-chatbot/scripts/ingest.py --docs-path ./docs
   
   # Verify in Qdrant console: http://localhost:6333/dashboard
   # Expected: Collection "ai-native-book" with 50+ vectors
   ```

2. **Query Test**:
   ```bash
   # Start the API server
   uvicorn src.rag.api.main:app --reload --port 8000
   
   # Test query
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I create a ROS 2 node?"}'
   
   # Expected: Answer with sources referencing ROS 2 docs
   ```

3. **Frontend Test**:
   - Open http://localhost:3000 (Docusaurus)
   - Click on the chat widget (bottom right)
   - Ask: "What is Isaac ROS?"
   - Expected: Grounded answer with link to Isaac AI Brain docs

---

## Dependencies

```txt
# requirements.txt additions
fastapi>=0.109.0
uvicorn>=0.27.0
qdrant-client>=1.7.0
openai>=1.10.0
pydantic>=2.5.0
python-frontmatter>=1.0.0
tiktoken>=0.5.0
httpx>=0.26.0
```

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| API costs | Use ada-002 for embeddings, GPT-3.5 for answers |
| Hallucination | Strict system prompt, require citations |
| Qdrant downtime | Health checks, graceful degradation |
| Large docs | Chunk with overlap, limit context window |

---

## Complexity Tracking

| Component | Complexity | Rationale |
|-----------|------------|-----------|
| Embeddings | Low | Simple OpenAI API calls |
| Vectorstore | Medium | Qdrant setup and queries |
| Retriever | Medium | Ranking and filtering |
| Generator | Medium | Prompt engineering |
| API | Low | Standard FastAPI |
| Frontend | Medium | React component integration |
| **Total** | **Medium** | ~40-60 hours estimated |

---

## Next Steps

1. ✅ Review and approve this plan
2. Create `research.md` with technology decisions
3. Create `data-model.md` with entity definitions
4. Create `tasks.md` with implementation checklist
5. Begin Phase 1 implementation
