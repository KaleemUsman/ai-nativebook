# Tasks: RAG Chatbot – Retrieval-Augmented Humanoid AI Assistant

**Branch**: `005-rag-chatbot` | **Date**: 2025-12-18 | **Plan**: [plan.md](./plan.md)
**Status**: Ready for Implementation

## Overview

Implementation checklist for Module 5: RAG Chatbot. A knowledge-grounded conversational agent that answers questions strictly from AI-Native Book content.

---

## Phase 1: Environment Setup

**Goal**: Prepare development environment for RAG implementation.
**Estimated Duration**: 1-2 hours

### 1.1 Python Environment

- [ ] **1.1.1** Install Python 3.10+ environment
  ```bash
  python --version  # Should be 3.10+
  python -m venv .venv
  source .venv/bin/activate
  ```
- [ ] **1.1.2** Install FastAPI and web dependencies
  ```bash
  pip install fastapi uvicorn httpx pydantic
  ```
- [ ] **1.1.3** Install Qdrant client
  ```bash
  pip install qdrant-client
  ```
- [ ] **1.1.4** Install OpenAI SDK
  ```bash
  pip install openai tiktoken
  ```
- [ ] **1.1.5** Install Neon runtime (optional)
  ```bash
  pip install neon-py  # Optional for hosting
  ```
- **Acceptance**: All packages import without error

### 1.2 Project Structure

- [ ] **1.2.1** Create `src/rag/` package directory
- [ ] **1.2.2** Create `examples/rag-chatbot/` directory
- [ ] **1.2.3** Create `tests/rag/` test directory
- [ ] **1.2.4** Set up environment variables
  ```bash
  export OPENAI_API_KEY="your-key-here"
  export QDRANT_URL="http://localhost:6333"
  ```
- **Acceptance**: Project structure matches plan.md

### 1.3 Qdrant Setup

- [ ] **1.3.1** Start Qdrant with Docker
  ```bash
  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
  ```
- [ ] **1.3.2** Verify Qdrant dashboard
  - Open http://localhost:6333/dashboard
- **Acceptance**: Qdrant dashboard accessible

---

## Phase 2: Data Ingestion 🎯 MVP

**Goal**: Ingest Docusaurus book content into vector database.
**Estimated Duration**: 3-4 hours

### 2.1 Content Preprocessing

- [ ] **2.1.1** Create `src/rag/ingestion.py`
- [ ] **2.1.2** Implement MDX file parser
  - [ ] Parse frontmatter metadata
  - [ ] Extract clean text content
  - [ ] Handle code blocks and special syntax
- [ ] **2.1.3** Implement content chunking
  - [ ] Split by paragraphs/sections
  - [ ] Target chunk size: 500-1000 tokens
  - [ ] Add overlap between chunks (100 tokens)
- [ ] **2.1.4** Extract metadata per chunk
  - [ ] Source file path
  - [ ] Module name
  - [ ] Section title
- **Acceptance**: MDX files parsed into clean chunks

### 2.2 Embedding Generation

- [ ] **2.2.1** Create `src/rag/embeddings.py`
- [ ] **2.2.2** Implement OpenAI embedding client
  - [ ] Use text-embedding-ada-002 model
  - [ ] Batch processing (max 100 texts)
  - [ ] Rate limiting and retry logic
- [ ] **2.2.3** Add token counting utilities
  - [ ] Use tiktoken for accurate counts
  - [ ] Validate chunk sizes
- [ ] **2.2.4** Unit tests for embedding generation
- **Acceptance**: Text → 1536-dim vectors with <1s latency

### 2.3 Vector Storage

- [ ] **2.3.1** Create `src/rag/vectorstore.py`
- [ ] **2.3.2** Implement Qdrant collection management
  - [ ] Create collection with cosine distance
  - [ ] Configure vector dimensions (1536)
- [ ] **2.3.3** Implement vector upsert
  - [ ] Batch insert with metadata
  - [ ] Handle duplicates
- [ ] **2.3.4** Create ingestion CLI script
  - [ ] Create `examples/rag-chatbot/scripts/ingest.py`
  - [ ] Accept `--docs-path` argument
  - [ ] Show progress bar
- [ ] **2.3.5** Run full ingestion
  ```bash
  python examples/rag-chatbot/scripts/ingest.py --docs-path ./docs
  ```
- **Acceptance**: SC-001 - 100% of MDX files ingested without errors

---

## Phase 3: Retrieval Pipeline 🎯 MVP

**Goal**: Implement semantic search to retrieve relevant passages.
**Estimated Duration**: 3-4 hours

### 3.1 Semantic Search

- [ ] **3.1.1** Create `src/rag/retriever.py`
- [ ] **3.1.2** Implement query embedding
  - [ ] Embed user question
  - [ ] Cache frequent queries (optional)
- [ ] **3.1.3** Implement similarity search
  - [ ] Top-k retrieval (k=5 default)
  - [ ] Score threshold filtering (>0.7)
  - [ ] Return with metadata
- **Acceptance**: Queries return relevant documents

### 3.2 Passage Ranking

- [ ] **3.2.1** Implement result ranking
  - [ ] Sort by similarity score
  - [ ] Deduplicate similar passages
- [ ] **3.2.2** Implement metadata filtering
  - [ ] Filter by module (optional)
  - [ ] Filter by recency (optional)
- **Acceptance**: Top results are most relevant

### 3.3 Retrieval Testing

- [ ] **3.3.1** Create test query set (10+ questions)
- [ ] **3.3.2** Evaluate retrieval accuracy
  - [ ] Manual relevance judgment
  - [ ] Calculate precision@k
- [ ] **3.3.3** Create `examples/rag-chatbot/scripts/query.py` CLI
  ```bash
  python examples/rag-chatbot/scripts/query.py "How do I create a ROS 2 node?"
  ```
- **Acceptance**: SC-002 - ≥90% relevant passages in top 5

---

## Phase 4: Response Generation 🎯 MVP

**Goal**: Generate grounded answers from retrieved passages.
**Estimated Duration**: 4-5 hours

### 4.1 Answer Generation

- [ ] **4.1.1** Create `src/rag/generator.py`
- [ ] **4.1.2** Implement system prompt
  - [ ] Grounding instructions
  - [ ] Citation format
  - [ ] "I don't know" policy
- [ ] **4.1.3** Implement context injection
  - [ ] Format retrieved passages
  - [ ] Include source metadata
  - [ ] Respect token limits
- [ ] **4.1.4** Implement OpenAI completion
  - [ ] Use GPT-3.5-turbo (cost-effective)
  - [ ] GPT-4 option for quality
  - [ ] Streaming support (optional)
- **Acceptance**: Answers cite sources correctly

### 4.2 Grounding Enforcement

- [ ] **4.2.1** Implement answer validation
  - [ ] Check citations exist in context
  - [ ] Detect potential hallucination
- [ ] **4.2.2** Implement "I don't know" handling
  - [ ] Detect low relevance scores
  - [ ] Return helpful message
- [ ] **4.2.3** Test with out-of-scope questions
  ```bash
  # Should return "I don't have information about that"
  curl -X POST http://localhost:8000/api/chat \
    -d '{"question": "What is the weather today?"}'
  ```
- **Acceptance**: SC-003, SC-005 - No hallucination, graceful fallback

### 4.3 Citation Extraction

- [ ] **4.3.1** Extract source references from response
- [ ] **4.3.2** Map to documentation URLs
- [ ] **4.3.3** Validate links are accessible
- **Acceptance**: SC-006 - All citations point to valid pages

---

## Phase 5: FastAPI Backend 🎯 MVP

**Goal**: Expose RAG pipeline as REST API.
**Estimated Duration**: 3-4 hours

### 5.1 API Setup

- [ ] **5.1.1** Create `src/rag/api/__init__.py`
- [ ] **5.1.2** Create `src/rag/api/models.py`
  - [ ] ChatRequest (question, conversation_id)
  - [ ] ChatResponse (answer, sources, conversation_id)
  - [ ] Source (title, path, snippet)
- [ ] **5.1.3** Create `src/rag/api/main.py`
  - [ ] FastAPI application
  - [ ] CORS middleware (allow localhost:3000)
  - [ ] Exception handlers
- **Acceptance**: App starts without errors

### 5.2 Endpoints

- [ ] **5.2.1** Create `src/rag/api/routes.py`
- [ ] **5.2.2** Implement `POST /api/chat`
  - [ ] Accept ChatRequest
  - [ ] Return ChatResponse
  - [ ] Handle errors gracefully
- [ ] **5.2.3** Implement `GET /api/health`
  - [ ] Check Qdrant connection
  - [ ] Check OpenAI API status
- [ ] **5.2.4** Implement `POST /api/ingest` (admin)
  - [ ] Trigger re-ingestion
  - [ ] Protected endpoint
- **Acceptance**: All endpoints respond correctly

### 5.3 Integration Testing

- [ ] **5.3.1** Create integration test suite
- [ ] **5.3.2** Test end-to-end flow
  ```bash
  uvicorn src.rag.api.main:app --port 8000
  curl http://localhost:8000/api/health
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "What is Isaac ROS?"}'
  ```
- **Acceptance**: SC-004 - Response time ≤3s for 95th percentile

---

## Phase 6: Frontend Integration

**Goal**: Chat widget for Docusaurus site.
**Estimated Duration**: 4-6 hours

### 6.1 React Component

- [ ] **6.1.1** Create `examples/rag-chatbot/frontend/ChatWidget.tsx`
  - [ ] Floating chat button
  - [ ] Expandable chat panel
  - [ ] Message input
- [ ] **6.1.2** Implement message display
  - [ ] User messages (right aligned)
  - [ ] Bot messages (left aligned)
  - [ ] Source citations with links
- [ ] **6.1.3** Implement loading states
  - [ ] Typing indicator
  - [ ] Skeleton loading
- [ ] **6.1.4** Style the component
  - [ ] Create `styles.css`
  - [ ] Match Docusaurus theme
- **Acceptance**: Widget renders correctly

### 6.2 Docusaurus Integration

- [ ] **6.2.1** Create Docusaurus plugin wrapper
- [ ] **6.2.2** Add to `docusaurus.config.js`
- [ ] **6.2.3** Test on live docs site
- **Acceptance**: Widget appears on all doc pages

---

## Phase 7: Documentation

**Goal**: Create Docusaurus-ready documentation.
**Estimated Duration**: 2-3 hours

### 7.1 Module Documentation

- [ ] **7.1.1** Create `docs/modules/rag-chatbot/introduction.mdx`
- [ ] **7.1.2** Create `docs/modules/rag-chatbot/ingestion.mdx`
  - [ ] Configuration options
  - [ ] Chunk size tuning
- [ ] **7.1.3** Create `docs/modules/rag-chatbot/api.mdx`
  - [ ] Endpoint reference
  - [ ] Request/response examples
- [ ] **7.1.4** Create `docs/modules/rag-chatbot/frontend.mdx`
  - [ ] Widget customization
  - [ ] Theme integration
- **Acceptance**: Docs build without errors

### 7.2 Update Sidebar

- [ ] **7.2.1** Add RAG Chatbot to `sidebars.js`
- [ ] **7.2.2** Verify navigation works
- **Acceptance**: RAG module accessible in docs

---

## Success Criteria Tracking

| Criterion | Description | Target | Phase | Status |
|-----------|-------------|--------|-------|--------|
| SC-001 | MDX ingestion success | 100% | Phase 2 | ⬜ |
| SC-002 | Relevant passages retrieved | ≥90% | Phase 3 | ⬜ |
| SC-003 | Grounded answers (no hallucination) | 100% | Phase 4 | ⬜ |
| SC-004 | API response time | ≤3s p95 | Phase 5 | ⬜ |
| SC-005 | "I don't know" handling | 100% | Phase 4 | ⬜ |
| SC-006 | Valid source citations | 100% | Phase 4 | ⬜ |

---

## Risk Matrix

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| OpenAI API costs | Medium | Medium | Use GPT-3.5, cache queries | Phase 4 |
| Hallucination | High | Medium | Strict prompting, validation | Phase 4 |
| Qdrant downtime | Medium | Low | Health checks, local fallback | Phase 3 |
| Large doc chunks | Low | Medium | Optimize chunk size (500-1000) | Phase 2 |

---

## File Checklist

### Core Library
- [ ] `src/rag/__init__.py`
- [ ] `src/rag/config.py`
- [ ] `src/rag/embeddings.py`
- [ ] `src/rag/vectorstore.py`
- [ ] `src/rag/ingestion.py`
- [ ] `src/rag/retriever.py`
- [ ] `src/rag/generator.py`

### API
- [ ] `src/rag/api/__init__.py`
- [ ] `src/rag/api/main.py`
- [ ] `src/rag/api/routes.py`
- [ ] `src/rag/api/models.py`

### Scripts & Config
- [ ] `examples/rag-chatbot/config/rag_config.yaml`
- [ ] `examples/rag-chatbot/scripts/ingest.py`
- [ ] `examples/rag-chatbot/scripts/query.py`

### Frontend
- [ ] `examples/rag-chatbot/frontend/ChatWidget.tsx`
- [ ] `examples/rag-chatbot/frontend/styles.css`

### Documentation
- [ ] `docs/modules/rag-chatbot/introduction.mdx`
- [ ] `docs/modules/rag-chatbot/ingestion.mdx`
- [ ] `docs/modules/rag-chatbot/api.mdx`
- [ ] `docs/modules/rag-chatbot/frontend.mdx`

### Tests
- [ ] `tests/rag/test_embeddings.py`
- [ ] `tests/rag/test_vectorstore.py`
- [ ] `tests/rag/test_retriever.py`
- [ ] `tests/rag/test_generator.py`
- [ ] `tests/rag/test_integration.py`
