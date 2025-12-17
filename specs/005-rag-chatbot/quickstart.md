# Quickstart: RAG Chatbot

Get the RAG chatbot running in 10 minutes.

## Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- OpenAI API key

## 1. Start Qdrant

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 -p 6334:6334 \
  --name qdrant \
  qdrant/qdrant

# Verify: http://localhost:6333/dashboard
```

## 2. Install Dependencies

```bash
cd ai-nativebook

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install RAG dependencies
pip install fastapi uvicorn qdrant-client openai \
  pydantic python-frontmatter tiktoken httpx
```

## 3. Configure Environment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Set Qdrant URL (optional, defaults to localhost)
export QDRANT_URL="http://localhost:6333"
```

## 4. Ingest Documentation

```bash
# Ingest all MDX documentation into Qdrant
python examples/rag-chatbot/scripts/ingest.py --docs-path ./docs

# Expected output:
# Ingesting docs from ./docs
# Processing 25 files...
# Created 150 embeddings
# Ingestion complete!
```

## 5. Start the API

```bash
# Start FastAPI server
uvicorn src.rag.api.main:app --reload --port 8000

# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 6. Test Query

```bash
# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a ROS 2 node?"}'

# Expected response:
# {
#   "answer": "To create a ROS 2 node, you need to...",
#   "sources": [
#     {"title": "ROS 2 Architecture", "path": "/docs/modules/ros2/architecture"}
#   ]
# }
```

## 7. Chat Widget (Optional)

Start the Docusaurus site with the chat widget:

```bash
# In a new terminal
npm run start

# Open http://localhost:3000
# Click the chat icon in the bottom right
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qdrant not running | `docker start qdrant` |
| OpenAI API error | Check `OPENAI_API_KEY` is set |
| No results found | Run ingestion first |
| Slow responses | Check network/API rate limits |

## Next Steps

- [Ingestion Guide](./ingestion.md) - Configure chunking
- [API Reference](./api.md) - Full endpoint docs
- [Frontend Setup](./frontend.md) - Chat widget config
