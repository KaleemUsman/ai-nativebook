---
title: AI-Native Book RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI-Native Book

The comprehensive guide to building intelligent humanoid robots. From ROS 2 fundamentals to vision-language-action models.

## RAG Chatbot API

This Hugging Face Space hosts a Retrieval-Augmented Generation (RAG) chatbot backend for the AI-Native Book on Physical AI & Humanoid Robotics.

### API Endpoints

- `GET /` - API information
- `GET /docs` - Interactive Swagger UI documentation
- `GET /api/health` - Health check endpoint
- `POST /api/chat` - Ask questions about the book content

### Example Usage

```bash
curl -X POST "https://kaleemusman-ai-nativebook.hf.space/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS 2?"}'
```

### Response Format

```json
{
  "answer": "Based on the documentation...",
  "sources": [
    {
      "title": "ROS 2 Architecture",
      "path": "/docs/modules/ros2-fundamentals",
      "module": "ros2-fundamentals",
      "snippet": "..."
    }
  ],
  "conversation_id": "uuid",
  "is_grounded": true
}
```
