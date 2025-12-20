# Hugging Face Spaces Dockerfile for RAG Chatbot
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY qdrant_db/ ./qdrant_db/
COPY docs/ ./docs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QDRANT_URL=./qdrant_db

# Expose port for HF Spaces
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
