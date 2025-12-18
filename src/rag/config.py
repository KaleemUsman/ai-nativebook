"""
RAG Configuration Module

Manages configuration for the RAG chatbot using Pydantic settings.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from functools import lru_cache


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    url: str = Field(default=":memory:", description="Qdrant storage (url or :memory: or /path/to/db)")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key (for cloud)")
    collection_name: str = Field(default="ai-native-book", description="Collection name")
    vector_size: int = Field(default=1536, description="Vector dimensions (ada-002)")
    distance: str = Field(default="Cosine", description="Distance metric")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default="", description="OpenAI API key")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    completion_model: str = Field(default="gpt-3.5-turbo", description="Completion model")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=1024, description="Max response tokens")
    timeout: float = Field(default=30.0, description="API timeout in seconds")


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    chunk_size: int = Field(default=800, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = Field(default=5, description="Number of results to retrieve")
    score_threshold: float = Field(default=-1.0, description="Minimum similarity score")
    max_context_tokens: int = Field(default=3000, description="Max tokens for context")


class RAGConfig(BaseModel):
    """Complete RAG configuration."""
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables."""
        return cls(
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "./qdrant_db"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION", "ai-native-book"),
            ),
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                completion_model=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-3.5-turbo"),
            ),
        )


# Global config instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global RAG configuration."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def set_config(config: RAGConfig):
    """Set the global RAG configuration."""
    global _config
    _config = config
