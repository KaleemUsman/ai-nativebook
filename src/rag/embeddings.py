"""
OpenAI Embeddings Client

Generates embeddings using OpenAI's text-embedding-ada-002 model.
"""

import time
from typing import List, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .config import get_config, RAGConfig


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int


class EmbeddingClient:
    """
    Client for generating embeddings using OpenAI API.
    
    Features:
    - Batch embedding generation
    - Token counting
    - Retry logic with exponential backoff
    - Rate limiting
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the embedding client.
        
        Args:
            config: RAG configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        
        if OPENAI_AVAILABLE and self.config.openai.api_key:
            self.client = OpenAI(api_key=self.config.openai.api_key)
        else:
            self.client = None
            
        self.model = self.config.openai.embedding_model
        
        # Token encoder for counting
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough estimate: ~4 characters per token
            return len(text) // 4
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (1536 dimensions)
        """
        if not self.client:
            # Return mock embedding for testing
            return self._mock_embedding(text)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return self._mock_embedding(text)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Maximum texts per API call
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if show_progress:
                print(f"Processing batch {batch_num}/{total_batches}...")
            
            embeddings = self._embed_batch_internal(batch)
            
            for text, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    token_count=self.count_tokens(text)
                ))
            
            # Rate limiting: small delay between batches
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return results
    
    def _embed_batch_internal(
        self,
        texts: List[str],
        max_retries: int = 3
    ) -> List[List[float]]:
        """Generate embeddings for a batch with retry logic."""
        if not self.client:
            return [self._mock_embedding(t) for t in texts]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retry {attempt + 1} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Batch embedding failed: {e}")
                    return [self._mock_embedding(t) for t in texts]
        
        return [self._mock_embedding(t) for t in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing."""
        import hashlib
        
        # Create deterministic embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Expand to 1536 dimensions
        embedding = []
        for i in range(1536):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)
        
        # Normalize to unit vector
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


# Global client instance
_client: Optional[EmbeddingClient] = None


def get_embedding_client(config: Optional[RAGConfig] = None) -> EmbeddingClient:
    """Get or create the global embedding client."""
    global _client
    if _client is None:
        _client = EmbeddingClient(config)
    return _client
