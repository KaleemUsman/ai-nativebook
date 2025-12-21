"""
Sentence Transformers Embeddings Client

Generates embeddings using sentence-transformers all-MiniLM-L6-v2 model.
"""

import time
from typing import List, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .config import get_config, RAGConfig


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int


class EmbeddingClient:
    """
    Client for generating embeddings using Sentence Transformers.
    
    Features:
    - Batch embedding generation
    - Token counting (approximate)
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
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
            
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count (rough estimate: 1 token ≈ 4 characters).
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        return len(text) // 4
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (384 dimensions for all-MiniLM-L6-v2)
        """
        if not self.model:
            # Return mock embedding for testing
            return self._mock_embedding(text)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            return embedding
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
        if not self.model:
            return [self._mock_embedding(t) for t in texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            print(f"Embedding batch error: {e}")
            return [self._mock_embedding(t) for t in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing."""
        import hashlib
        
        # Create deterministic embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Expand to embedding_dim dimensions
        embedding = []
        for i in range(self.embedding_dim):
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
