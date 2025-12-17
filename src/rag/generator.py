"""
Answer Generator

Generates grounded answers using OpenAI GPT models.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .config import get_config, RAGConfig
from .retriever import Retriever, get_retriever, RetrievalResult
from .vectorstore import Document


SYSTEM_PROMPT = """You are a helpful AI assistant for the AI-Native Book on Physical AI & Humanoid Robotics.

Your role is to answer questions ONLY based on the provided documentation context. Follow these rules strictly:

1. **Ground your answers**: Only use information from the provided context. Do not use external knowledge.
2. **Cite sources**: Reference the source documents when providing information (e.g., "According to the ROS 2 Architecture guide...").
3. **Be accurate**: If the context doesn't contain enough information to answer the question, say "I don't have information about that in the documentation."
4. **Be helpful**: Provide clear, concise answers. Include relevant code examples if they appear in the context.
5. **Stay on topic**: The documentation covers ROS 2, Gazebo/Unity simulation, Isaac AI, and Vision-Language-Action modules.

If asked about topics not covered in the documentation (like current events, personal opinions, or unrelated subjects), politely explain that you can only answer questions about the book content.

Remember: It's better to say "I don't know" than to make up information."""


@dataclass
class GenerationResult:
    """Result from answer generation."""
    answer: str
    sources: List[Dict[str, str]]
    context_used: str
    model: str
    is_grounded: bool


class Generator:
    """
    Answer generator using OpenAI GPT models.
    
    Features:
    - Grounded answer generation
    - Source citation
    - "I don't know" handling
    - Context injection
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        retriever: Optional[Retriever] = None
    ):
        """
        Initialize the generator.
        
        Args:
            config: RAG configuration
            retriever: Retriever to use for document retrieval
        """
        self.config = config or get_config()
        self.retriever = retriever or get_retriever(self.config)
        
        if OPENAI_AVAILABLE and self.config.openai.api_key:
            self.client = OpenAI(api_key=self.config.openai.api_key)
        else:
            self.client = None
        
        self.model = self.config.openai.completion_model
        self.temperature = self.config.openai.temperature
        self.max_tokens = self.config.openai.max_tokens
    
    def generate(
        self,
        question: str,
        retrieval_result: Optional[RetrievalResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> GenerationResult:
        """
        Generate an answer for a question.
        
        Args:
            question: User question
            retrieval_result: Pre-retrieved documents (optional)
            conversation_history: Previous messages (optional)
            
        Returns:
            GenerationResult with answer and metadata
        """
        # Retrieve documents if not provided
        if retrieval_result is None:
            retrieval_result = self.retriever.retrieve(question)
        
        documents = retrieval_result.documents
        
        # Check if we have relevant context
        if not documents:
            return GenerationResult(
                answer="I don't have information about that in the documentation. Could you try rephrasing your question or ask about ROS 2, simulation, Isaac AI, or the VLA module?",
                sources=[],
                context_used="",
                model=self.model,
                is_grounded=True  # Saying "I don't know" is grounded
            )
        
        # Format context
        context = self.retriever.format_context(documents)
        sources = self.retriever.get_sources(documents)
        
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-4:]:  # Last 4 messages
                messages.append(msg)
        
        # Add context and question
        user_message = f"""Based on the following documentation:

{context}

---

Question: {question}

Please provide a helpful, accurate answer based only on the documentation above. If the documentation doesn't contain the answer, say so."""

        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        if self.client:
            answer = self._generate_with_openai(messages)
        else:
            answer = self._generate_mock(question, documents)
        
        # Validate grounding
        is_grounded = self._validate_grounding(answer, context)
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            context_used=context,
            model=self.model,
            is_grounded=is_grounded
        )
    
    def _generate_with_openai(self, messages: List[Dict[str, str]]) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I encountered an error generating a response. Please try again. (Error: {str(e)[:100]})"
    
    def _generate_mock(self, question: str, documents: List[Document]) -> str:
        """Generate a mock answer for testing without API."""
        if not documents:
            return "I don't have information about that topic in the documentation."
        
        # Create a simple answer from the first document
        doc = documents[0]
        title = doc.metadata.get('title', 'the documentation')
        snippet = doc.text[:500] if len(doc.text) > 500 else doc.text
        
        return f"""Based on {title}:

{snippet}

[Note: This is a simulated response. Set OPENAI_API_KEY for real answers.]"""
    
    def _validate_grounding(self, answer: str, context: str) -> bool:
        """
        Validate that the answer is grounded in the context.
        
        This is a simple heuristic check. Production systems would use
        more sophisticated methods.
        """
        # Check for "I don't know" responses
        dont_know_phrases = [
            "i don't have information",
            "not covered in the documentation",
            "i'm not sure",
            "cannot find",
            "no information about"
        ]
        
        answer_lower = answer.lower()
        
        # If it's an "I don't know" response, it's grounded
        for phrase in dont_know_phrases:
            if phrase in answer_lower:
                return True
        
        # Check if the answer references the context
        reference_phrases = [
            "according to",
            "the documentation",
            "based on",
            "the guide",
            "as shown",
            "from the"
        ]
        
        has_reference = any(phrase in answer_lower for phrase in reference_phrases)
        
        # Simple check: some key terms from context should appear in answer
        context_words = set(context.lower().split())
        answer_words = set(answer_lower.split())
        
        overlap = len(context_words & answer_words) / max(len(answer_words), 1)
        
        return has_reference or overlap > 0.1


def ask(
    question: str,
    config: Optional[RAGConfig] = None
) -> GenerationResult:
    """
    Convenience function to ask a question and get an answer.
    
    Args:
        question: User question
        config: RAG configuration
        
    Returns:
        GenerationResult with answer and sources
    """
    generator = get_generator(config)
    return generator.generate(question)


# Global generator instance
_generator: Optional[Generator] = None


def get_generator(config: Optional[RAGConfig] = None) -> Generator:
    """Get or create the global generator."""
    global _generator
    if _generator is None:
        _generator = Generator(config)
    return _generator
