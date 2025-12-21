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

try:
    from openai_agents import Agent, Runner, tool
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .config import get_config, RAGConfig
from .retriever import Retriever, get_retriever, RetrievalResult
from .vectorstore import Document


SYSTEM_PROMPT_TEMPLATE = """You are a friendly and knowledgeable AI assistant for the AI-Native Book on Physical AI & Humanoid Robotics.

I want you to act like a helpful human expert who's passionate about robotics and AI. Answer questions in a natural, conversational way - like you're chatting with a colleague over coffee.

Key guidelines:
- **Use only the provided context**: Base your answers entirely on the documentation excerpts given to you.
- **Be conversational**: Use phrases like "Oh, that's a great question!" or "Let me explain this..." to make it feel human.
- **Cite sources naturally**: Mention where the info comes from in a casual way, like "According to the ROS 2 guide..." or "In the Gazebo section, it says..."
- **Admit limitations**: If the context doesn't cover something, say "Hmm, I'm not seeing that in the docs I have access to right now."
- **Keep it engaging**: Ask follow-up questions or suggest related topics when appropriate.
- **Stay on topic**: Focus on {module_list} and related robotics/AI concepts.

Remember, you're helping someone learn about this fascinating field - be encouraging and clear!"""


NO_INFO_TEMPLATE = "I don't have information about that in the documentation. Could you try rephrasing your question or ask about {module_list}?"


if AGENTS_AVAILABLE:
    @tool
    def get_book_context(query: str) -> str:
        """Retrieve relevant context from the AI-Native Book based on the query."""
        retriever = get_retriever()
        result = retriever.retrieve(query)
        if result.documents:
            context = retriever.format_context(result.documents)
            return f"Relevant book content:\n{context}"
        else:
            return "No relevant information found in the book."


def get_available_modules(vector_store) -> List[str]:
    """Discover available modules from the vector store metadata."""
    try:
        # Query a sample of documents to extract module names
        from .embeddings import get_embedding_client
        client = get_embedding_client()
        sample_query = client.embed("module documentation")
        
        docs = vector_store.search(
            query_embedding=sample_query,
            top_k=50,
            score_threshold=-1.0
        )
        
        modules = set()
        for doc in docs:
            module = doc.metadata.get('module', '')
            if module:
                modules.add(module.replace('-', ' ').title())
        
        return sorted(list(modules)) if modules else ["the documentation"]
    except Exception:
        return ["the documentation"]


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
    - Dynamic module discovery
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
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.config.openai.api_key:
            self.openai_client = OpenAI(api_key=self.config.openai.api_key)
        else:
            self.openai_client = None

        # Initialize Hugging Face client
        if HF_AVAILABLE and self.config.huggingface.api_key:
            self.hf_client = InferenceClient(token=self.config.huggingface.api_key)
        else:
            self.hf_client = None

        # Determine which model to use (prefer OpenAI if available, else HF)
        if self.openai_client:
            self.model = self.config.openai.completion_model
            self.temperature = self.config.openai.temperature
            self.max_tokens = self.config.openai.max_tokens
        elif self.hf_client:
            self.model = self.config.huggingface.model
            self.temperature = self.config.huggingface.temperature
            self.max_tokens = self.config.huggingface.max_tokens
        else:
            self.model = "mock"
            self.temperature = 0.1
            self.max_tokens = 1024
        
        # Initialize lazy-loaded attributes
        self._modules = None
    
    @property
    def modules(self) -> List[str]:
        """Get available modules (lazy-loaded)."""
        if self._modules is None:
            from .vectorstore import get_vector_store
            self._modules = get_available_modules(get_vector_store(self.config))
        return self._modules
    
    @property
    def module_list_str(self) -> str:
        """Get formatted module list string."""
        if len(self.modules) <= 1:
            return self.modules[0] if self.modules else "the documentation"
        return ", ".join(self.modules[:-1]) + ", or " + self.modules[-1]
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt with dynamic modules."""
        return SYSTEM_PROMPT_TEMPLATE.format(module_list=self.module_list_str)
    
    def get_no_info_message(self) -> str:
        """Get the 'no information' message with dynamic modules."""
        return NO_INFO_TEMPLATE.format(module_list=self.module_list_str)
    
    def generate(
        self,
        question: str,
        retrieval_result: Optional[RetrievalResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        selected_text: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate an answer for a question.
        
        Args:
            question: User question
            retrieval_result: Pre-retrieved documents (optional)
            conversation_history: Previous messages (optional) - ignored for stateless
            selected_text: User-selected text to base answer on (optional)
            
        Returns:
            GenerationResult with answer and metadata
        """
        if AGENTS_AVAILABLE and self.openai_client:
            # Use OpenAI Agents for soft-coded behavior
            agent = Agent(
                name="Book Chatbot",
                instructions=self.system_prompt,
                tools=[get_book_context],
                model=self.model
            )
            
            if selected_text:
                user_query = f"Based on this selected text from the book: {selected_text}\n\nQuestion: {question}"
            else:
                user_query = question
            
            result = Runner.run(agent, user_query)
            answer = result.final_output
            
            # Mock sources since agent doesn't provide them
            sources = [{"title": "AI-Native Book", "path": "", "module": "", "snippet": selected_text[:200] if selected_text else ""}]
            
            return GenerationResult(
                answer=answer,
                sources=sources,
                context_used=selected_text or "Retrieved via agent",
                model=self.model,
                is_grounded=True
            )
        else:
            # Fallback to original RAG logic
            if selected_text:
                context = selected_text
                sources = [{"title": "Selected Text", "path": "", "module": "", "snippet": selected_text[:200]}]
            else:
                # Retrieve documents if not provided
                if retrieval_result is None:
                    retrieval_result = self.retriever.retrieve(question)
                
                documents = retrieval_result.documents
                
                if not documents:
                    return GenerationResult(
                        answer=self.get_no_info_message(),
                        sources=[],
                        context_used="",
                        model=self.model,
                        is_grounded=True  # Saying "I don't know" is grounded
                    )
                
                # Format context
                context = self.retriever.format_context(documents)
                sources = self.retriever.get_sources(documents)
            
            # Build messages with dynamic system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context and question
            user_message = f"""Based on the following documentation:

{context}

---

Question: {question}

Please provide a helpful, accurate answer based only on the documentation above. If the documentation doesn't contain the answer, say so."""

            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            if self.openai_client:
                answer = self._generate_with_openai(messages)
            elif self.hf_client:
                answer = self._generate_with_hf(messages)
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
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"I encountered an error generating a response. Please try again. (Error: {str(e)[:100]})"

    def _generate_with_hf(self, messages: List[Dict[str, str]]) -> str:
        """Generate answer using Hugging Face API."""
        try:
            # Convert messages to a single prompt for HF
            prompt = self._messages_to_prompt(messages)
            
            response = self.hf_client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
            return response
        except Exception as e:
            print(f"Hugging Face API error: {e}")
            return f"I encountered an error generating a response. Please try again. (Error: {str(e)[:100]})"

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
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
