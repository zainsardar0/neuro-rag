from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.llm.prompt_builder import build_rag_prompt, build_fallback_prompt
from app.core.config import get_settings
from app.core.logger import app_logger
from app.core.exceptions import LLMError

settings = get_settings()

# LLM Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 1024


class Generator:
    """
    Handles LLM response generation using Groq's LLaMA3.
    Generates grounded, citation-based answers from retrieved context.
    """

    def __init__(self):
        """Initialize the Groq LLM client."""
        try:
            app_logger.info(f"Initializing LLM: {MODEL_NAME}")
            self.llm = ChatGroq(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_key=settings.groq_api_key
            )
            app_logger.info("LLM initialized successfully")
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM: {str(e)}")

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved chunks from Phase 6

        Returns:
            Dict with 'answer', 'sources', and 'model' keys

        Raises:
            LLMError: If generation fails
        """
        if not query or not query.strip():
            raise LLMError("Query cannot be empty")

        try:
            # Build prompt
            if chunks:
                app_logger.info(f"Generating answer with {len(chunks)} context chunks")
                prompt = build_rag_prompt(query, chunks)
            else:
                app_logger.warning("No chunks provided — using fallback prompt")
                prompt = build_fallback_prompt(query)

            # Call Groq API
            app_logger.info("Calling Groq API...")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # Extract unique sources from chunks
            sources = []
            seen = set()
            for chunk in chunks:
                key = f"{chunk['source']}::{chunk['page']}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": chunk["source"],
                        "page": chunk["page"],
                        "score": chunk["score"]
                    })

            app_logger.info("Answer generated successfully")
            app_logger.debug(f"Answer preview: {answer[:200]}")

            return {
                "answer": answer,
                "sources": sources,
                "model": MODEL_NAME
            }

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Failed to generate answer: {str(e)}")

    def generate_with_fallback(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate answer with automatic fallback if no chunks provided.
        Designed for LangGraph integration in Phase 8.

        Args:
            query: User's question
            chunks: Retrieved chunks (can be empty list)

        Returns:
            Dict with 'answer', 'sources', 'model', 'used_fallback' keys
        """
        result = self.generate(query, chunks)
        result["used_fallback"] = len(chunks) == 0
        return result