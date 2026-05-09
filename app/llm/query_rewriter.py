from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.llm.prompt_builder import build_rewrite_prompt
from app.core.config import get_settings
from app.core.logger import app_logger
from app.core.exceptions import LLMError

settings = get_settings()

# Rewriter uses same model but different temperature
# Lower temp = more deterministic, consistent rewrites
MODEL_NAME = "llama-3.3-70b-versatile"
REWRITER_TEMPERATURE = 0.0
REWRITER_MAX_TOKENS = 100  # Rewrites are short — no need for 1024


class QueryRewriter:
    """
    Rewrites user queries for better semantic retrieval.

    Uses the same Groq LLaMA model as the generator but with:
    - Temperature 0.0 for deterministic, consistent rewrites
    - Max tokens 100 — rewrites are always short
    - Fallback to original query on any failure (never breaks the pipeline)
    """

    def __init__(self):
        """Initialize the Groq rewriter client."""
        try:
            app_logger.info("Initializing QueryRewriter...")
            self.llm = ChatGroq(
                model=MODEL_NAME,
                temperature=REWRITER_TEMPERATURE,
                max_tokens=REWRITER_MAX_TOKENS,
                api_key=settings.groq_api_key
            )
            app_logger.info("QueryRewriter initialized successfully")
        except Exception as e:
            raise LLMError(f"Failed to initialize QueryRewriter: {str(e)}")

    def rewrite(self, query: str) -> str:
        """
        Rewrite a query for improved vector retrieval.

        On any failure, returns the original query unchanged.
        This ensures query rewriting never breaks the pipeline.

        Args:
            query: Original user query

        Returns:
            Rewritten query string (or original if rewriting fails)
        """
        if not query or not query.strip():
            return query

        try:
            app_logger.info(f"Rewriting query: {query[:60]}...")
            prompt = build_rewrite_prompt(query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()

            # Safety check: if LLM returns empty or garbage, fall back
            if not rewritten or len(rewritten) < 3:
                app_logger.warning("Rewriter returned empty response — using original query")
                return query

            app_logger.info(f"Rewritten query: {rewritten[:80]}...")
            return rewritten

        except Exception as e:
            # NEVER let rewriting failure break the pipeline
            app_logger.warning(f"Query rewriting failed — using original query. Error: {e}")
            return query