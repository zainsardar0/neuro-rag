from app.embedding.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.core.logger import app_logger
from app.core.exceptions import RetrievalError

# Minimum similarity score to consider a chunk relevant
SIMILARITY_THRESHOLD = 0.3


class Retriever:
    """
    Handles retrieval of relevant document chunks
    for a given user query.
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize retriever with embedder and vector store.

        Args:
            top_k: Number of chunks to retrieve per query
        """
        try:
            self.top_k = top_k
            self.embedder = Embedder()
            self.vector_store = VectorStore()
            app_logger.info(f"Retriever initialized with top_k={top_k}")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}")

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: User's question string

        Returns:
            List of dicts with 'text', 'page', 'source', 'score' keys
            Sorted by relevance score descending

        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        try:
            app_logger.info(f"Retrieving chunks for query: {query[:50]}...")

            # Step 1 — Embed the query
            query_embedding = self.embedder.embed_query(query)

            # Step 2 — Query ChromaDB
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.top_k
            )

            # Step 3 — Filter by similarity threshold
            filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

            if not filtered:
                app_logger.warning(
                    f"No chunks above threshold {SIMILARITY_THRESHOLD} "
                    f"for query: {query[:50]}"
                )
                return []

            app_logger.info(
                f"Retrieved {len(filtered)} relevant chunks "
                f"(filtered from {len(results)})"
            )

            # Log top result for debugging
            app_logger.debug(
                f"Top result — Score: {filtered[0]['score']} | "
                f"Source: {filtered[0]['source']} | "
                f"Page: {filtered[0]['page']}"
            )

            return filtered

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def retrieve_with_fallback(self, query: str) -> tuple[list[dict], bool]:
        """
        Retrieve chunks with fallback flag for LangGraph.
        Returns results and a boolean indicating if fallback is needed.

        Args:
            query: User's question string

        Returns:
            Tuple of (results, needs_fallback)
            needs_fallback=True means no relevant chunks found
        """
        results = self.retrieve(query)
        needs_fallback = len(results) == 0
        return results, needs_fallback