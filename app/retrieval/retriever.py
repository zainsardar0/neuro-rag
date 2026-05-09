from app.embedding.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.core.logger import app_logger
from app.core.exceptions import RetrievalError

# Minimum similarity score to consider a chunk relevant
SIMILARITY_THRESHOLD = 0.3

# V2: Fetch more candidates for reranker to work with
RETRIEVAL_TOP_K = 10


class Retriever:
    """
    Handles retrieval of relevant document chunks
    for a given user query.
    V2: Added reranking step after initial retrieval.
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize retriever with embedder, vector store and reranker.

        Args:
            top_k: Number of final chunks to return after reranking
        """
        try:
            self.top_k = top_k
            self.embedder = Embedder()
            self.vector_store = VectorStore()
            self.reranker = Reranker()              # V2: NEW
            app_logger.info(f"Retriever initialized with top_k={top_k}")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}")

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve and rerank most relevant chunks for a query.

        V2 Flow:
            1. Embed query
            2. Fetch top 10 from ChromaDB (wider candidate pool)
            3. Filter by similarity threshold
            4. Rerank with CrossEncoder
            5. Return top 5 reranked chunks

        Args:
            query: User's question string (rewritten query in V2)

        Returns:
            List of dicts with 'text', 'page', 'source',
            'score', 'rerank_score' keys
            Sorted by rerank_score descending

        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        try:
            app_logger.info(f"Retrieving chunks for query: {query[:50]}...")

            # Step 1 — Embed the query
            query_embedding = self.embedder.embed_query(query)

            # Step 2 — Query ChromaDB with wider candidate pool
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=RETRIEVAL_TOP_K           # V2: fetch 10 instead of 5
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

            # Step 4 — Rerank with CrossEncoder
            reranked = self.reranker.rerank(query, filtered)   # V2: NEW

            # Log top result for debugging
            app_logger.debug(
                f"Top result after reranking — "
                f"Rerank Score: {reranked[0]['rerank_score']} | "
                f"Cosine Score: {reranked[0]['score']} | "
                f"Source: {reranked[0]['source']} | "
                f"Page: {reranked[0]['page']}"
            )

            return reranked

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def retrieve_with_fallback(self, query: str) -> tuple[list[dict], bool]:
        """
        Retrieve and rerank chunks with fallback flag for LangGraph.
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