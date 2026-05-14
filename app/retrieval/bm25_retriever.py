from rank_bm25 import BM25Okapi
from app.core.logger import app_logger
from app.core.exceptions import RetrievalError

# Number of top BM25 candidates to return
BM25_TOP_K = 10


class BM25Retriever:
    """
    Keyword-based retriever using BM25 algorithm.

    BM25 excels at exact keyword matching — complements
    semantic search which excels at meaning-based matching.

    Index is built in-memory from all chunks in ChromaDB.
    Rebuilds automatically on every server startup.
    """

    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from all stored chunks.

        Args:
            chunks: All chunks from ChromaDB with 'text',
                   'page', 'source' keys
        """
        self.chunks = chunks
        self.index = None

        if not chunks:
            app_logger.warning("BM25Retriever: no chunks provided — index is empty")
            return

        try:
            app_logger.info(f"Building BM25 index from {len(chunks)} chunks...")

            # Tokenize each chunk text into words
            # BM25 works on token lists, not raw strings
            tokenized = [
                chunk["text"].lower().split()
                for chunk in chunks
            ]

            self.index = BM25Okapi(tokenized)
            app_logger.info("BM25 index built successfully")

        except Exception as e:
            raise RetrievalError(f"Failed to build BM25 index: {str(e)}")

    def retrieve(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        """
        Retrieve top_k chunks using BM25 keyword matching.

        On empty index or any failure, returns empty list —
        hybrid search will fall back to semantic only.

        Args:
            query: User query string (rewritten query in V2)
            top_k: Number of top BM25 results to return

        Returns:
            List of dicts with 'text', 'page', 'source',
            'score', 'bm25_rank' keys
            Sorted by BM25 score descending
        """
        if not self.index or not self.chunks:
            app_logger.warning("BM25 index is empty — returning no results")
            return []

        if not query or not query.strip():
            return []

        try:
            # Tokenize query same way as index
            tokenized_query = query.lower().split()

            # Get BM25 scores for all chunks
            scores = self.index.get_scores(tokenized_query)

            # Pair each chunk with its BM25 score and rank
            scored_chunks = [
                {
                    "text": self.chunks[i]["text"],
                    "page": self.chunks[i]["page"],
                    "source": self.chunks[i]["source"],
                    "score": 0.0,           # placeholder — filled by hybrid merger
                    "bm25_score": float(scores[i]),
                    "bm25_rank": 0          # filled after sorting
                }
                for i in range(len(self.chunks))
            ]

            # Sort by BM25 score descending
            scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)

            # Assign ranks (1-based)
            for rank, chunk in enumerate(scored_chunks, start=1):
                chunk["bm25_rank"] = rank

            # Return top_k
            top_chunks = scored_chunks[:top_k]

            app_logger.info(
                f"BM25 retrieved {len(top_chunks)} chunks | "
                f"Top score: {top_chunks[0]['bm25_score']:.4f}"
                if top_chunks else "BM25 retrieved 0 chunks"
            )

            return top_chunks

        except Exception as e:
            app_logger.warning(f"BM25 retrieval failed — returning empty. Error: {e}")
            return []