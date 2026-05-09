from sentence_transformers import CrossEncoder
from app.core.logger import app_logger
from app.core.exceptions import RetrievalError

# Industry standard reranking model
# Lightweight, runs on CPU, no API key needed
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum number of chunks required to perform reranking
# If fewer chunks retrieved, return as-is
MIN_CHUNKS_FOR_RERANKING = 2

# Number of top chunks to keep after reranking
TOP_N_AFTER_RERANK = 5


class Reranker:
    """
    Reranks retrieved chunks using a CrossEncoder model.

    CrossEncoder scores each (query, chunk) pair together,
    providing much more accurate relevance scores than
    cosine similarity alone.

    Flow:
        ChromaDB top_k=10 → CrossEncoder → top 5 reranked chunks
    """

    def __init__(self):
        """Initialize the CrossEncoder model."""
        try:
            app_logger.info(f"Initializing Reranker: {MODEL_NAME}")
            # show_progress_bar=False keeps logs clean
            self.model = CrossEncoder(MODEL_NAME, max_length=512)
            app_logger.info("Reranker initialized successfully")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize Reranker: {str(e)}")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Rerank chunks by relevance to the query.

        Each chunk gets a rerank_score (0-1) from the CrossEncoder.
        Chunks are sorted by rerank_score descending.
        Only top TOP_N_AFTER_RERANK chunks are returned.

        On any failure, returns original chunks unchanged —
        reranking never breaks the pipeline.

        Args:
            query: User's original or rewritten query
            chunks: Retrieved chunks from ChromaDB with cosine scores

        Returns:
            Reranked list of chunks with added 'rerank_score' field
        """
        if not chunks:
            return chunks

        # Not enough chunks to meaningfully rerank
        if len(chunks) < MIN_CHUNKS_FOR_RERANKING:
            app_logger.warning(
                f"Only {len(chunks)} chunks retrieved — "
                f"skipping reranking, returning as-is"
            )
            # Still add rerank_score field for schema consistency
            for chunk in chunks:
                chunk["rerank_score"] = round(float(chunk["score"]), 4)
            return chunks

        try:
            app_logger.info(
                f"Reranking {len(chunks)} chunks for query: {query[:50]}..."
            )

            # Build (query, chunk_text) pairs for CrossEncoder
            pairs = [[query, chunk["text"]] for chunk in chunks]

            # Get raw scores from CrossEncoder
            raw_scores = self.model.predict(pairs)

            # Normalize scores to 0-1 range using sigmoid
            import math
            def sigmoid(x):
                return round(1 / (1 + math.exp(-x)), 4)

            normalized_scores = [sigmoid(score) for score in raw_scores]

            # Attach rerank_score to each chunk
            for chunk, score in zip(chunks, normalized_scores):
                chunk["rerank_score"] = score

            # Sort by rerank_score descending
            reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

            # Keep only top N
            reranked = reranked[:TOP_N_AFTER_RERANK]

            app_logger.info(
                f"Reranking complete — "
                f"top score: {reranked[0]['rerank_score']} | "
                f"bottom score: {reranked[-1]['rerank_score']}"
            )

            return reranked

        except Exception as e:
            # NEVER let reranking failure break the pipeline
            app_logger.warning(
                f"Reranking failed — returning original chunks. Error: {e}"
            )
            # Add rerank_score field for schema consistency
            for chunk in chunks:
                chunk["rerank_score"] = round(float(chunk["score"]), 4)
            return chunks[:TOP_N_AFTER_RERANK]