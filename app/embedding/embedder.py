from sentence_transformers import SentenceTransformer
from app.core.logger import app_logger
from app.core.exceptions import EmbeddingError


# Model name constant — change here to switch models globally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Handles embedding generation using sentence-transformers.
    Designed to be model-switchable via EMBEDDING_MODEL constant.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedder with a sentence-transformer model.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        try:
            app_logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            app_logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")

    def embed_chunks(self, chunks: list[dict], batch_size: int = 32) -> list[dict]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of dicts with 'text', 'page', 'source', 'chunk_id' keys
            batch_size: Number of chunks to embed at once

        Returns:
            Same list of dicts with 'embedding' key added to each

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            raise EmbeddingError("No chunks provided for embedding")

        try:
            app_logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")

            # Extract just the text for embedding
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_list=True
            )

            # Add embedding back to each chunk dict
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding

            app_logger.info(f"Successfully embedded {len(chunks)} chunks")
            app_logger.info(f"Embedding dimensions: {len(embeddings[0])}")
            return chunks

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query string.
        Used at retrieval time to embed user questions.

        Args:
            query: User's question string

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise EmbeddingError("Query cannot be empty")

        try:
            app_logger.info(f"Embedding query: {query[:50]}...")
            embedding = self.model.encode(query, convert_to_list=True)
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {str(e)}")