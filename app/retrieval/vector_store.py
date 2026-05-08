import chromadb
from chromadb.config import Settings
from app.core.config import get_settings
from app.core.logger import app_logger
from app.core.exceptions import VectorStoreError

settings = get_settings()

# Collection name constant
COLLECTION_NAME = "neurorag"


class VectorStore:
    """
    Handles all ChromaDB operations including storing
    and retrieving embedded document chunks.
    """

    def __init__(self):
        """Initialize ChromaDB client with persistent storage."""
        try:
            app_logger.info(f"Initializing ChromaDB at: {settings.chroma_db_path}")
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            app_logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready")
            app_logger.info(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Store embedded chunks in ChromaDB.

        Args:
            chunks: List of dicts with 'text', 'page', 'source',
                   'chunk_id', 'embedding' keys

        Raises:
            VectorStoreError: If storing fails
        """
        if not chunks:
            raise VectorStoreError("No chunks provided to store")

        try:
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            embeddings = [chunk["embedding"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [
                {
                    "page": chunk["page"],
                    "source": chunk["source"]
                }
                for chunk in chunks
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            app_logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
            app_logger.info(f"Total documents in collection: {self.collection.count()}")

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to store chunks: {str(e)}")

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Retrieve most similar chunks for a query embedding.

        Args:
            query_embedding: Embedding vector of the user query
            top_k: Number of most similar chunks to retrieve

        Returns:
            List of dicts with 'text', 'page', 'source', 'score' keys

        Raises:
            VectorStoreError: If retrieval fails
        """
        try:
            app_logger.info(f"Querying ChromaDB for top {top_k} results")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                retrieved.append({
                    "text": doc,
                    "page": metadata["page"],
                    "source": metadata["source"],
                    "score": round(1 - distance, 4)
                })

            app_logger.info(f"Retrieved {len(retrieved)} chunks")
            return retrieved

        except Exception as e:
            raise VectorStoreError(f"Failed to query ChromaDB: {str(e)}")

    def count(self) -> int:
        """Return total number of chunks stored in ChromaDB."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        Useful for reingesting documents from scratch.
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            app_logger.warning("ChromaDB collection reset successfully")
        except Exception as e:
            raise VectorStoreError(f"Failed to reset collection: {str(e)}")

    def document_exists(self, filename: str) -> bool:
        """
        Check if a document with given filename already exists in ChromaDB.

        Args:
            filename: PDF filename to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            results = self.collection.get(where={"source": filename})
            return len(results["ids"]) > 0
        except Exception:
            return False

    def delete_document(self, filename: str) -> None:
        """
        Delete all chunks belonging to a specific document.

        Args:
            filename: PDF filename whose chunks to delete
        """
        try:
            self.collection.delete(where={"source": filename})
            app_logger.info(f"Deleted all chunks for: {filename}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document {filename}: {str(e)}")

    def list_documents(self) -> list[str]:
        """
        List all unique document filenames currently in ChromaDB.

        Returns:
            List of unique filenames
        """
        try:
            results = self.collection.get(include=["metadatas"])
            filenames = list(set(
                m["source"] for m in results["metadatas"]
                if "source" in m
            ))
            return sorted(filenames)
        except Exception as e:
            raise VectorStoreError(f"Failed to list documents: {str(e)}")