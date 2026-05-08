class NeuroRAGException(Exception):
    """Base exception for all NeuroRAG errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DocumentLoadError(NeuroRAGException):
    """Raised when a document fails to load or parse."""
    pass


class ChunkingError(NeuroRAGException):
    """Raised when document chunking fails."""
    pass


class EmbeddingError(NeuroRAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(NeuroRAGException):
    """Raised when vector database operations fail."""
    pass


class RetrievalError(NeuroRAGException):
    """Raised when document retrieval fails."""
    pass


class LLMError(NeuroRAGException):
    """Raised when LLM generation fails."""
    pass


class ConfigurationError(NeuroRAGException):
    """Raised when configuration is invalid or missing."""
    pass