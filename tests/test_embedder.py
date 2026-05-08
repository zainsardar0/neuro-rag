from app.ingestion.document_loader import load_single_document
from app.ingestion.chunker import chunk_documents
from app.embedding.embedder import Embedder
from app.core.exceptions import EmbeddingError
import pytest


def test_embed_chunks():
    """Test embedding generation for document chunks."""
    pages = load_single_document("data/raw/attention.pdf")
    chunks = chunk_documents(pages)

    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks[:5])  # Test with first 5 only

    assert len(embedded_chunks) == 5
    assert "embedding" in embedded_chunks[0]
    assert len(embedded_chunks[0]["embedding"]) == 384  # all-MiniLM-L6-v2 dimensions
    print(f"\n✅ Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")
    print(f"✅ Sample embedding (first 5 values): {embedded_chunks[0]['embedding'][:5]}")


def test_embed_query():
    """Test embedding a single query."""
    embedder = Embedder()
    embedding = embedder.embed_query("What is the attention mechanism?")

    assert len(embedding) == 384
    print(f"\n✅ Query embedding dimensions: {len(embedding)}")


def test_empty_chunks_raises_error():
    """Test that empty chunks raises EmbeddingError."""
    embedder = Embedder()
    with pytest.raises(EmbeddingError):
        embedder.embed_chunks([])


def test_empty_query_raises_error():
    """Test that empty query raises EmbeddingError."""
    embedder = Embedder()
    with pytest.raises(EmbeddingError):
        embedder.embed_query("")