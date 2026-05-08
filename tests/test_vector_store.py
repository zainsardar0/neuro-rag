from app.ingestion.document_loader import load_single_document
from app.ingestion.chunker import chunk_documents
from app.embedding.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.core.exceptions import VectorStoreError
import pytest


def test_store_and_count():
    """Test storing chunks and counting them."""
    pages = load_single_document("data/raw/attention.pdf")
    chunks = chunk_documents(pages)

    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks[:10])

    store = VectorStore()
    store.reset()  # Start fresh
    store.add_chunks(embedded_chunks)

    assert store.count() == 10
    print(f"\n✅ Stored and verified 10 chunks in ChromaDB")


def test_query_returns_results():
    """Test querying ChromaDB returns relevant results."""
    embedder = Embedder()
    store = VectorStore()

    query_embedding = embedder.embed_query("What is the attention mechanism?")
    results = store.query(query_embedding, top_k=3)

    assert len(results) > 0
    assert "text" in results[0]
    assert "page" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]

    print(f"\n✅ Retrieved {len(results)} results")
    print(f"✅ Top result score: {results[0]['score']}")
    print(f"✅ Top result preview: {results[0]['text'][:200]}")


def test_empty_chunks_raises_error():
    """Test that empty chunks raises VectorStoreError."""
    store = VectorStore()
    with pytest.raises(VectorStoreError):
        store.add_chunks([])


def test_reset_clears_collection():
    """Test that reset empties the collection."""
    store = VectorStore()
    store.reset()
    assert store.count() == 0
    print(f"\n✅ Collection reset successfully")