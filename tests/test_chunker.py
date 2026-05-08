from app.ingestion.document_loader import load_single_document
from app.ingestion.chunker import chunk_documents
from app.core.exceptions import ChunkingError
import pytest


def test_chunking_basic():
    """Test basic chunking of a real document."""
    pages = load_single_document("data/raw/attention.pdf")
    chunks = chunk_documents(pages)

    assert len(chunks) > 0
    print(f"\n✅ Total chunks created: {len(chunks)}")
    print(f"✅ First chunk preview: {chunks[0]['text'][:200]}")


def test_chunk_metadata():
    """Test that every chunk has required metadata."""
    pages = load_single_document("data/raw/attention.pdf")
    chunks = chunk_documents(pages)

    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "source" in chunk
        assert "chunk_id" in chunk

    print(f"\n✅ All {len(chunks)} chunks have correct metadata")


def test_chunk_size():
    """Test that chunks respect the size limit."""
    pages = load_single_document("data/raw/attention.pdf")
    chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)

    oversized = [c for c in chunks if len(c["text"]) > 1200]
    print(f"\n✅ Total chunks: {len(chunks)}")
    print(f"✅ Oversized chunks: {len(oversized)}")
    assert len(oversized) == 0


def test_empty_pages_raises_error():
    """Test that empty input raises ChunkingError."""
    with pytest.raises(ChunkingError):
        chunk_documents([])