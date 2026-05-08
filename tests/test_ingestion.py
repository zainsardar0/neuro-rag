from app.ingestion.document_loader import load_documents, load_single_document
from app.core.exceptions import DocumentLoadError
import pytest


def test_load_single_document():
    """Test loading a single PDF file."""
    pages = load_single_document("data/raw/attention.pdf")
    
    assert len(pages) > 0
    assert "text" in pages[0]
    assert "page" in pages[0]
    assert "source" in pages[0]
    print(f"\n✅ Loaded {len(pages)} pages")
    print(f"✅ First page preview: {pages[0]['text'][:200]}")


def test_load_documents_directory():
    """Test loading all PDFs from a directory."""
    pages = load_documents("data/raw/")
    
    assert len(pages) > 0
    print(f"\n✅ Total pages loaded: {len(pages)}")


def test_file_not_found():
    """Test that missing file raises DocumentLoadError."""
    with pytest.raises(DocumentLoadError):
        load_single_document("data/raw/nonexistent.pdf")


def test_empty_directory():
    """Test that empty directory raises DocumentLoadError."""
    with pytest.raises(DocumentLoadError):
        load_documents("data/processed/")