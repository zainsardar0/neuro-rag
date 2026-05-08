from app.retrieval.retriever import Retriever
from app.core.exceptions import RetrievalError
import pytest


def test_retrieve_relevant_chunks():
    """Test retrieval returns relevant chunks for a query."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve("What is the attention mechanism?")

    assert len(results) > 0
    assert "text" in results[0]
    assert "page" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]

    print(f"\n✅ Retrieved {len(results)} chunks")
    print(f"✅ Top score: {results[0]['score']}")
    print(f"✅ Top chunk preview: {results[0]['text'][:200]}")


def test_retrieve_scores_above_threshold():
    """Test all retrieved chunks are above similarity threshold."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve("What is multi-head attention?")

    for result in results:
        assert result["score"] >= 0.3

    print(f"\n✅ All {len(results)} chunks above threshold")


def test_retrieve_with_fallback_finds_results():
    """Test retrieve_with_fallback returns correct flag when results found."""
    retriever = Retriever(top_k=5)
    results, needs_fallback = retriever.retrieve_with_fallback(
        "What is the transformer architecture?"
    )

    assert needs_fallback == False
    assert len(results) > 0
    print(f"\n✅ needs_fallback: {needs_fallback}")
    print(f"✅ Results found: {len(results)}")


def test_empty_query_raises_error():
    """Test that empty query raises RetrievalError."""
    retriever = Retriever()
    with pytest.raises(RetrievalError):
        retriever.retrieve("")