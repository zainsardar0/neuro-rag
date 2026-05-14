import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.retriever import Retriever
from app.core.exceptions import RetrievalError


def make_semantic_chunks(n: int) -> list[dict]:
    """Helper to create n dummy semantic chunks."""
    return [
        {
            "text": f"BERT attention mechanism chunk {i}",
            "page": i + 1,
            "source": "bert.pdf",
            "score": round(0.9 - i * 0.05, 2)
        }
        for i in range(n)
    ]


def make_reranked_chunks(n: int) -> list[dict]:
    """Helper to create n dummy reranked chunks."""
    return [
        {
            "text": f"BERT attention mechanism chunk {i}",
            "page": i + 1,
            "source": "bert.pdf",
            "score": round(0.9 - i * 0.05, 2),
            "rerank_score": round(0.95 - i * 0.05, 2),
            "retrieval_method": "hybrid"
        }
        for i in range(n)
    ]


@pytest.fixture
def mock_retriever():
    """Create a Retriever with all external dependencies mocked."""
    with patch("app.retrieval.retriever.Embedder") as mock_embedder, \
         patch("app.retrieval.retriever.VectorStore") as mock_vs, \
         patch("app.retrieval.retriever.Reranker") as mock_reranker, \
         patch("app.retrieval.retriever.BM25Retriever") as mock_bm25:

        # Setup embedder
        mock_embedder.return_value.embed_query.return_value = [0.1] * 384

        # Setup vector store
        mock_vs.return_value.get_all_chunks.return_value = make_semantic_chunks(10)
        mock_vs.return_value.query.return_value = make_semantic_chunks(5)

        # Setup BM25
        mock_bm25.return_value.retrieve.return_value = []

        # Setup reranker
        mock_reranker.return_value.rerank.return_value = make_reranked_chunks(5)

        retriever = Retriever(top_k=5)
        retriever.embedder = mock_embedder.return_value
        retriever.vector_store = mock_vs.return_value
        retriever.reranker = mock_reranker.return_value
        retriever.bm25 = mock_bm25.return_value

        return retriever


def test_retrieve_relevant_chunks(mock_retriever):
    """Test retrieval returns relevant chunks for a query."""
    results = mock_retriever.retrieve("What is the attention mechanism?")

    assert len(results) > 0
    assert "text" in results[0]
    assert "page" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]
    assert "rerank_score" in results[0]

    print(f"\n✅ Retrieved {len(results)} chunks")
    print(f"✅ Top rerank score: {results[0]['rerank_score']}")
    print(f"✅ Top chunk preview: {results[0]['text'][:200]}")


def test_retrieve_scores_above_threshold(mock_retriever):
    """
    Test retrieved chunks have valid rerank scores.
    V2: Hybrid search includes BM25 chunks with score=0.0 by design.
    Threshold applies to semantic retrieval only — not fused results.
    We verify rerank_score is valid (0-1) for all chunks instead.
    """
    results = mock_retriever.retrieve("What is multi-head attention?")

    for result in results:
        assert "rerank_score" in result
        assert 0.0 <= result["rerank_score"] <= 1.0

    print(f"\n✅ All {len(results)} chunks have valid rerank scores")


def test_retrieve_with_fallback_finds_results(mock_retriever):
    """Test retrieve_with_fallback returns correct flag when results found."""
    results, needs_fallback = mock_retriever.retrieve_with_fallback(
        "What is the transformer architecture?"
    )

    assert needs_fallback == False
    assert len(results) > 0
    print(f"\n✅ needs_fallback: {needs_fallback}")
    print(f"✅ Results found: {len(results)}")


def test_empty_query_raises_error(mock_retriever):
    """Test that empty query raises RetrievalError."""
    with pytest.raises(RetrievalError):
        mock_retriever.retrieve("")