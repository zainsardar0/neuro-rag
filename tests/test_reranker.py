import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.reranker import Reranker


@pytest.fixture
def reranker():
    """Create a Reranker with mocked CrossEncoder model."""
    with patch("app.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        mock_ce.return_value = mock_model
        r = Reranker()
        r.model = mock_model
        return r


def make_chunks(n: int) -> list[dict]:
    """Helper to create n dummy chunks."""
    return [
        {
            "text": f"Sample text for chunk {i}",
            "page": i,
            "source": "bert.pdf",
            "score": round(0.9 - i * 0.05, 2)
        }
        for i in range(n)
    ]


def test_rerank_returns_rerank_score(reranker):
    """Every returned chunk must have a rerank_score field."""
    chunks = make_chunks(5)
    # CrossEncoder returns raw scores (higher = more relevant)
    reranker.model.predict.return_value = [2.1, 0.5, 1.8, -0.3, 1.1]
    result = reranker.rerank("what is BERT?", chunks)
    assert all("rerank_score" in chunk for chunk in result)


def test_rerank_sorts_by_score_descending(reranker):
    """Chunks must be sorted by rerank_score highest first."""
    chunks = make_chunks(5)
    reranker.model.predict.return_value = [0.5, 2.1, -0.3, 1.8, 1.1]
    result = reranker.rerank("what is BERT?", chunks)
    scores = [chunk["rerank_score"] for chunk in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_returns_top_n(reranker):
    """Reranker must return at most TOP_N_AFTER_RERANK chunks."""
    from app.retrieval.reranker import TOP_N_AFTER_RERANK
    chunks = make_chunks(10)
    reranker.model.predict.return_value = [float(i) for i in range(10)]
    result = reranker.rerank("what is BERT?", chunks)
    assert len(result) <= TOP_N_AFTER_RERANK


def test_rerank_fallback_on_single_chunk(reranker):
    """Single chunk should skip reranking and return as-is with rerank_score."""
    chunks = make_chunks(1)
    result = reranker.rerank("what is BERT?", chunks)
    assert len(result) == 1
    assert "rerank_score" in result[0]
    # predict should NOT be called for single chunk
    reranker.model.predict.assert_not_called()


def test_rerank_fallback_on_exception(reranker):
    """On CrossEncoder failure, return original chunks unchanged."""
    chunks = make_chunks(5)
    reranker.model.predict.side_effect = Exception("Model error")
    result = reranker.rerank("what is BERT?", chunks)
    assert len(result) > 0
    assert all("rerank_score" in chunk for chunk in result)


def test_rerank_empty_chunks(reranker):
    """Empty chunk list should return empty list immediately."""
    result = reranker.rerank("what is BERT?", [])
    assert result == []
    reranker.model.predict.assert_not_called()