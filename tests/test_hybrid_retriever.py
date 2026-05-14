import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.bm25_retriever import BM25Retriever


def make_chunks(n: int) -> list[dict]:
    """Helper to create n dummy chunks."""
    return [
        {
            "text": f"BERT is a transformer model chunk {i} with attention mechanism",
            "page": i + 1,
            "source": "bert.pdf"
        }
        for i in range(n)
    ]


def test_bm25_index_builds_successfully():
    """BM25 index should build without errors given valid chunks."""
    chunks = make_chunks(5)
    retriever = BM25Retriever(chunks)
    assert retriever.index is not None


def test_bm25_retrieve_returns_results():
    """BM25 should return results for a matching query."""
    chunks = make_chunks(5)
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("BERT transformer attention")
    assert len(results) > 0


def test_bm25_retrieve_has_required_fields():
    """Every BM25 result must have required fields."""
    chunks = make_chunks(5)
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("BERT transformer")
    for result in results:
        assert "text" in result
        assert "page" in result
        assert "source" in result
        assert "bm25_score" in result
        assert "bm25_rank" in result


def test_bm25_retrieve_sorted_by_score():
    """BM25 results must be sorted by score descending."""
    chunks = make_chunks(10)
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("BERT transformer attention mechanism")
    scores = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_chunks_returns_empty():
    """BM25 with no chunks should return empty list."""
    retriever = BM25Retriever([])
    results = retriever.retrieve("BERT transformer")
    assert results == []


def test_bm25_empty_query_returns_empty():
    """BM25 with empty query should return empty list."""
    chunks = make_chunks(5)
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("")
    assert results == []


def test_bm25_top_k_respected():
    """BM25 should return at most top_k results."""
    chunks = make_chunks(20)
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("BERT transformer attention", top_k=5)
    assert len(results) <= 5


def test_rrf_fusion_combines_results():
    """RRF fusion should return more unique chunks than either list alone."""
    from app.retrieval.retriever import Retriever

    with patch("app.retrieval.retriever.Embedder"), \
         patch("app.retrieval.retriever.VectorStore"), \
         patch("app.retrieval.retriever.Reranker"), \
         patch("app.retrieval.retriever.BM25Retriever"):

        retriever = Retriever.__new__(Retriever)
        retriever.top_k = 5

        semantic = [
            {"text": f"semantic chunk {i}", "page": i, "source": "bert.pdf", "score": 0.9 - i * 0.1}
            for i in range(5)
        ]
        bm25 = [
            {"text": f"bm25 chunk {i}", "page": i + 10, "source": "bert.pdf",
             "score": 0.0, "bm25_score": 1.0, "bm25_rank": i + 1}
            for i in range(5)
        ]

        fused = retriever._reciprocal_rank_fusion(semantic, bm25)
        assert len(fused) >= max(len(semantic), len(bm25))


def test_rrf_boosts_chunks_in_both_lists():
    """Chunks appearing in both lists should have higher RRF score."""
    from app.retrieval.retriever import Retriever

    with patch("app.retrieval.retriever.Embedder"), \
         patch("app.retrieval.retriever.VectorStore"), \
         patch("app.retrieval.retriever.Reranker"), \
         patch("app.retrieval.retriever.BM25Retriever"):

        retriever = Retriever.__new__(Retriever)
        retriever.top_k = 5

        shared_chunk = {
            "text": "BERT bidirectional transformer",
            "page": 1,
            "source": "bert.pdf",
            "score": 0.9
        }
        semantic = [shared_chunk]
        bm25 = [{**shared_chunk, "bm25_score": 1.0, "bm25_rank": 1}]

        fused = retriever._reciprocal_rank_fusion(semantic, bm25)

        # Shared chunk should have higher RRF score than 1/(60+1) alone
        assert fused[0]["rrf_score"] > 1 / (60 + 1)