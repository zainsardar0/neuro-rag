import pytest
from unittest.mock import MagicMock, patch
from app.llm.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter():
    """Create a QueryRewriter with mocked Groq client."""
    with patch("app.llm.query_rewriter.ChatGroq") as mock_groq:
        mock_llm = MagicMock()
        mock_groq.return_value = mock_llm
        rw = QueryRewriter()
        rw.llm = mock_llm
        return rw


def test_rewrite_returns_string(rewriter):
    """Rewriter should return a non-empty string."""
    rewriter.llm.invoke.return_value = MagicMock(
        content="memory consolidation neural networks cognitive processes"
    )
    result = rewriter.rewrite("what does the paper say about memory?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_falls_back_on_exception(rewriter):
    """On LLM failure, should return original query unchanged."""
    rewriter.llm.invoke.side_effect = Exception("API timeout")
    original = "how does attention work?"
    result = rewriter.rewrite(original)
    assert result == original


def test_rewrite_falls_back_on_empty_response(rewriter):
    """On empty LLM response, should return original query."""
    rewriter.llm.invoke.return_value = MagicMock(content="")
    original = "side effects of the treatment"
    result = rewriter.rewrite(original)
    assert result == original