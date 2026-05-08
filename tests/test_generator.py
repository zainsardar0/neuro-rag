from app.llm.generator import Generator
from app.retrieval.retriever import Retriever
from app.core.exceptions import LLMError
import pytest


def test_generate_with_context():
    """Test answer generation with retrieved context."""
    retriever = Retriever(top_k=5)
    generator = Generator()

    query = "What is the attention mechanism in transformers?"
    chunks = retriever.retrieve(query)
    result = generator.generate(query, chunks)

    assert "answer" in result
    assert "sources" in result
    assert "model" in result
    assert len(result["answer"]) > 0

    print(f"\n✅ Model used: {result['model']}")
    print(f"✅ Sources cited: {result['sources']}")
    print(f"\n✅ Answer:\n{result['answer']}")


def test_generate_with_fallback():
    """Test fallback when no chunks provided."""
    generator = Generator()
    result = generator.generate_with_fallback(
        "What is quantum computing?", []
    )

    assert result["used_fallback"] == True
    assert len(result["answer"]) > 0
    print(f"\n✅ Fallback answer:\n{result['answer']}")


def test_empty_query_raises_error():
    """Test that empty query raises LLMError."""
    generator = Generator()
    with pytest.raises(LLMError):
        generator.generate("", [])