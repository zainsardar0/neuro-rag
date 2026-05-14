import pytest
from unittest.mock import MagicMock, patch
from app.evaluation.evaluator import Evaluator


def make_mock_result(query: str, answer: str, used_fallback: bool) -> dict:
    """Helper to create a mock workflow result."""
    return {
        "query": query,
        "rewritten_query": query,
        "answer": answer,
        "sources": [{"file": "bert.pdf", "page": 1, "score": 0.9,
                     "rerank_score": 0.95}] if not used_fallback else [],
        "model": "llama-3.3-70b-versatile",
        "used_fallback": used_fallback,
        "retry_count": 0,
        "retrieval_method": "hybrid"
    }


def test_full_evaluation():
    """Run full evaluation suite with mocked workflow."""

    # Mock answers for each evaluation case
    mock_answers = {
        "What is the attention mechanism in transformers?": make_mock_result(
            "What is the attention mechanism in transformers?",
            "The attention mechanism uses query, key, and value matrices to compute weighted representations [Source 1].",
            used_fallback=False
        ),
        "What is multi-head attention?": make_mock_result(
            "What is multi-head attention?",
            "Multi-head attention runs attention in parallel across multiple heads [Source 1].",
            used_fallback=False
        ),
        "What are the encoder and decoder components?": make_mock_result(
            "What are the encoder and decoder components?",
            "The encoder processes input through multiple layer stacks. The decoder generates output [Source 1].",
            used_fallback=False
        ),
        "What optimizer was used for training?": make_mock_result(
            "What optimizer was used for training?",
            "The Adam optimizer was used with a specific learning rate schedule [Source 1].",
            used_fallback=False
        ),
        "What is the recipe for pizza?": make_mock_result(
            "What is the recipe for pizza?",
            "I cannot find sufficient information in the provided documents.",
            used_fallback=True
        ),
    }

    def mock_run(query: str) -> dict:
        return mock_answers.get(query, make_mock_result(query, "No answer.", True))

    with patch.object(Evaluator, "__init__", lambda self: None):
        evaluator = Evaluator.__new__(Evaluator)

        # Mock the workflow
        evaluator.workflow = MagicMock()
        evaluator.workflow.run.side_effect = mock_run

        # Run real evaluate_all logic
        from app.evaluation.evaluator import Evaluator as RealEvaluator
        report = RealEvaluator.evaluate_all(evaluator)

    print("\n")
    print("=" * 60)
    print("         NeuroRAG EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total Cases:       {report['total_cases']}")
    print(f"  Passed:            {report['passed']}")
    print(f"  Failed:            {report['failed']}")
    print(f"  Pass Rate:         {report['pass_rate'] * 100:.0f}%")
    print(f"  Avg Keyword Score: {report['avg_keyword_score'] * 100:.0f}%")
    print(f"  Fallback Accuracy: {report['fallback_accuracy'] * 100:.0f}%")
    print(f"  Citation Rate:     {report['citation_rate'] * 100:.0f}%")
    print("=" * 60)

    assert report["pass_rate"] >= 0.6, \
        f"Pass rate too low: {report['pass_rate']}"
    assert report["fallback_accuracy"] >= 0.8, \
        f"Fallback accuracy too low: {report['fallback_accuracy']}"