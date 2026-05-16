import pytest
from unittest.mock import MagicMock, patch
from app.evaluation.ragas_evaluator import RagasEvaluator


@pytest.fixture
def mock_ragas_evaluator():
    """Create a RagasEvaluator with all external dependencies mocked."""
    with patch("app.evaluation.ragas_evaluator.RAGWorkflow") as mock_workflow, \
         patch("app.evaluation.ragas_evaluator.ChatGroq"), \
         patch("app.evaluation.ragas_evaluator.HuggingFaceEmbeddings"), \
         patch("app.evaluation.ragas_evaluator.LangchainLLMWrapper"), \
         patch("app.evaluation.ragas_evaluator.LangchainEmbeddingsWrapper"), \
         patch("app.evaluation.ragas_evaluator.Faithfulness"), \
         patch("app.evaluation.ragas_evaluator.LLMContextPrecisionWithoutReference"):

        mock_workflow.return_value.run.return_value = {
            "answer": "GPT-2 is a large language model [Source 1].",
            "sources": [{"file": "gpt2.pdf", "page": 1,
                        "score": 0.9, "rerank_score": 0.95}],
            "retrieved_chunks_text": [
                {"text": "GPT-2 is trained on WebText dataset.", "page": 1, "source": "gpt2.pdf"},
                {"text": "GPT-2 uses transformer architecture.", "page": 2, "source": "gpt2.pdf"},
            ],
            "used_fallback": False,
            "retrieval_method": "hybrid",
            "rewritten_query": "GPT-2 transformer model",
            "model": "llama-3.3-70b-versatile",
            "retry_count": 0
        }

        evaluator = RagasEvaluator()
        evaluator.workflow = mock_workflow.return_value
        return evaluator


def test_ragas_evaluator_initializes(mock_ragas_evaluator):
    """RagasEvaluator should initialize without errors."""
    assert mock_ragas_evaluator is not None
    assert mock_ragas_evaluator.workflow is not None
    assert mock_ragas_evaluator.metrics is not None


def test_build_dataset_returns_correct_structure(mock_ragas_evaluator):
    """Dataset should have required RAGAS fields for each query."""
    test_cases = ["What is GPT-2?", "How does MLM work?"]
    dataset = mock_ragas_evaluator._build_dataset(test_cases)

    assert len(dataset) == 2
    for item in dataset:
        assert "user_input" in item
        assert "response" in item
        assert "retrieved_contexts" in item
        assert isinstance(item["retrieved_contexts"], list)
        assert len(item["retrieved_contexts"]) > 0


def test_build_dataset_uses_chunk_texts(mock_ragas_evaluator):
    """Dataset should use real chunk texts not source metadata."""
    dataset = mock_ragas_evaluator._build_dataset(["What is GPT-2?"])
    assert dataset[0]["retrieved_contexts"][0] == "GPT-2 is trained on WebText dataset."


def test_build_dataset_handles_workflow_failure(mock_ragas_evaluator):
    """Dataset builder should handle workflow errors gracefully."""
    mock_ragas_evaluator.workflow.run.side_effect = Exception("LLM error")
    dataset = mock_ragas_evaluator._build_dataset(["What is GPT-2?"])

    assert len(dataset) == 1
    assert dataset[0]["response"] == "Error generating response."
    assert dataset[0]["retrieved_contexts"] == ["No context available."]


def test_evaluate_returns_required_keys(mock_ragas_evaluator):
    """Evaluate should return dict with all required metric keys."""
    import pandas as pd

    mock_results = MagicMock()
    mock_results.to_pandas.return_value = pd.DataFrame([{
        "faithfulness": 0.85,
        "llm_context_precision_without_reference": 0.80,
        "user_input": "What is GPT-2?"
    }])

    with patch("app.evaluation.ragas_evaluator.evaluate",
               return_value=mock_results), \
         patch("app.evaluation.ragas_evaluator.EvaluationDataset"):

        results = mock_ragas_evaluator.evaluate(["What is GPT-2?"])

    assert "faithfulness" in results
    assert "response_relevancy" in results
    assert "context_precision" in results
    assert "overall_score" in results
    assert "total_queries" in results
    assert "per_query_results" in results


def test_evaluate_calculates_overall_score(mock_ragas_evaluator):
    """Overall score should be average of faithfulness and precision."""
    import pandas as pd

    mock_results = MagicMock()
    mock_results.to_pandas.return_value = pd.DataFrame([{
        "faithfulness": 0.8,
        "llm_context_precision_without_reference": 0.6,
        "user_input": "What is GPT-2?"
    }])

    with patch("app.evaluation.ragas_evaluator.evaluate",
               return_value=mock_results), \
         patch("app.evaluation.ragas_evaluator.EvaluationDataset"):

        results = mock_ragas_evaluator.evaluate(["What is GPT-2?"])

    expected_overall = round((0.8 + 0.6) / 2, 4)
    assert results["overall_score"] == expected_overall


def test_evaluate_uses_default_test_cases(mock_ragas_evaluator):
    """Evaluate with no args should use default RAGAS_TEST_CASES."""
    import pandas as pd
    from app.evaluation.ragas_evaluator import RAGAS_TEST_CASES

    mock_results = MagicMock()
    mock_results.to_pandas.return_value = pd.DataFrame([{
        "faithfulness": 0.85,
        "llm_context_precision_without_reference": 0.80,
        "user_input": q
    } for q in RAGAS_TEST_CASES])

    with patch("app.evaluation.ragas_evaluator.evaluate",
               return_value=mock_results), \
         patch("app.evaluation.ragas_evaluator.EvaluationDataset"):

        results = mock_ragas_evaluator.evaluate()

    assert results["total_queries"] == len(RAGAS_TEST_CASES)


def test_safe_score_handles_nan():
    """safe_score should return 0.0 for NaN values."""
    from app.evaluation.ragas_evaluator import safe_score
    import math
    assert safe_score(float("nan")) == 0.0
    assert safe_score(float("inf")) == 0.0
    assert safe_score(0.85) == 0.85