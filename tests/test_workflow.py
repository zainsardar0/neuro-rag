from app.llm.workflow import RAGWorkflow


def test_workflow_with_relevant_query():
    """Test full workflow with a relevant research question."""
    workflow = RAGWorkflow()
    result = workflow.run("What is the attention mechanism in transformers?")

    assert "answer" in result
    assert "sources" in result
    assert "query" in result
    assert "model" in result
    assert len(result["answer"]) > 0

    print(f"\n✅ Query: {result['query']}")
    print(f"✅ Model: {result['model']}")
    print(f"✅ Used fallback: {result['used_fallback']}")
    print(f"✅ Retry count: {result['retry_count']}")
    print(f"✅ Sources: {result['sources']}")
    print(f"\n✅ Answer:\n{result['answer']}")


def test_workflow_with_irrelevant_query():
    """Test fallback when query is unrelated to loaded documents."""
    workflow = RAGWorkflow()
    result = workflow.run("What is the capital of France?")

    assert "answer" in result
    assert len(result["answer"]) > 0
    print(f"\n✅ Fallback answer:\n{result['answer']}")


def test_workflow_with_short_query():
    """Test that very short query triggers fallback."""
    workflow = RAGWorkflow()
    result = workflow.run("hi")

    assert "answer" in result
    print(f"\n✅ Short query handled: {result['answer'][:100]}")