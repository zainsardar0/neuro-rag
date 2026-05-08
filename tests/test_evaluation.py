from app.evaluation.evaluator import Evaluator


def test_full_evaluation():
    """Run full evaluation suite and print report."""
    evaluator = Evaluator()
    report = evaluator.evaluate_all()

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
    print("\n  Per-Case Results:")
    print("-" * 60)

    for i, result in enumerate(report["results"], 1):
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"\n  Case {i}: {status}")
        print(f"  Query:          {result['query'][:50]}...")
        print(f"  Keyword Score:  {result['keyword_score'] * 100:.0f}%")
        print(f"  Keywords Found: {result['keywords_found']}")
        print(f"  Missing:        {result['keywords_missing']}")
        print(f"  Fallback OK:    {result['fallback_correct']}")
        print(f"  Has Citations:  {result['has_citations']}")
        print(f"  Answer Length:  {result['answer_length']} chars")

    print("\n" + "=" * 60)

    # Assertions
    assert report["pass_rate"] >= 0.6, \
        f"Pass rate too low: {report['pass_rate']}"
    assert report["fallback_accuracy"] >= 0.8, \
        f"Fallback accuracy too low: {report['fallback_accuracy']}"