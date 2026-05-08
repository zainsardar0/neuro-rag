from app.llm.workflow import RAGWorkflow
from app.core.logger import app_logger


# Test questions with expected keywords in answers
TEST_CASES = [
    {
        "query": "What is the attention mechanism in transformers?",
        "expected_keywords": ["query", "key", "value", "weighted"],
        "should_use_fallback": False
    },
    {
        "query": "What is multi-head attention?",
        "expected_keywords": ["head", "parallel", "attention"],
        "should_use_fallback": False
    },
    {
        "query": "What are the encoder and decoder components?",
        "expected_keywords": ["encoder", "decoder", "layer"],
        "should_use_fallback": False
    },
    {
        "query": "What optimizer was used for training?",
        "expected_keywords": ["adam", "optimizer", "learning rate"],
        "should_use_fallback": False
    },
    {
        "query": "What is the recipe for pizza?",
        "expected_keywords": [],
        "should_use_fallback": True
    }
]


class Evaluator:
    """
    Evaluates RAG system quality using test cases.
    Measures retrieval quality, answer relevance, and fallback accuracy.
    """

    def __init__(self):
        self.workflow = RAGWorkflow()

    def evaluate_single(self, test_case: dict) -> dict:
        """
        Evaluate a single test case.

        Args:
            test_case: Dict with query, expected_keywords, should_use_fallback

        Returns:
            Dict with evaluation results
        """
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        should_use_fallback = test_case["should_use_fallback"]

        app_logger.info(f"Evaluating: {query[:50]}...")

        # Run workflow
        result = self.workflow.run(query)
        answer = result["answer"].lower()

        # Check keyword presence
        keywords_found = []
        keywords_missing = []
        for keyword in expected_keywords:
            if keyword.lower() in answer:
                keywords_found.append(keyword)
            else:
                keywords_missing.append(keyword)

        # Calculate keyword score
        if expected_keywords:
            keyword_score = len(keywords_found) / len(expected_keywords)
        else:
            keyword_score = 1.0

        # Check fallback accuracy
        fallback_correct = result["used_fallback"] == should_use_fallback

        # Check citations present
        has_citations = len(result["sources"]) > 0

        # Overall pass/fail
        passed = (
            keyword_score >= 0.5 and
            fallback_correct and
            len(result["answer"]) > 50
        )

        return {
            "query": query,
            "passed": passed,
            "keyword_score": round(keyword_score, 2),
            "keywords_found": keywords_found,
            "keywords_missing": keywords_missing,
            "fallback_correct": fallback_correct,
            "has_citations": has_citations,
            "used_fallback": result["used_fallback"],
            "answer_length": len(result["answer"]),
            "sources_count": len(result["sources"])
        }

    def evaluate_all(self) -> dict:
        """
        Run evaluation on all test cases and return summary report.

        Returns:
            Dict with per-case results and overall metrics
        """
        app_logger.info("Starting full evaluation...")
        app_logger.info(f"Running {len(TEST_CASES)} test cases")

        results = []
        for test_case in TEST_CASES:
            result = self.evaluate_single(test_case)
            results.append(result)

        # Calculate overall metrics
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_keyword_score = sum(r["keyword_score"] for r in results) / total
        fallback_accuracy = sum(1 for r in results if r["fallback_correct"]) / total
        citation_rate = sum(1 for r in results if r["has_citations"]) / total

        summary = {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 2),
            "avg_keyword_score": round(avg_keyword_score, 2),
            "fallback_accuracy": round(fallback_accuracy, 2),
            "citation_rate": round(citation_rate, 2),
            "results": results
        }

        app_logger.info(f"Evaluation complete — Pass rate: {summary['pass_rate']}")
        return summary
    