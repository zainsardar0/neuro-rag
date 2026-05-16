import math
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from app.llm.workflow import RAGWorkflow
from app.core.config import get_settings
from app.core.logger import app_logger

settings = get_settings()

MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RAGAS_TEST_CASES = [
    "What is GPT-2 and what makes it different from previous language models?",
    "What dataset was used to train GPT-2?",
    "How does GPT-2 perform on language modeling benchmarks?",
    "What is zero-shot learning in the context of GPT-2?",
    "What are the different model sizes of GPT-2?",
]


def safe_score(value) -> float:
    """Convert NaN/inf to 0.0 for JSON safety."""
    try:
        v = float(value)
        return 0.0 if math.isnan(v) or math.isinf(v) else round(v, 4)
    except Exception:
        return 0.0


class RagasEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.

    Metrics used (all reference-free — no ground truth needed):
    - Faithfulness: Is the answer grounded in retrieved context?
    - LLMContextPrecisionWithoutReference: Are retrieved chunks relevant?

    Note: ResponseRelevancy removed — Groq API does not support n>1
    generations which RAGAS requires for this metric.
    """

    def __init__(self):
        """Initialize RAGAS evaluator with Groq LLM and HF embeddings."""
        try:
            app_logger.info("Initializing RagasEvaluator...")

            self.workflow = RAGWorkflow()

            groq_llm = ChatGroq(
                model=MODEL_NAME,
                temperature=0.0,
                api_key=settings.groq_api_key
            )
            self.evaluator_llm = LangchainLLMWrapper(groq_llm)

            hf_embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

            self.metrics = [
                Faithfulness(llm=self.evaluator_llm),
                LLMContextPrecisionWithoutReference(llm=self.evaluator_llm),
            ]

            app_logger.info("RagasEvaluator initialized successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize RagasEvaluator: {str(e)}")

    def _build_dataset(self, test_cases: list[str]) -> list[dict]:
        """
        Run workflow for each test case and collect results.

        Args:
            test_cases: List of query strings

        Returns:
            List of dicts with user_input, response, retrieved_contexts
        """
        dataset = []

        for query in test_cases:
            try:
                app_logger.info(f"Running RAGAS query: {query[:50]}...")
                result = self.workflow.run(query)

                # V2 Phase 4: Use real chunk texts for RAGAS context evaluation
                contexts = [
                    chunk["text"]
                    for chunk in result.get("retrieved_chunks_text", [])
                    if chunk.get("text")
                ]

                # Fallback: use source metadata if chunk texts not available
                if not contexts and result.get("sources"):
                    contexts = [
                        f"Source: {s['file']} Page: {s['page']}"
                        for s in result["sources"]
                    ]

                # RAGAS needs at least one context
                if not contexts:
                    contexts = ["No relevant context found in documents."]

                dataset.append({
                    "user_input": query,
                    "response": result["answer"],
                    "retrieved_contexts": contexts,
                })

                app_logger.info(f"Query completed — contexts: {len(contexts)}")

            except Exception as e:
                app_logger.warning(f"Failed to run query '{query[:30]}': {e}")
                dataset.append({
                    "user_input": query,
                    "response": "Error generating response.",
                    "retrieved_contexts": ["No context available."],
                })

        return dataset

    def evaluate(self, test_cases: list[str] = None) -> dict:
        """
        Run RAGAS evaluation on test cases.

        Args:
            test_cases: Optional list of queries. Uses defaults if None.

        Returns:
            Dict with per-metric scores and overall summary
        """
        if test_cases is None:
            test_cases = RAGAS_TEST_CASES

        app_logger.info(
            f"Starting RAGAS evaluation with {len(test_cases)} queries..."
        )

        try:
            raw_dataset = self._build_dataset(test_cases)
            evaluation_dataset = EvaluationDataset.from_list(raw_dataset)

            app_logger.info("Running RAGAS metrics...")
            results = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
            )

            scores_df = results.to_pandas()

            faithfulness_score = safe_score(scores_df["faithfulness"].mean())
            precision_score = safe_score(
                scores_df["llm_context_precision_without_reference"].mean()
            )
            overall_score = safe_score(
                (faithfulness_score + precision_score) / 2
            )

            # Sanitize per-query results for JSON compliance
            per_query = scores_df.to_dict(orient="records")
            for row in per_query:
                for key, val in row.items():
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        row[key] = 0.0

            summary = {
                "total_queries": len(test_cases),
                "faithfulness": faithfulness_score,
                "response_relevancy": 0.0,
                "context_precision": precision_score,
                "overall_score": overall_score,
                "per_query_results": per_query,
            }

            app_logger.info(
                f"RAGAS evaluation complete — "
                f"Overall: {overall_score} | "
                f"Faithfulness: {faithfulness_score} | "
                f"Precision: {precision_score}"
            )

            return summary

        except Exception as e:
            app_logger.error(f"RAGAS evaluation failed: {str(e)}")
            raise RuntimeError(f"RAGAS evaluation failed: {str(e)}")