from langgraph.graph import StateGraph, END
from app.core.state import RAGState
from app.retrieval.retriever import Retriever
from app.llm.generator import Generator
from app.llm.query_rewriter import QueryRewriter
from app.core.logger import app_logger

# Constants
MAX_RETRIES = 2
MIN_ANSWER_LENGTH = 50


class RAGWorkflow:
    """
    LangGraph-based RAG workflow with decision making,
    retry logic and validation.
    V2 Phase 1: Query Rewriting
    V2 Phase 2: Reranking
    V2 Phase 3: Hybrid Search
    V2 Phase 4: RAGAS Evaluation support
    """

    def __init__(self):
        """Initialize workflow with retriever, generator, and rewriter."""
        app_logger.info("Initializing RAG Workflow...")
        self.retriever = Retriever(top_k=5)
        self.generator = Generator()
        self.rewriter = QueryRewriter()
        self.graph = self._build_graph()
        app_logger.info("RAG Workflow ready")

    # ─── Nodes ────────────────────────────────────────────

    def analyze_query(self, state: RAGState) -> RAGState:
        """
        Node 1: Analyze and validate the incoming query.
        Checks if query is meaningful before retrieval.
        """
        query = state["query"].strip()
        app_logger.info(f"Analyzing query: {query[:50]}...")

        if len(query) < 5:
            app_logger.warning("Query too short — routing to fallback")
            state["needs_fallback"] = True
            state["retrieved_chunks"] = []
        else:
            state["needs_fallback"] = False
            state["retrieved_chunks"] = []

        state["retry_count"] = 0
        state["is_valid"] = False
        state["retrieval_method"] = ""
        return state

    def rewrite_query(self, state: RAGState) -> RAGState:
        """
        Node 2: Rewrite the query for better vector retrieval.
        Always falls back to original query on failure.
        """
        rewritten = self.rewriter.rewrite(state["query"])
        state["rewritten_query"] = rewritten
        app_logger.info(
            f"Query rewrite | Original: '{state['query'][:50]}' "
            f"| Rewritten: '{rewritten[:50]}'"
        )
        return state

    def retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Node 3: Hybrid retrieval — BM25 + semantic + RRF + reranking.
        """
        app_logger.info("Retrieving relevant documents (hybrid)...")

        retrieval_query = state.get("rewritten_query") or state["query"]

        results, needs_fallback = self.retriever.retrieve_with_fallback(
            retrieval_query
        )

        state["retrieved_chunks"] = results
        state["needs_fallback"] = needs_fallback

        if results:
            state["retrieval_method"] = results[0].get("retrieval_method", "hybrid")
        else:
            state["retrieval_method"] = "none"
            app_logger.warning("No relevant documents found — will use fallback")

        app_logger.info(
            f"Retrieved {len(results)} chunks via {state['retrieval_method']}"
        )

        return state

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Node 4: Generate answer using LLM with retrieved context.
        Uses original query so the answer addresses what user actually asked.
        """
        app_logger.info("Generating answer...")

        result = self.generator.generate_with_fallback(
            query=state["query"],
            chunks=state["retrieved_chunks"]
        )

        state["answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["model"] = result["model"]
        return state

    def validate_answer(self, state: RAGState) -> RAGState:
        """
        Node 5: Validate the generated answer quality.
        """
        answer = state["answer"]
        app_logger.info("Validating answer quality...")

        is_long_enough = len(answer) >= MIN_ANSWER_LENGTH
        is_not_empty = bool(answer.strip())
        not_error_response = "cannot find" not in answer.lower() or len(answer) > 200

        state["is_valid"] = is_long_enough and is_not_empty and not_error_response

        if state["is_valid"]:
            app_logger.info("Answer validation passed")
        else:
            app_logger.warning(
                f"Answer validation failed — "
                f"retry {state['retry_count'] + 1}/{MAX_RETRIES}"
            )

        return state

    def fallback_response(self, state: RAGState) -> RAGState:
        """
        Node 6: Handle cases where no relevant documents found.
        """
        app_logger.warning("Executing fallback response")

        result = self.generator.generate(
            query=state["query"],
            chunks=[]
        )

        state["answer"] = result["answer"]
        state["sources"] = []
        state["model"] = result["model"]
        state["is_valid"] = True
        return state

    def prepare_final_response(self, state: RAGState) -> RAGState:
        """
        Node 7: Package the final response.
        V2 Phase 4: Include retrieved_chunks_text for RAGAS evaluation.
        """
        # V2 Phase 4: Extract chunk texts for RAGAS context evaluation
        retrieved_chunks_text = [
            {"text": chunk["text"], "page": chunk["page"], "source": chunk["source"]}
            for chunk in state["retrieved_chunks"]
        ] if state["retrieved_chunks"] else []

        state["final_response"] = {
            "query": state["query"],
            "rewritten_query": state.get("rewritten_query", ""),
            "answer": state["answer"],
            "sources": state["sources"],
            "model": state["model"],
            "used_fallback": state["needs_fallback"],
            "retry_count": state["retry_count"],
            "retrieval_method": state.get("retrieval_method", "hybrid"),
            "retrieved_chunks_text": retrieved_chunks_text,  # V2 Phase 4: for RAGAS
        }
        app_logger.info("Final response prepared")
        return state

    # ─── Routing Logic ─────────────────────────────────────

    def route_after_analysis(self, state: RAGState) -> str:
        """Route after query analysis."""
        if state["needs_fallback"]:
            return "fallback"
        return "rewrite"

    def route_after_retrieval(self, state: RAGState) -> str:
        """Route after retrieval based on results."""
        if state["needs_fallback"]:
            return "fallback"
        return "generate"

    def route_after_validation(self, state: RAGState) -> str:
        """Route after validation with retry logic."""
        if state["is_valid"]:
            return "finalize"

        state["retry_count"] += 1
        if state["retry_count"] <= MAX_RETRIES:
            app_logger.info(
                f"Retrying generation ({state['retry_count']}/{MAX_RETRIES})"
            )
            return "generate"

        app_logger.warning("Max retries reached — finalizing anyway")
        return "finalize"

    # ─── Graph Builder ─────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""

        graph = StateGraph(RAGState)

        graph.add_node("analyze", self.analyze_query)
        graph.add_node("rewrite", self.rewrite_query)
        graph.add_node("retrieve", self.retrieve_documents)
        graph.add_node("generate", self.generate_answer)
        graph.add_node("validate", self.validate_answer)
        graph.add_node("fallback", self.fallback_response)
        graph.add_node("finalize", self.prepare_final_response)

        graph.set_entry_point("analyze")

        graph.add_conditional_edges(
            "analyze",
            self.route_after_analysis,
            {"rewrite": "rewrite", "fallback": "fallback"}
        )

        graph.add_edge("rewrite", "retrieve")

        graph.add_conditional_edges(
            "retrieve",
            self.route_after_retrieval,
            {"generate": "generate", "fallback": "fallback"}
        )

        graph.add_edge("generate", "validate")

        graph.add_conditional_edges(
            "validate",
            self.route_after_validation,
            {"generate": "generate", "finalize": "finalize"}
        )

        graph.add_edge("fallback", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ─── Public Interface ──────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Run the complete RAG workflow for a query.

        Args:
            query: User's question

        Returns:
            Final response dict with answer, sources, metadata
        """
        app_logger.info("=" * 50)
        app_logger.info(f"Running RAG Workflow for: {query[:50]}...")

        initial_state: RAGState = {
            "query": query,
            "rewritten_query": "",
            "retrieved_chunks": [],
            "needs_fallback": False,
            "retrieval_method": "",
            "answer": "",
            "sources": [],
            "model": "",
            "is_valid": False,
            "retry_count": 0,
            "final_response": {}
        }

        final_state = self.graph.invoke(initial_state)
        app_logger.info("Workflow completed successfully")
        app_logger.info("=" * 50)

        return final_state["final_response"]