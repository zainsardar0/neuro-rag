from langgraph.graph import StateGraph, END
from app.core.state import RAGState
from app.retrieval.retriever import Retriever
from app.llm.generator import Generator
from app.core.logger import app_logger

# Constants
MAX_RETRIES = 2
MIN_ANSWER_LENGTH = 50


class RAGWorkflow:
    """
    LangGraph-based RAG workflow with decision making,
    retry logic and validation.
    """

    def __init__(self):
        """Initialize workflow with retriever and generator."""
        app_logger.info("Initializing RAG Workflow...")
        self.retriever = Retriever(top_k=5)
        self.generator = Generator()
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

        # Basic validation
        if len(query) < 5:
            app_logger.warning("Query too short — routing to fallback")
            state["needs_fallback"] = True
            state["retrieved_chunks"] = []
        else:
            state["needs_fallback"] = False
            state["retrieved_chunks"] = []

        state["retry_count"] = 0
        state["is_valid"] = False
        return state

    def retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Node 2: Retrieve relevant chunks from ChromaDB.
        Sets needs_fallback if no relevant docs found.
        """
        app_logger.info("Retrieving relevant documents...")

        results, needs_fallback = self.retriever.retrieve_with_fallback(
            state["query"]
        )

        state["retrieved_chunks"] = results
        state["needs_fallback"] = needs_fallback

        if needs_fallback:
            app_logger.warning("No relevant documents found — will use fallback")
        else:
            app_logger.info(f"Retrieved {len(results)} relevant chunks")

        return state

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Node 3: Generate answer using LLM with retrieved context.
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
        Node 4: Validate the generated answer quality.
        Checks length and basic quality signals.
        """
        answer = state["answer"]
        app_logger.info("Validating answer quality...")

        # Quality checks
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
        Node 5: Handle cases where no relevant documents found.
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
        Node 6: Package the final response.
        """
        state["final_response"] = {
            "query": state["query"],
            "answer": state["answer"],
            "sources": state["sources"],
            "model": state["model"],
            "used_fallback": state["needs_fallback"],
            "retry_count": state["retry_count"]
        }
        app_logger.info("Final response prepared")
        return state

    # ─── Routing Logic ─────────────────────────────────────

    def route_after_analysis(self, state: RAGState) -> str:
        """Route after query analysis."""
        if state["needs_fallback"]:
            return "fallback"
        return "retrieve"

    def route_after_retrieval(self, state: RAGState) -> str:
        """Route after retrieval based on results."""
        if state["needs_fallback"]:
            return "fallback"
        return "generate"

    def route_after_validation(self, state: RAGState) -> str:
        """
        Route after validation.
        Retry if answer is poor quality and retries remain.
        """
        if state["is_valid"]:
            return "finalize"

        state["retry_count"] += 1
        if state["retry_count"] <= MAX_RETRIES:
            app_logger.info(f"Retrying generation ({state['retry_count']}/{MAX_RETRIES})")
            return "generate"

        app_logger.warning("Max retries reached — finalizing anyway")
        return "finalize"

    # ─── Graph Builder ─────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""

        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("analyze", self.analyze_query)
        graph.add_node("retrieve", self.retrieve_documents)
        graph.add_node("generate", self.generate_answer)
        graph.add_node("validate", self.validate_answer)
        graph.add_node("fallback", self.fallback_response)
        graph.add_node("finalize", self.prepare_final_response)

        # Set entry point
        graph.set_entry_point("analyze")

        # Add edges
        graph.add_conditional_edges(
            "analyze",
            self.route_after_analysis,
            {"retrieve": "retrieve", "fallback": "fallback"}
        )

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
            "retrieved_chunks": [],
            "needs_fallback": False,
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