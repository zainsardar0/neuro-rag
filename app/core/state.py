from typing import TypedDict


class RAGState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Every node reads from and writes to this state.
    """

    # Input
    query: str                      # Original user query (never modified)

    # V2 Phase 1: Query Rewriting
    rewritten_query: str            # LLM-optimized query for retrieval

    # Retrieval
    retrieved_chunks: list[dict]    # V2: chunks include rerank_score, rrf_score
    needs_fallback: bool

    # V2 Phase 3: Hybrid Search
    retrieval_method: str           # "hybrid", "semantic", or "bm25"

    # Generation
    answer: str
    sources: list[dict]
    model: str

    # Validation
    is_valid: bool
    retry_count: int

    # Final
    final_response: dict