from typing import TypedDict


class RAGState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Every node reads from and writes to this state.
    """

    # Input
    query: str

    # Retrieval
    retrieved_chunks: list[dict]
    needs_fallback: bool

    # Generation
    answer: str
    sources: list[dict]
    model: str

    # Validation
    is_valid: bool
    retry_count: int

    # Final
    final_response: dict