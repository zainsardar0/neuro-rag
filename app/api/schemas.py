from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    query: str


class SourceSchema(BaseModel):
    """Schema for a single source citation."""
    file: str
    page: int
    score: float                # Cosine similarity score from ChromaDB
    rerank_score: float         # V2 Phase 2: CrossEncoder reranking score


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    query: str
    rewritten_query: str        # V2 Phase 1: Query Rewriting
    answer: str
    sources: list[SourceSchema]
    model: str
    used_fallback: bool
    retry_count: int
    retrieval_method: str       # V2 Phase 3: Hybrid Search


class IngestRequest(BaseModel):
    """Request schema for the /ingest endpoint."""
    reset: bool = False


class IngestResponse(BaseModel):
    """Response schema for the /ingest endpoint."""
    message: str
    total_chunks: int


class UploadResponse(BaseModel):
    """Response schema for the /upload endpoint."""
    success: bool
    filename: str
    chunks_added: int
    message: str
    total_chunks: int


class DocumentsResponse(BaseModel):
    """Response schema for the /documents endpoint."""
    documents: list[str]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    app_name: str
    environment: str
    total_chunks_in_db: int