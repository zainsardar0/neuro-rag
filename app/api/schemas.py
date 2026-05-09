from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    query: str


class SourceSchema(BaseModel):
    """Schema for a single source citation."""
    file: str
    page: int
    score: float


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    query: str
    rewritten_query: str        # V2: Query Rewriting
    answer: str
    sources: list[SourceSchema]
    model: str
    used_fallback: bool
    retry_count: int


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