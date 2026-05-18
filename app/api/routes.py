from fastapi import APIRouter, HTTPException, UploadFile, File
from app.api.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    UploadResponse, DocumentsResponse,
    HealthResponse, CacheStatsResponse,
    RagasEvaluationRequest, RagasEvaluationResponse
)
from app.llm.workflow import RAGWorkflow
from app.retrieval.vector_store import VectorStore
from app.core.config import get_settings
from app.core.logger import app_logger
from app.core.cache import CacheManager          # V2 Phase 5
from scripts.ingest import run_ingestion, ingest_single_file
import os

settings = get_settings()
router = APIRouter()

# Initialize workflow, vector store and cache once at startup
workflow = RAGWorkflow()
vector_store = VectorStore()
cache = CacheManager()                           # V2 Phase 5


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check system health and status."""
    try:
        cache_stats = cache.stats()
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            environment=settings.app_env,
            total_chunks_in_db=vector_store.count(),
            cache_enabled=cache_stats.get("enabled", False),       # V2 Phase 5
            cache_connected=cache_stats.get("connected", False)    # V2 Phase 5
        )
    except Exception as e:
        app_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="System unhealthy")


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    V2 Phase 5: Checks Redis cache before running pipeline.
    Returns grounded answer with citations.
    """
    app_logger.info(f"Received query: {request.query[:50]}...")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # V2 Phase 5: Check cache first
        cached_result = cache.get(request.query)
        if cached_result:
            cached_result["cache_hit"] = True
            return QueryResponse(**cached_result)

        # Cache miss — run full pipeline
        result = workflow.run(request.query)
        result["cache_hit"] = False

        # Store in cache for future requests
        cache.set(request.query, result)

        return QueryResponse(**result)

    except Exception as e:
        app_logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Trigger document ingestion pipeline.
    V2 Phase 5: Flushes cache on ingestion to avoid stale results.
    """
    app_logger.info(f"Ingestion triggered (reset={request.reset})")

    try:
        run_ingestion(reset=request.reset)
        total = vector_store.count()

        # V2 Phase 5: Flush cache when documents change
        flushed = cache.flush()
        if flushed > 0:
            app_logger.info(f"Flushed {flushed} cache entries after ingestion")

        return IngestResponse(
            message="Ingestion completed successfully",
            total_chunks=total
        )
    except Exception as e:
        app_logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into ChromaDB.
    V2 Phase 5: Flushes cache after upload to avoid stale results.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB"
        )

    save_path = os.path.join("data", "raw", file.filename)
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        app_logger.info(f"Saved uploaded file: {file.filename}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    result = ingest_single_file(save_path)
    total = vector_store.count()

    # V2 Phase 5: Flush cache when new document added
    cache.flush()

    return UploadResponse(
        success=result["success"],
        filename=result["filename"],
        chunks_added=result["chunks_added"],
        message=result["message"],
        total_chunks=total
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """List all documents currently ingested in ChromaDB."""
    try:
        documents = vector_store.list_documents()
        return DocumentsResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=vector_store.count()
        )
    except Exception as e:
        app_logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats():
    """
    V2 Phase 5: Get Redis cache statistics.
    Returns number of cached queries and connection status.
    """
    try:
        stats = cache.stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        app_logger.error(f"Cache stats failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache stats failed: {str(e)}"
        )


@router.delete("/cache/flush")
def flush_cache():
    """
    V2 Phase 5: Manually flush all cached query results.
    Useful when documents are updated outside normal ingestion.
    """
    try:
        deleted = cache.flush()
        app_logger.info(f"Manual cache flush — deleted {deleted} entries")
        return {"message": f"Cache flushed successfully", "deleted": deleted}
    except Exception as e:
        app_logger.error(f"Cache flush failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache flush failed: {str(e)}"
        )


@router.post("/evaluate/ragas", response_model=RagasEvaluationResponse)
def evaluate_ragas(request: RagasEvaluationRequest):
    """
    V2 Phase 4: Run RAGAS evaluation on the RAG pipeline.
    """
    app_logger.info("RAGAS evaluation requested...")

    try:
        from app.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator()
        test_cases = request.test_cases if request.test_cases else None
        results = evaluator.evaluate(test_cases=test_cases)
        return RagasEvaluationResponse(**results)

    except Exception as e:
        app_logger.error(f"RAGAS evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"RAGAS evaluation failed: {str(e)}"
        )