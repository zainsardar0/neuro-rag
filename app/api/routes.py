from fastapi import APIRouter, HTTPException, UploadFile, File
from app.api.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    UploadResponse, DocumentsResponse,
    HealthResponse,
    RagasEvaluationRequest, RagasEvaluationResponse  # V2 Phase 4
)
from app.llm.workflow import RAGWorkflow
from app.retrieval.vector_store import VectorStore
from app.core.config import get_settings
from app.core.logger import app_logger
from scripts.ingest import run_ingestion, ingest_single_file
import shutil
import os

settings = get_settings()
router = APIRouter()

# Initialize workflow and vector store once at startup
workflow = RAGWorkflow()
vector_store = VectorStore()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check system health and status."""
    try:
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            environment=settings.app_env,
            total_chunks_in_db=vector_store.count()
        )
    except Exception as e:
        app_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="System unhealthy")


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    Returns grounded answer with citations.
    """
    app_logger.info(f"Received query: {request.query[:50]}...")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = workflow.run(request.query)
        return QueryResponse(**result)
    except Exception as e:
        app_logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Trigger document ingestion pipeline.
    Loads all PDFs from data/raw/ and stores in ChromaDB.
    """
    app_logger.info(f"Ingestion triggered (reset={request.reset})")

    try:
        run_ingestion(reset=request.reset)
        total = vector_store.count()
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
    Handles duplicates — replaces existing if already ingested.
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

    return UploadResponse(
        success=result["success"],
        filename=result["filename"],
        chunks_added=result["chunks_added"],
        message=result["message"],
        total_chunks=total
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """
    List all documents currently ingested in ChromaDB.
    """
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


@router.post("/evaluate/ragas", response_model=RagasEvaluationResponse)
def evaluate_ragas(request: RagasEvaluationRequest):
    """
    V2 Phase 4: Run RAGAS evaluation on the RAG pipeline.
    Uses Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference.
    Returns per-metric scores and overall RAGAS score.
    """
    app_logger.info("RAGAS evaluation requested...")

    try:
        # Lazy import to avoid loading RAGAS on every server startup
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