"""
Ingestion Script — NeuroRAG
Runs the complete ingestion pipeline:
Load PDFs → Chunk → Embed → Store in ChromaDB

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --reset
"""

import sys
import os

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.document_loader import load_documents, load_single_document
from app.ingestion.chunker import chunk_documents
from app.embedding.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.core.logger import app_logger
from app.core.exceptions import NeuroRAGException


def run_ingestion(data_dir: str = "data/raw/", reset: bool = False):
    """
    Run the complete ingestion pipeline on all PDFs in a directory.

    Args:
        data_dir: Directory containing PDF files
        reset: If True, clears existing ChromaDB before ingesting
    """
    app_logger.info("=" * 50)
    app_logger.info("Starting NeuroRAG Ingestion Pipeline")
    app_logger.info("=" * 50)

    try:
        # Step 1 — Load documents
        app_logger.info("Step 1: Loading documents...")
        pages = load_documents(data_dir)
        app_logger.info(f"Loaded {len(pages)} pages from {data_dir}")

        # Step 2 — Chunk documents
        app_logger.info("Step 2: Chunking documents...")
        chunks = chunk_documents(pages)
        app_logger.info(f"Created {len(chunks)} chunks")

        # Step 3 — Generate embeddings
        app_logger.info("Step 3: Generating embeddings...")
        embedder = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks)
        app_logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        # Step 4 — Store in ChromaDB
        app_logger.info("Step 4: Storing in ChromaDB...")
        store = VectorStore()

        if reset:
            app_logger.warning("Reset flag set — clearing existing collection")
            store.reset()

        store.add_chunks(embedded_chunks)
        app_logger.info(f"Total chunks in ChromaDB: {store.count()}")

        app_logger.info("=" * 50)
        app_logger.info("✅ Ingestion Pipeline Completed Successfully")
        app_logger.info("=" * 50)

    except NeuroRAGException as e:
        app_logger.error(f"Ingestion failed: {e.message}")
        sys.exit(1)
    except Exception as e:
        app_logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


def ingest_single_file(file_path: str) -> dict:
    """
    Ingest a single PDF file into ChromaDB.
    Handles duplicate detection — replaces existing if already ingested.

    Args:
        file_path: Full path to the PDF file

    Returns:
        Dict with 'success', 'filename', 'chunks_added', 'message' keys
    """
    filename = os.path.basename(file_path)
    app_logger.info(f"Starting single file ingestion: {filename}")

    try:
        store = VectorStore()

        # Check if document already exists — delete old chunks first
        if store.document_exists(filename):
            app_logger.warning(f"Document already exists — replacing: {filename}")
            store.delete_document(filename)

        # Load and parse
        app_logger.info(f"Loading: {filename}")
        pages = load_single_document(file_path)
        app_logger.info(f"Loaded {len(pages)} pages")

        # Chunk
        app_logger.info("Chunking...")
        chunks = chunk_documents(pages)
        app_logger.info(f"Created {len(chunks)} chunks")

        # Embed
        app_logger.info("Generating embeddings...")
        embedder = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        # Store
        app_logger.info("Storing in ChromaDB...")
        store.add_chunks(embedded_chunks)

        app_logger.info(f"✅ Successfully ingested: {filename}")

        return {
            "success": True,
            "filename": filename,
            "chunks_added": len(embedded_chunks),
            "message": f"Successfully ingested {filename} ({len(embedded_chunks)} chunks)"
        }

    except NeuroRAGException as e:
        app_logger.error(f"Ingestion failed for {filename}: {e.message}")
        return {
            "success": False,
            "filename": filename,
            "chunks_added": 0,
            "message": f"Failed: {e.message}"
        }
    except Exception as e:
        app_logger.error(f"Unexpected error ingesting {filename}: {str(e)}")
        return {
            "success": False,
            "filename": filename,
            "chunks_added": 0,
            "message": f"Unexpected error: {str(e)}"
        }


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    run_ingestion(reset=reset)