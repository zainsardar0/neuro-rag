from pathlib import Path
from app.core.logger import app_logger
from app.core.exceptions import DocumentLoadError
from app.ingestion.pdf_parser import parse_pdf


def load_documents(directory: str) -> list[dict]:
    """
    Load all PDF documents from a directory.
    
    Args:
        directory: Path to folder containing PDF files
        
    Returns:
        List of dicts with 'text', 'page', and 'source' keys
        
    Raises:
        DocumentLoadError: If directory doesn't exist or no PDFs found
    """
    dir_path = Path(directory)

    # Check directory exists
    if not dir_path.exists():
        raise DocumentLoadError(f"Directory not found: {directory}")

    # Find all PDFs
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        raise DocumentLoadError(f"No PDF files found in: {directory}")

    app_logger.info(f"Found {len(pdf_files)} PDF(s) in {directory}")

    all_pages = []

    for pdf_file in pdf_files:
        try:
            pages = parse_pdf(str(pdf_file))
            all_pages.extend(pages)
        except DocumentLoadError as e:
            app_logger.error(f"Skipping {pdf_file.name}: {e.message}")
            continue

    if not all_pages:
        raise DocumentLoadError("No content extracted from any PDF")

    app_logger.info(f"Total pages extracted: {len(all_pages)}")
    return all_pages


def load_single_document(file_path: str) -> list[dict]:
    """
    Load a single PDF document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of dicts with 'text', 'page', and 'source' keys
    """
    app_logger.info(f"Loading single document: {file_path}")
    return parse_pdf(file_path)