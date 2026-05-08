from pypdf import PdfReader
from pathlib import Path
from app.core.logger import app_logger
from app.core.exceptions import DocumentLoadError


def parse_pdf(file_path: str) -> list[dict]:
    """
    Parse a PDF file and extract text with metadata.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of dicts with 'text', 'page', and 'source' keys
        
    Raises:
        DocumentLoadError: If file doesn't exist or can't be parsed
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        raise DocumentLoadError(f"File not found: {file_path}")

    # Check it's actually a PDF
    if path.suffix.lower() != ".pdf":
        raise DocumentLoadError(f"File is not a PDF: {file_path}")

    pages = []

    try:
        reader = PdfReader(str(path))
        app_logger.info(f"Loading PDF: {path.name} ({len(reader.pages)} pages)")

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            # Skip empty pages
            if not text or not text.strip():
                app_logger.warning(f"Empty page skipped: page {page_num} in {path.name}")
                continue

            pages.append({
                "text": text.strip(),
                "page": page_num,
                "source": path.name
            })

        app_logger.info(f"Successfully parsed {len(pages)} pages from {path.name}")
        return pages

    except DocumentLoadError:
        raise
    except Exception as e:
        raise DocumentLoadError(f"Failed to parse PDF {path.name}: {str(e)}")