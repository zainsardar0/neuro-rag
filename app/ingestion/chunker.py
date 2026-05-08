from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.logger import app_logger
from app.core.exceptions import ChunkingError


def chunk_documents(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Split extracted pages into smaller overlapping chunks.

    Args:
        pages: List of dicts with 'text', 'page', 'source' keys (from document_loader)
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks in characters

    Returns:
        List of dicts with 'text', 'page', 'source', 'chunk_id' keys

    Raises:
        ChunkingError: If chunking fails
    """

    if not pages:
        raise ChunkingError("No pages provided for chunking")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_chunks = []
        chunk_id = 0

        for page in pages:
            text = page["text"]
            splits = splitter.split_text(text)

            for split in splits:
                if not split.strip():
                    continue

                all_chunks.append({
                    "text": split.strip(),
                    "page": page["page"],
                    "source": page["source"],
                    "chunk_id": chunk_id
                })
                chunk_id += 1

        app_logger.info(f"Chunked {len(pages)} pages into {len(all_chunks)} chunks")
        return all_chunks

    except ChunkingError:
        raise
    except Exception as e:
        raise ChunkingError(f"Failed to chunk documents: {str(e)}")