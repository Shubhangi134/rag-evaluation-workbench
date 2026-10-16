import re
from app.models import DocumentChunk
from pypdf import PdfReader
from pathlib import Path
def clean_text(text: str) -> str:
    """Normalize whitespace in extracted document text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into word-aligned chunks with overlapping content."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        added_length = len(word) if not current_chunk else len(word) + 1

        if current_chunk and current_length + added_length > chunk_size:
            chunks.append(" ".join(current_chunk))

            overlap_words = []
            overlap_length = 0

            for existing_word in reversed(current_chunk):
                extra_length = len(existing_word) if not overlap_words else len(existing_word) + 1
                if overlap_length + extra_length > overlap:
                    break
                overlap_words.insert(0, existing_word)
                overlap_length += extra_length

            current_chunk = overlap_words
            current_length = overlap_length

            added_length = len(word) if not current_chunk else len(word) + 1

        current_chunk.append(word)
        current_length += added_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
def create_document_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Create text chunks with document metadata."""
    if not document_id.strip():
        raise ValueError("document_id cannot be empty")

    text_chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return [
        DocumentChunk(
            document_id=document_id,
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(text_chunks)
    ]
def extract_text_from_pdf(file_path: str) -> str:
    """Extract and clean text from a PDF file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("File must have a .pdf extension")

    reader = PdfReader(str(path))
    extracted_pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            extracted_pages.append(page_text)

    return clean_text(" ".join(extracted_pages))