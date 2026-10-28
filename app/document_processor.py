
"""Extract and split text from PDF documents."""

import re
from pathlib import Path

from pypdf import PdfReader

from app.models import DocumentChunk


def clean_text(text: str) -> str:
    """Remove extra spaces and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap size")

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

            for previous_word in reversed(current_chunk):
                extra_length = (
                    len(previous_word)
                    if not overlap_words
                    else len(previous_word) + 1
                )

                if overlap_length + extra_length > overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += extra_length

            current_chunk = overlap_words
            current_length = overlap_length

        current_chunk.append(word)
        current_length += (
            len(word) if len(current_chunk) == 1
            else len(word) + 1
        )

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_document_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Create document chunks from extracted text."""

    if not document_id.strip():
        raise ValueError("document_id cannot be empty")

    text_chunks = chunk_text(text, chunk_size, overlap)

    return [
        DocumentChunk(
            document_id=document_id,
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(text_chunks)
    ]


def extract_pages_from_pdf(
    file_path: str,
) -> list[tuple[int, str]]:
    """Extract text from each PDF page separately.

    Returns:
        A list containing (page_number, text).
        Page numbering starts at 1.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("File must have a .pdf extension")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if text:
            pages.append((page_number, text))

    return pages


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all PDF text as one string.

    Kept for compatibility with the existing API.
    """

    pages = extract_pages_from_pdf(file_path)

    return clean_text(
        " ".join(text for _, text in pages)
    )

def create_page_chunks(
    file_path: str,
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Create PDF chunks while preserving page numbers."""

    pages = extract_pages_from_pdf(file_path)
    all_chunks = []

    for page_number, page_text in pages:
        # Split the current page into smaller text chunks.
        text_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for text in text_chunks:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=len(all_chunks),
                text=text,
                page_number=page_number,
            )

            all_chunks.append(chunk)

    return all_chunks
