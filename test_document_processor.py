from app.document_processor import clean_text
from app.document_processor import chunk_text
from app.models import DocumentChunk
from app.document_processor import create_document_chunks
from app.document_processor import extract_text_from_pdf
from app.retriever import search_chunks
import pytest

def test_clean_text_normalizes_whitespace():
    text = "  Hello   world!\n\nThis is a test.  "

    result = clean_text(text)

    assert result == "Hello world! This is a test."

def test_clean_text_handles_empty_input():
    assert clean_text("") == ""

def test_clean_text_handles_whitespace_only():
    assert clean_text("   \n\t  ") == ""

def test_chunk_text_returns_single_chunk_for_short_text():
    text = "This is a short document."

    result = chunk_text(text, chunk_size=100, overlap=10)

    assert result == [text]

def test_chunk_text_respects_size_and_overlap():
    text = "one two three four five six seven eight"

    chunks = chunk_text(text, chunk_size=20, overlap=8)

    assert all(len(chunk) <= 20 for chunk in chunks)
    assert chunks[0] == "one two three four"
    assert chunks[1] == "four five six seven"
    assert chunks[0].split()[-1] == chunks[1].split()[0]
def test_chunk_text_rejects_invalid_settings():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=10, overlap=10)

def test_chunk_text_handles_empty_input_and_long_words():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("extraordinary", chunk_size=5, overlap=1) == [
        "extraordinary"
    ]

def test_document_chunk_stores_metadata():
    chunk = DocumentChunk(
        document_id="doc-001",
        chunk_index=0,
        text="This is a sample chunk.",
    )

    assert chunk.document_id == "doc-001"
    assert chunk.chunk_index == 0
    assert chunk.text == "This is a sample chunk."
def test_document_chunk_rejects_invalid_metadata():
    import pytest

    with pytest.raises(ValueError):
        DocumentChunk(
            document_id="",
            chunk_index=0,
            text="Sample text",
        )

    with pytest.raises(ValueError):
        DocumentChunk(
            document_id="doc-001",
            chunk_index=-1,
            text="Sample text",
        )
def test_create_document_chunks_assigns_metadata():
    chunks = create_document_chunks(
        text="First sentence. Second sentence.",
        document_id="doc-001",
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) == 2
    assert chunks[0].document_id == "doc-001"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(chunk.text for chunk in chunks)

def test_create_document_chunks_handles_empty_text():
    chunks = create_document_chunks(
        text="",
        document_id="doc-001",
    )

    assert chunks == []

from pypdf import PdfWriter
from pathlib import Path


def test_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    with open(pdf_path, "wb") as pdf_file:
        writer.write(pdf_file)

    result = extract_text_from_pdf(str(pdf_path))

    assert result == ""
def test_extract_text_from_pdf_with_text(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample_with_text.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "RAG evaluation sample document")
    pdf.save()

    result = extract_text_from_pdf(str(pdf_path))

    assert "RAG evaluation sample document" in result
def test_extract_text_from_pdf_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(str(missing_file))
def test_extract_text_from_pdf_rejects_non_pdf_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="File must have a .pdf extension"):
        extract_text_from_pdf(str(text_file))

def test_chunk_text_respects_size_and_overlap():
    text = "one two three four five six"

    result = chunk_text(text, chunk_size=13, overlap=3)

    assert result == [
        "one two three",
        "four five six",
    ]
def test_search_chunks_returns_best_keyword_match():
    from app.models import DocumentChunk
    from app.retriever import search_chunks

    chunks = [
        DocumentChunk("doc1", 0, "Vehicle safety regulations"),
        DocumentChunk("doc1", 1, "Cooking recipes and ingredients"),
        DocumentChunk("doc1", 2, "Vehicle braking safety requirements"),
    ]

    results = search_chunks("vehicle safety", chunks, top_k=2)

    assert len(results) == 2
    assert results[0].text == "Vehicle safety regulations"
    assert results[1].text == "Vehicle braking safety requirements"

def test_search_chunks_tfidf_returns_relevant_chunks():
    from app.models import DocumentChunk
    from app.retriever import search_chunks_tfidf

    chunks = [
        DocumentChunk("doc1", 0, "Vehicle safety regulations"),
        DocumentChunk("doc1", 1, "Cooking recipes and ingredients"),
        DocumentChunk("doc1", 2, "Vehicle braking safety requirements"),
    ]

    results = search_chunks_tfidf("vehicle safety", chunks, top_k=2)

    assert len(results) == 2
    assert results[0].text == "Vehicle safety regulations"
    assert results[1].text == "Vehicle braking safety requirements"