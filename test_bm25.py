
"""Tests for BM25 document retrieval."""

from app.models import DocumentChunk
from app.retriever import search_chunks_bm25


def make_chunks():
    """Create sample chunks for BM25 testing."""
    return [
        DocumentChunk(
            document_id="sample.pdf",
            chunk_index=0,
            text="Cats sleep on sofas.",
            page_number=1,
        ),
        DocumentChunk(
            document_id="sample.pdf",
            chunk_index=1,
            text="Automated driving systems use sensors for safety.",
            page_number=2,
        ),
        DocumentChunk(
            document_id="sample.pdf",
            chunk_index=2,
            text="Driving safety requires careful planning.",
            page_number=3,
        ),
    ]


def test_bm25_ranks_relevant_chunks():
    chunks = make_chunks()

    results = search_chunks_bm25(
        "automated driving safety",
        chunks,
        top_k=3,
    )

    assert [chunk.chunk_index for chunk in results] == [1, 2]


def test_bm25_empty_query():
    assert search_chunks_bm25("", make_chunks()) == []


def test_bm25_empty_documents():
    assert search_chunks_bm25("driving safety", []) == []


def test_bm25_invalid_top_k():
    assert search_chunks_bm25(
        "driving safety",
        make_chunks(),
        top_k=0,
    ) == []


def test_bm25_respects_top_k():
    results = search_chunks_bm25(
        "driving safety",
        make_chunks(),
        top_k=1,
    )

    assert len(results) == 1


def test_bm25_no_matching_terms():
    results = search_chunks_bm25(
        "quantum physics",
        make_chunks(),
    )

    assert results == []
