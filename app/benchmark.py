
def compare_retrieval_strategies(
    query: str,
    chunks,
    relevant_ids: set[str],
    k: int = 3,
) -> dict[str, dict[str, float]]:
    """Compare BM25, TF-IDF, and semantic retrieval on one query."""

    from app.evaluator import evaluate_retrieval
    from app.retriever import (
        search_chunks_bm25,
        search_chunks_tfidf,
        search_chunks_semantic,
    )

    # Run each method on the same document chunks.
    strategies = {
        "bm25": search_chunks_bm25,
        "tfidf": search_chunks_tfidf,
        "semantic": search_chunks_semantic,
    }

    results = {}

    for name, retrieve in strategies.items():
        # Retrieve the top-k most relevant chunks.
        retrieved_chunks = retrieve(
            query,
            chunks,
            top_k=k,
        )

        # Convert each chunk into its unique ID.
        retrieved_ids = [
            f"{chunk.document_id}:{chunk.chunk_index}"
            for chunk in retrieved_chunks
        ]

        # Evaluate retrieval against known relevant chunks.
        results[name] = evaluate_retrieval(
            retrieved_ids,
            relevant_ids,
            k,
        )

    return results
