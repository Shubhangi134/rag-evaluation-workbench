def compare_retrieval_strategies(
    query: str,
    chunks,
    relevant_ids: set[str],
    k: int = 3,
) -> dict[str, dict[str, float]]:
    """Compare keyword, TF-IDF, and semantic retrieval on the same query."""
    from app.evaluator import evaluate_retrieval
    from app.retriever import (
        search_chunks,
        search_chunks_tfidf,
        search_chunks_semantic,
    )

    strategies = {
        "keyword": search_chunks,
        "tfidf": search_chunks_tfidf,
        "semantic": search_chunks_semantic,
    }

    results = {}

    for name, retrieve in strategies.items():
        retrieved_chunks = retrieve(query, chunks, top_k=k)
        retrieved_ids = [
            f"{chunk.document_id}:{chunk.chunk_index}"
            for chunk in retrieved_chunks
        ]
        results[name] = evaluate_retrieval(
            retrieved_ids,
            relevant_ids,
            k,
        )

    return results