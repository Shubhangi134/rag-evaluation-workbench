def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of the top-k retrieved chunks that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for chunk_id in top_k if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k)
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of all relevant chunks retrieved in the top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for chunk_id in top_k if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)

def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Evaluate retrieval using precision and recall at k."""
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
    }