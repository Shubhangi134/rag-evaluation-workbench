
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of retrieved top-k results that are relevant."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant results found in the top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    for rank, document_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Calculate retrieval metrics for one query."""

    return {
        "precision_at_k": precision_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "recall_at_k": recall_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "reciprocal_rank_at_k": reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, k
        ),
    }
