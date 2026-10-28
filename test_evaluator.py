from app.evaluator import precision_at_k, recall_at_k, evaluate_retrieval


def test_precision_at_k():
    retrieved = ["chunk1", "chunk2", "chunk3"]
    relevant = {"chunk1", "chunk3"}

    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3


def test_recall_at_k():
    retrieved = ["chunk1", "chunk2", "chunk3"]
    relevant = {"chunk1", "chunk3", "chunk4"}

    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3


def test_evaluate_retrieval():
    retrieved = ["chunk1", "chunk2", "chunk3"]
    relevant = {"chunk1", "chunk3", "chunk4"}

    
    assert evaluate_retrieval(retrieved, relevant, k=3) == {
        "precision_at_k": 2 / 3,
        "recall_at_k": 2 / 3,
        "reciprocal_rank_at_k": 1.0,
    }


def test_compare_retrieval_strategies_returns_all_metrics():
    from app.benchmark import compare_retrieval_strategies
    from app.models import DocumentChunk

    chunks = [
        DocumentChunk("doc1", 0, "Vehicle safety regulations"),
        DocumentChunk("doc1", 1, "Cooking recipes and ingredients"),
        DocumentChunk("doc1", 2, "Automobile braking safety requirements"),
    ]

    relevant_ids = {"doc1:0", "doc1:2"}

    results = compare_retrieval_strategies(
        query="vehicle safety",
        chunks=chunks,
        relevant_ids=relevant_ids,
        k=2,
    )

    assert set(results.keys()) == {"bm25", "tfidf", "semantic"}
    for metrics in results.values():
        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics


def test_reciprocal_rank_first_position():
    from app.evaluator import reciprocal_rank_at_k

    result = reciprocal_rank_at_k(
        ["doc1", "doc2", "doc3"],
        {"doc1"},
        k=3,
    )

    assert result == 1.0


def test_reciprocal_rank_second_position():
    from app.evaluator import reciprocal_rank_at_k

    result = reciprocal_rank_at_k(
        ["doc1", "doc2", "doc3"],
        {"doc2"},
        k=3,
    )

    assert result == 0.5


def test_reciprocal_rank_third_position():
    from app.evaluator import reciprocal_rank_at_k

    result = reciprocal_rank_at_k(
        ["doc1", "doc2", "doc3"],
        {"doc3"},
        k=3,
    )

    assert result == 1 / 3


def test_reciprocal_rank_no_relevant_result():
    from app.evaluator import reciprocal_rank_at_k

    result = reciprocal_rank_at_k(
        ["doc1", "doc2", "doc3"],
        {"doc4"},
        k=3,
    )

    assert result == 0.0
