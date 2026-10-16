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
    }