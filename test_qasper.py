
"""Automated tests for QASPER answer evaluation."""

import json
from unittest.mock import MagicMock
from evaluate_qasper import summarize_qasper_results
from app.models import DocumentChunk
from evaluate_qasper import (
    generate_qasper_answer,
    evaluate_faithfulness,
    answer_token_f1,
    best_reference_f1,
)


def sample_chunks():
    """Create evidence for our tests."""

    return [
        DocumentChunk(
            document_id="test-paper",
            chunk_index=0,
            text="The system uses sensing, planning, and control.",
        )
    ]


def test_qasper_generator_uses_evidence_prompt(monkeypatch):
    """Generator should request a grounded plain-text answer."""

    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {
            "content": "Sensing, planning, and control."
        }
    }

    monkeypatch.setattr(
        "evaluate_qasper.ollama.Client",
        lambda: fake_client,
    )

    answer = generate_qasper_answer(
        "What functions does the system use?",
        sample_chunks(),
    )

    assert answer == "Sensing, planning, and control."

    arguments = fake_client.chat.call_args.kwargs
    system_prompt = arguments["messages"][0]["content"]
    user_prompt = arguments["messages"][1]["content"]

    assert "evidence" in system_prompt.lower()
    assert "plain text" in system_prompt.lower()
    assert "sensing, planning, and control" in user_prompt
    assert "format" not in arguments


def test_qasper_generator_without_evidence(monkeypatch):
    """No evidence means no factual answer or LLM call."""

    def unexpected_client():
        raise AssertionError("Ollama should not be called.")

    monkeypatch.setattr(
        "evaluate_qasper.ollama.Client",
        unexpected_client,
    )

    answer = generate_qasper_answer(
        "Who won the World Cup?",
        [],
    )

    assert answer == "Insufficient evidence."


def test_faithfulness_evaluator_checks_claims(monkeypatch):
    """Evaluator should request JSON and calculate support ratio."""

    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {
            "content": json.dumps({
                "claims": [
                    {
                        "claim": "The system uses sensing.",
                        "supported": True,
                    },
                    {
                        "claim": "The system uses teleportation.",
                        "supported": False,
                    },
                ]
            })
        }
    }

    monkeypatch.setattr(
        "evaluate_qasper.ollama.Client",
        lambda: fake_client,
    )

    result = evaluate_faithfulness(
        "The system uses sensing and teleportation.",
        sample_chunks(),
    )

    assert result["faithfulness_score"] == 0.5
    assert len(result["claims"]) == 2

    arguments = fake_client.chat.call_args.kwargs

    assert arguments["format"] == "json"
    assert "supported" in (
        arguments["messages"][0]["content"].lower()
    )


def test_reference_token_f1():
    """Exact and partial matches should score correctly."""

    assert answer_token_f1(
        "multilingual NMT",
        "multilingual NMT",
    ) == 1.0

    assert answer_token_f1(
        "multilingual NMT BIBREF19",
        "multilingual NMT BIBREF20",
    ) == 2 / 3


def test_best_reference_f1():
    """Use the closest available human reference."""

    result = best_reference_f1(
        "multilingual NMT",
        [
            "BIBREF19 BIBREF20",
            "multilingual NMT",
        ],
    )

    assert result == 1.0


def test_faithfulness_abstention(monkeypatch):
    """Abstention should not be treated as a hallucination."""

    def unexpected_client():
        raise AssertionError("Ollama should not be called.")

    monkeypatch.setattr(
        "evaluate_qasper.ollama.Client",
        unexpected_client,
    )

    result = evaluate_faithfulness(
        "Insufficient evidence.",
        sample_chunks(),
    )

    assert result["faithfulness_score"] is None
    assert result["claims"] == []

def test_qasper_summary_with_abstention():
    """Abstentions must be counted separately from faithfulness."""

    results = [
        {
            "method": "BM25",
            "generated_answer": "Multilingual NMT.",
            "reference_token_f1": 0.8,
            "faithfulness_score": 1.0,
        },
        {
            "method": "BM25",
            "generated_answer": "Insufficient evidence.",
            "reference_token_f1": 0.0,
            "faithfulness_score": None,
        },
        {
            "method": "TF-IDF",
            "generated_answer": "Another answer.",
            "reference_token_f1": 0.5,
            "faithfulness_score": 0.5,
        },
    ]

    summary = summarize_qasper_results(results, "BM25")

    assert summary["question_count"] == 2
    assert summary["mean_token_f1"] == 0.4
    assert summary["mean_faithfulness"] == 1.0
    assert summary["abstention_count"] == 1
    assert summary["abstention_rate"] == 0.5


def test_qasper_summary_all_abstentions():
    """If every answer abstains, mean faithfulness is unavailable."""

    results = [
        {
            "method": "Semantic",
            "generated_answer": "Insufficient evidence.",
            "reference_token_f1": 0.0,
            "faithfulness_score": None,
        },
        {
            "method": "Semantic",
            "generated_answer": "Insufficient evidence.",
            "reference_token_f1": 0.0,
            "faithfulness_score": None,
        },
    ]

    summary = summarize_qasper_results(results, "Semantic")

    assert summary["question_count"] == 2
    assert summary["mean_token_f1"] == 0.0
    assert summary["mean_faithfulness"] is None
    assert summary["abstention_count"] == 2
    assert summary["abstention_rate"] == 1.0

def test_qasper_comparison_workflow(monkeypatch, tmp_path):
    """Test all three methods, abstentions, and saved results."""

    import json
    import evaluate_qasper as qasper

    # Create a tiny QASPER dataset for this test.
     # Create three different papers for the evaluation.
    paper_data = {}
    for index in range(1, 4):
        paper_id = f"test-paper-{index}"

        paper_data[paper_id] = {
            "title": f"Test Research Paper {index}",
            "abstract": "The system uses sensing and planning.",
            "full_text": [],
            "qas": [
                {
                    "question_id": f"question-{index}",
                    "question": "What does the system use?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "yes_no": None,
                                "free_form_answer": "Sensing and planning.",
                                "extractive_spans": [],
                            }
                        }
                    ],
                }
            ],
        }

    data_path = tmp_path / "qasper-test.json"
    data_path.write_text(
        json.dumps(paper_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(qasper, "DATA_PATH", data_path)
    monkeypatch.setattr(qasper, "MAX_QUESTIONS", 3)

    # Redirect the generated JSON into pytest's temporary folder.
    monkeypatch.setattr(
        qasper,
        "__file__",
        str(tmp_path / "evaluate_qasper.py"),
    )

    retrieval_calls = []

    def make_retriever(method):
        def retrieve(query, chunks, top_k):
            retrieval_calls.append((method, query, top_k))

            return [
                qasper.DocumentChunk(
                    document_id="test-paper",
                    chunk_index={
                        "BM25": 1,
                        "TF-IDF": 2,
                        "Semantic": 3,
                    }[method],
                    text=f"Evidence from {method}",
                )
            ]

        return retrieve

    for name, attribute in [
        ("BM25", "search_chunks_bm25"),
        ("TF-IDF", "search_chunks_tfidf"),
        ("Semantic", "search_chunks_semantic"),
    ]:
        monkeypatch.setattr(
            qasper,
            attribute,
            make_retriever(name),
        )

    def fake_generate(question, chunks):
        if chunks[0].chunk_index == 3:
            return "Insufficient evidence."

        return "Sensing and planning."

    def fake_faithfulness(answer, chunks):
        if answer == "Insufficient evidence.":
            return {
                "faithfulness_score": None,
                "claims": [],
            }

        return {
            "faithfulness_score": 1.0,
            "claims": [
                {
                    "claim": "Sensing and planning.",
                    "supported": True,
                }
            ],
        }

    monkeypatch.setattr(
        qasper,
        "generate_qasper_answer",
        fake_generate,
    )
    monkeypatch.setattr(
        qasper,
        "evaluate_faithfulness",
        fake_faithfulness,
    )

    # Run the complete comparison with mocked dependencies.
    qasper.compare_qasper_methods()

    # All three methods must receive the same question and top_k.
    assert len(retrieval_calls) == 9
    assert {call[0] for call in retrieval_calls} == {
        "BM25",
        "TF-IDF",
        "Semantic",
    }
    assert {call[1] for call in retrieval_calls} == {
        "What does the system use?",
    }
    assert all(call[2] == 3 for call in retrieval_calls)

    # Verify that the JSON file was created.
    output_path = (
        tmp_path / "results" / "qasper_comparison.json"
    )
    assert output_path.exists()

    results = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert len(results) == 9

# Verify that the questions come from three different papers.
    assert len({result["paper_id"] for result in results}) == 3

# Every paper must be evaluated by every retrieval method.
    for paper_id in ("test-paper-1", "test-paper-2", "test-paper-3"):
        methods = {
            result["method"]
            for result in results
            if result["paper_id"] == paper_id
        }
    assert methods == {"BM25", "TF-IDF", "Semantic"}

    semantic_summary = qasper.summarize_qasper_results(
        results,
        "Semantic",
    )

    assert semantic_summary["abstention_count"] == 3
    assert semantic_summary["abstention_rate"] == 1.0
    assert semantic_summary["mean_faithfulness"] is None

    for method in ("BM25", "TF-IDF"):
        summary = qasper.summarize_qasper_results(
            results,
            method,
        )
        assert summary["abstention_count"] == 0
        assert summary["mean_faithfulness"] == 1.0

def test_selected_qasper_method(monkeypatch, tmp_path):
    """Test .env-selected evaluation without calling Ollama."""

    import json
    import evaluate_qasper as qasper

    paper_data = {
        "test-paper": {
            "title": "Test Paper",
            "abstract": "The system uses sensing and planning.",
            "full_text": [],
            "qas": [
                {
                    "question_id": "q1",
                    "question": "What does the system use?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "yes_no": None,
                                "free_form_answer": "Sensing and planning.",
                                "extractive_spans": [],
                            }
                        }
                    ],
                }
            ],
        }
    }

    data_path = tmp_path / "qasper-test.json"
    data_path.write_text(
        json.dumps(paper_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(qasper, "DATA_PATH", data_path)
    monkeypatch.setattr(qasper, "MAX_QUESTIONS", 1)
    monkeypatch.setattr(
        qasper,
        "__file__",
        str(tmp_path / "evaluate_qasper.py"),
    )

    monkeypatch.setenv("QASPER_RETRIEVER", "bm25")
    monkeypatch.setenv("QASPER_TOP_K", "3")
    monkeypatch.setenv("QASPER_MAX_QUESTIONS", "1")

    retrieval_calls = []

    def fake_retrieve(query, chunks, top_k):
        retrieval_calls.append((query, top_k))
        return chunks[:top_k]

    monkeypatch.setattr(
        qasper,
        "search_chunks_bm25",
        fake_retrieve,
    )

    monkeypatch.setattr(
        qasper,
        "generate_qasper_answer",
        lambda question, chunks: "Sensing and planning.",
    )

    monkeypatch.setattr(
        qasper,
        "evaluate_faithfulness",
        lambda answer, chunks: {
            "faithfulness_score": 1.0,
            "claims": [
                {"claim": answer, "supported": True}
            ],
        },
    )

    qasper.evaluate_selected_qasper_method()

    assert retrieval_calls == [
        ("What does the system use?", 3)
    ]
    output_path = (
    tmp_path / "results" / "qasper_bm25_q1_results.json"
    )
    assert output_path.exists()

    results = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert len(results) == 1
    assert results[0]["method"] == "BM25"
    assert results[0]["top_k"] == 3
    assert results[0]["generated_answer"] == "Sensing and planning."
    assert results[0]["abstained"] is False
