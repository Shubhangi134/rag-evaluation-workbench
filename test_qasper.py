
"""Automated tests for QASPER answer evaluation."""

import json
from unittest.mock import MagicMock

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
