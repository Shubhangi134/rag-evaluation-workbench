
"""Tests for follow-up questions and conversation memory."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatMessage, DocumentChunk
from app.session_manager import get_session, delete_session
from unittest.mock import MagicMock

from app.generator import rewrite_question

client = TestClient(app)


@pytest.fixture
def test_session():
    """Create a session with one sample document chunk."""

    response = client.post("/sessions")
    assert response.status_code == 200

    session_id = response.json()["session_id"]
    session = get_session(session_id)

    session.documents["sample.pdf"] = [
        DocumentChunk(
            document_id="sample.pdf",
            chunk_index=0,
            text="An ADS performs the dynamic driving task.",
            page_number=1,
        )
    ]

    yield session_id, session

    delete_session(session_id)


def test_follow_up_uses_rewritten_query(
    test_session, monkeypatch
):
    """Retrieval should use the rewritten follow-up."""

    session_id, session = test_session

    session.messages.append(
        ChatMessage(
            role="user",
            content="What is an Automated Driving System?",
        )
    )
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="It performs the dynamic driving task.",
        )
    )

    captured = {}

    def fake_rewrite(question, conversation_history):
        captured["history"] = conversation_history
        return "What are the functions of an Automated Driving System?"

    def fake_retrieve(query, chunks, method, top_k):
        captured["retrieval_query"] = query
        return chunks[:top_k]

    def fake_generate(question, chunks, conversation_history):
        captured["generation_question"] = question
        return "It performs sensing, planning, and control."

    monkeypatch.setattr(
        "app.main.rewrite_question", fake_rewrite
    )
    monkeypatch.setattr(
        "app.main.retrieve_documents", fake_retrieve
    )
    monkeypatch.setattr(
        "app.main.generate_answer", fake_generate
    )

    response = client.get(
        "/ask",
        params={"query": "What are its functions?"},
        headers={"x-session-id": session_id},
    )

    assert response.status_code == 200

    assert captured["retrieval_query"] == (
        "What are the functions of an Automated Driving System?"
    )
    assert captured["generation_question"] == (
        "What are its functions?"
    )
    assert captured["history"][0]["role"] == "user"


def test_previous_question_skips_retrieval(
    test_session, monkeypatch
):
    """A previous-question request needs no PDF search."""

    session_id, session = test_session

    session.messages.append(
        ChatMessage(
            role="user",
            content="What are its main functions?",
        )
    )
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="Sensing, planning, and control.",
        )
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError(
            "PDF retrieval or LLM should not be called."
        )

    monkeypatch.setattr(
        "app.main.rewrite_question", unexpected_call
    )
    monkeypatch.setattr(
        "app.main.retrieve_documents", unexpected_call
    )
    monkeypatch.setattr(
        "app.main.generate_answer", unexpected_call
    )

    response = client.get(
        "/ask",
        params={"query": "What was my previous question?"},
        headers={"x-session-id": session_id},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        'Your previous question was: '
        '"What are its main functions?"'
    )
    assert data["retrieved_chunks"] == []


def test_no_previous_question(
    test_session, monkeypatch
):
    """The chatbot should handle empty conversation history."""

    session_id, _ = test_session

    def unexpected_call(*args, **kwargs):
        raise AssertionError("No model call is needed.")

    monkeypatch.setattr(
        "app.main.rewrite_question", unexpected_call
    )
    monkeypatch.setattr(
        "app.main.retrieve_documents", unexpected_call
    )

    response = client.get(
        "/ask",
        params={"query": "What was my previous question?"},
        headers={"x-session-id": session_id},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "You haven't asked any previous questions yet."
    )


def test_sessions_have_separate_conversations(
    test_session, monkeypatch
):
    """One session must not see another session's history."""

    first_id, first_session = test_session

    first_session.messages.append(
        ChatMessage(
            role="user",
            content="What is semantic retrieval?",
        )
    )

    second_response = client.post("/sessions")
    second_id = second_response.json()["session_id"]

    second_session = get_session(second_id)
    second_session.documents["sample.pdf"] = (
        first_session.documents["sample.pdf"]
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("No model call is needed.")

    monkeypatch.setattr(
        "app.main.rewrite_question", unexpected_call
    )
    monkeypatch.setattr(
        "app.main.retrieve_documents", unexpected_call
    )

    try:
        response = client.get(
            "/ask",
            params={
                "query": "What was my previous question?"
            },
            headers={"x-session-id": second_id},
        )

        assert response.status_code == 200
        assert response.json()["answer"] == (
            "You haven't asked any previous questions yet."
        )

    finally:
        delete_session(second_id)

def test_rewrite_preserves_new_topic(monkeypatch):
    """A new topic should not inherit the previous topic."""

    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {
            "content": "What is semantic retrieval?"
        }
    }

    monkeypatch.setattr(
        "app.generator.ollama.Client",
        lambda: fake_client,
    )

    history = [
        {
            "role": "user",
            "content": "What is an Automated Driving System?",
        },
        {
            "role": "assistant",
            "content": "It performs the dynamic driving task.",
        },
    ]

    result = rewrite_question(
        "What is semantic retrieval?",
        history,
    )

    assert result == "What is semantic retrieval?"
    assert fake_client.chat.call_count == 1


def test_rewrite_falls_back_when_ollama_fails(monkeypatch):
    """The original question is used if rewriting fails."""

    def failing_client():
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        "app.generator.ollama.Client",
        failing_client,
    )

    history = [
        {
            "role": "user",
            "content": "What is an Automated Driving System?",
        }
    ]

    result = rewrite_question(
        "What are its main functions?",
        history,
    )

    assert result == "What are its main functions?"
