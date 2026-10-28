
"""Tests for temporary user sessions."""

import time

from app.config import SESSION_TIMEOUT_SECONDS
from app.models import ChatMessage, DocumentChunk
from app.session_manager import (
    create_session,
    get_session,
    delete_session,
    cleanup_expired_sessions,
)


def test_create_session():
    """A new session starts empty."""

    session = create_session()

    assert session.session_id
    assert session.documents == {}
    assert session.messages == []

    delete_session(session.session_id)


def test_sessions_are_separate():
    """Two users must have separate document data."""

    user_a = create_session()
    user_b = create_session()

    chunk = DocumentChunk(
        document_id="research.pdf",
        chunk_index=0,
        text="Example research content",
    )

    user_a.documents["research.pdf"] = [chunk]

    assert "research.pdf" in user_a.documents
    assert user_b.documents == {}

    delete_session(user_a.session_id)
    delete_session(user_b.session_id)


def test_conversation_history_is_separate():
    """One user's conversation must not appear in another."""

    user_a = create_session()
    user_b = create_session()

    user_a.messages.append(
        ChatMessage(
            role="user",
            content="What is semantic retrieval?",
        )
    )

    assert len(user_a.messages) == 1
    assert user_b.messages == []

    delete_session(user_a.session_id)
    delete_session(user_b.session_id)


def test_delete_session():
    """Deleted sessions must no longer be accessible."""

    session = create_session()
    session_id = session.session_id

    assert delete_session(session_id) is True
    assert get_session(session_id) is None


def test_expired_session():
    """Inactive sessions should expire."""

    session = create_session()
    session_id = session.session_id

    # Simulate inactivity without waiting 30 minutes.
    session.last_activity = (
        time.time() - SESSION_TIMEOUT_SECONDS - 1
    )

    removed_count = cleanup_expired_sessions()

    assert removed_count >= 1
    assert get_session(session_id) is None


def test_session_activity_updates():
    """Using a session should refresh its activity time."""

    session = create_session()
    old_time = session.last_activity

    time.sleep(0.01)

    active_session = get_session(session.session_id)

    assert active_session is not None
    assert active_session.last_activity >= old_time

    delete_session(session.session_id)
