
"""Manage temporary user sessions."""

import secrets
import time
from threading import Lock

from app.config import SESSION_TIMEOUT_SECONDS
from app.models import SessionData


# Store sessions in server memory
sessions: dict[str, SessionData] = {}

# Protect the dictionary when requests run concurrently
sessions_lock = Lock()


def create_session() -> SessionData:
    """Create a new anonymous session."""

    with sessions_lock:
        _cleanup_expired_sessions()

        session_id = secrets.token_urlsafe(32)
        session = SessionData(session_id=session_id)

        sessions[session_id] = session

        return session


def get_session(session_id: str) -> SessionData | None:
    """Find an active session and update its activity time."""

    with sessions_lock:
        _cleanup_expired_sessions()

        session = sessions.get(session_id)

        if session is not None:
            session.last_activity = time.time()

        return session


def delete_session(session_id: str) -> bool:
    """Delete a session and its stored data."""

    with sessions_lock:
        removed = sessions.pop(session_id, None)

        return removed is not None


def _cleanup_expired_sessions() -> int:
    """Remove expired sessions while the lock is held."""

    current_time = time.time()

    expired_ids = [
        session_id
        for session_id, session in sessions.items()
        if current_time - session.last_activity >= SESSION_TIMEOUT_SECONDS
    ]

    for session_id in expired_ids:
        del sessions[session_id]

    return len(expired_ids)


def cleanup_expired_sessions() -> int:
    """Public function for removing expired sessions."""

    with sessions_lock:
        return _cleanup_expired_sessions()
