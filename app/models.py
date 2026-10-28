
from dataclasses import dataclass, field
from time import time



@dataclass
class DocumentChunk:
    """Represents a text chunk from a PDF document."""

    document_id: str
    chunk_index: int
    text: str
    page_number: int | None = None

    def __post_init__(self):
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")

        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be at least 1")


@dataclass
class ChatMessage:
    """Represents one message in a conversation."""

    role: str
    content: str

    def __post_init__(self):
        if self.role not in ("user", "assistant"):
            raise ValueError("Invalid chat role")

        if not self.content.strip():
            raise ValueError("Message cannot be empty")


@dataclass
class SessionData:
    """Stores temporary PDF data and chat history for one user."""

    session_id: str

    # PDF filename -> extracted chunks
    documents: dict[str, list[DocumentChunk]] = field(
        default_factory=dict
    )

    # Previous questions and answers
    messages: list[ChatMessage] = field(
        default_factory=list
    )

    # Time of last session activity
    last_activity: float = field(default_factory=time)
