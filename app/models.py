from dataclasses import dataclass


@dataclass
class DocumentChunk:
    document_id: str
    chunk_index: int
    text: str
    def __post_init__(self):
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")        