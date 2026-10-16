import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent.parent / "rag_workbench.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (document_id, chunk_index)
            )
            """
        )
def save_chunks(document_id: str, chunks):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            (document_id,),
        )

        connection.executemany(
            """
            INSERT INTO document_chunks (document_id, chunk_index, text)
            VALUES (?, ?, ?)
            """,
            [
                (document_id, chunk.chunk_index, chunk.text)
                for chunk in chunks
            ],
        )

from app.models import DocumentChunk


def load_chunks():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT document_id, chunk_index, text
            FROM document_chunks
            ORDER BY document_id, chunk_index
            """
        ).fetchall()

    return [
        DocumentChunk(
            document_id=row[0],
            chunk_index=row[1],
            text=row[2],
        )
        for row in rows
    ]