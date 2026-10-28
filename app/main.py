
"""FastAPI endpoints for the RAG PDF chatbot."""

import json
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Header,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from app.config import (
    MAX_PDFS_PER_SESSION,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
)
from app.models import ChatMessage, SessionData
from app.session_manager import (
    create_session,
    get_session,
    delete_session,
    sessions,
    sessions_lock,
)
from app.document_processor import (
    extract_text_from_pdf,
    create_page_chunks,
)
from app.retriever import (
    search_chunks,
    search_chunks_tfidf,
    search_chunks_semantic,
)
from app.generator import generate_answer
from app.benchmark import compare_retrieval_strategies


app = FastAPI(
    title="RAG Evaluation Workbench",
    description="Chat with PDFs and compare retrieval methods.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Session helpers
# --------------------------------------------------

def require_session(
    x_session_id: str = Header(...),
) -> SessionData:
    """Get the session belonging to the current request."""

    session = get_session(x_session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired.",
        )

    return session


def get_all_chunks(session: SessionData):
    """Get document chunks from one session only."""

    with sessions_lock:
        return [
            chunk
            for chunks in session.documents.values()
            for chunk in chunks
        ]


def retrieve_documents(query, chunks, method, top_k):
    """Choose the requested retrieval method."""

    methods = {
        "Keyword": search_chunks,
        "TF-IDF": search_chunks_tfidf,
        "Semantic": search_chunks_semantic,
    }

    if method not in methods:
        raise HTTPException(
            status_code=400,
            detail="Choose Keyword, TF-IDF, or Semantic.",
        )

    return methods[method](query, chunks, top_k=top_k)


def format_chunks(chunks):
    """Prepare retrieved chunks for the frontend."""

    return [
        {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "RAG Evaluation Workbench API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# --------------------------------------------------
# Session endpoints
# --------------------------------------------------

@app.post("/sessions")
def start_session():
    """Create a new temporary user session."""

    session = create_session()

    return {
        "session_id": session.session_id,
        "message": "Session created.",
    }


@app.delete("/sessions")
def clear_session(
    session: SessionData = Depends(require_session),
):
    """Delete the current session and its data."""

    delete_session(session.session_id)

    return {"message": "Session deleted."}


# --------------------------------------------------
# Document endpoints
# --------------------------------------------------

@app.get("/documents")
def get_documents(
    session: SessionData = Depends(require_session),
):
    """List PDFs uploaded in this session."""

    with sessions_lock:
        documents = [
            {
                "filename": filename,
                "chunk_count": len(chunks),
            }
            for filename, chunks in session.documents.items()
        ]

    return {"documents": documents}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    session: SessionData = Depends(require_session),
):
    """Upload and process a PDF in the current session."""

    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    # Read at most the allowed size plus one byte.
    contents = await file.read(MAX_PDF_BYTES + 1)

    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=400,
            detail="PDF must be 10 MB or smaller.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    # Check the PDF before extracting its text.
    try:
        reader = PdfReader(BytesIO(contents))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported.",
            )

        if len(reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail="PDF cannot contain more than 50 pages.",
            )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    # Existing PDF extraction expects a file path.
    # Use a temporary file and remove it afterward.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        text = extract_text_from_pdf(str(temp_path))

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the PDF.",
            )

        chunks = create_page_chunks(
            file_path=str(temp_path),
            document_id=filename,
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    # Save only inside this user's temporary session.
    with sessions_lock:
        if sessions.get(session.session_id) is not session:
            raise HTTPException(
                status_code=404,
                detail="Session expired during upload.",
            )

        if (
            filename not in session.documents
            and len(session.documents) >= MAX_PDFS_PER_SESSION
        ):
            raise HTTPException(
                status_code=400,
                detail="Maximum 3 PDFs allowed per session.",
            )

        session.documents[filename] = chunks

        # Old conversation context may refer to old documents.
        session.messages.clear()

    return {
        "filename": filename,
        "content_type": file.content_type,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "text_preview": text[:200],
    }


# --------------------------------------------------
# Search endpoint
# --------------------------------------------------

@app.get("/search")
def search_documents(
    query: str,
    top_k: int = 3,
    method: str = "Keyword",
    session: SessionData = Depends(require_session),
):
    """Search only the current session's documents."""

    if not query.strip() or top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Enter a query and a valid top_k.",
        )

    chunks = get_all_chunks(session)

    results = retrieve_documents(
        query, chunks, method, top_k
    )

    return {
        "query": query,
        "results": format_chunks(results),
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.get("/ask")
def ask_question(
    query: str,
    top_k: int = 3,
    method: str = "Keyword",
    session: SessionData = Depends(require_session),
):
    """Answer a question using the session's PDFs."""

    if not query.strip() or top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Enter a question and a valid top_k.",
        )

    chunks = get_all_chunks(session)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking questions.",
        )

    # Save a snapshot of recent conversation history.
    with sessions_lock:
        history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in session.messages[-10:]
        ]

    results = retrieve_documents(
        query, chunks, method, top_k
    )

    try:
        answer = generate_answer(
            query,
            results,
            conversation_history=history,
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Answer generation is currently unavailable.",
        )

    # Remember this question and answer for follow-ups.
    with sessions_lock:
        if sessions.get(session.session_id) is not session:
            raise HTTPException(
                status_code=404,
                detail="Session expired during answer generation.",
            )

        session.messages.append(
            ChatMessage(role="user", content=query)
        )

        session.messages.append(
            ChatMessage(role="assistant", content=answer)
        )

        # Keep only the 20 most recent messages.
        session.messages[:] = session.messages[-20:]

    return {
        "question": query,
        "answer": answer,
        "retrieved_chunks": format_chunks(results),
    }


# --------------------------------------------------
# Benchmark endpoint
# --------------------------------------------------

@app.get("/benchmark")
def benchmark_question(
    query: str,
    top_k: int = 3,
    relevant_ids: str = "",
    session: SessionData = Depends(require_session),
):
    """Compare retrieval methods on session documents."""

    if not query.strip() or top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Enter a query and a valid top_k.",
        )

    chunks = get_all_chunks(session)

    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "evaluation_dataset.json"
    )

    with dataset_path.open(
        "r", encoding="utf-8"
    ) as dataset_file:
        evaluation_dataset = json.load(dataset_file)

    matching_entry = next(
        (
            item for item in evaluation_dataset
            if item["query"].strip().lower()
            == query.strip().lower()
        ),
        None,
    )

    relevant_id_set = set(
        matching_entry["relevant_chunk_ids"]
        if matching_entry else []
    )

    if relevant_ids:
        relevant_id_set.update(
            item.strip()
            for item in relevant_ids.split(",")
            if item.strip()
        )

    return {
        "query": query,
        "results": compare_retrieval_strategies(
            query=query,
            chunks=chunks,
            relevant_ids=relevant_id_set,
            k=top_k,
        ),
    }
