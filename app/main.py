from fastapi import FastAPI, UploadFile, File, HTTPException
from app.document_processor import extract_text_from_pdf
from app.document_processor import create_document_chunks
from app.benchmark import compare_retrieval_strategies
from app.retriever import (
    search_chunks,
    search_chunks_tfidf,
    search_chunks_semantic,
)
from app.database import initialize_database
from app.database import save_chunks
from app.database import load_chunks
from app.generator import generate_answer
from fastapi.middleware.cors import CORSMiddleware
import json
uploaded_chunks = []
app = FastAPI(
    title="RAG Evaluation Workbench",
    description="Evaluate and compare retrieval strategies for RAG systems.",
    version="0.1.0",
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
initialize_database()

@app.get("/")
def read_root():
    return {"message": "RAG Evaluation Workbench API is running"}
@app.get("/health")
def health_check():
    return {"status": "healthy"}
@app.get("/documents")
def get_documents():
    chunks = load_chunks()

    documents = {}

    for chunk in chunks:
        documents.setdefault(chunk.document_id, 0)
        documents[chunk.document_id] += 1

    return {
        "documents": [
            {
                "filename": filename,
                "chunk_count": chunk_count,
            }
            for filename, chunk_count in documents.items()
        ]
    }
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    contents = await file.read()

    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(contents)
        temp_path = Path(temp_file.name)

    try:
        text = extract_text_from_pdf(str(temp_path))
        chunks = create_document_chunks(text, document_id=file.filename)
        save_chunks(file.filename, chunks)
        uploaded_chunks[:] = [
          chunk for chunk in uploaded_chunks
          if chunk.document_id != file.filename
        ]
        uploaded_chunks.extend(chunks)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "text_preview": text[:200],
    }
@app.get("/search")
def search_documents(query: str, top_k: int = 3):
    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )
    chunks = load_chunks()
    results = search_chunks(query, chunks, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk in results
        ],
    }

@app.get("/ask")
def ask_question(
    query: str,
    top_k: int = 3,
    history: str | None = None,
    method: str = "Keyword",
):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    conversation_history = []

    if history:
        try:
            conversation_history = json.loads(history)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid conversation history format.",
            )

        if not isinstance(conversation_history, list):
            raise HTTPException(
                status_code=400,
                detail="Conversation history must be a list.",
            )

        if len(conversation_history) > 20:
            raise HTTPException(
                status_code=400,
                detail="Conversation history cannot exceed 20 messages.",
            )

        for item in conversation_history:
            if (
                not isinstance(item, dict)
                or item.get("role") not in {"user", "assistant"}
                or not isinstance(item.get("content"), str)
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Each history message must have "
                        "a valid role and text content."
                    ),
                )

    chunks = load_chunks()
    if method == "Keyword":
        results = search_chunks(query, chunks, top_k=top_k)
    elif method == "TF-IDF":
        results = search_chunks_tfidf(query, chunks, top_k=top_k)
    elif method == "Semantic":
        results = search_chunks_semantic(query, chunks, top_k=top_k)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid retrieval method. Choose Keyword, TF-IDF, or Semantic.",
        )

    answer = generate_answer(
        query,
        results,
        conversation_history=conversation_history,
    )

    return {
        "question": query,
        "answer": answer,
        "retrieved_chunks": [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk in results
        ],
    }
@app.get("/benchmark")
def benchmark_question(
    query: str,
    top_k: int = 3,
    relevant_ids: str = "",
):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    chunks = load_chunks()

    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "evaluation_dataset.json"

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        evaluation_dataset = json.load(dataset_file)

    matching_entry = next(
        (
            item for item in evaluation_dataset
            if item["query"].strip().lower() == query.strip().lower()
        ),
        None,
    )

    relevant_id_set = set(
        matching_entry["relevant_chunk_ids"]
        if matching_entry
        else []
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