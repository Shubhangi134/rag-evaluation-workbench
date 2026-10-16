from fastapi import FastAPI, UploadFile, File, HTTPException
from app.document_processor import extract_text_from_pdf
from app.document_processor import create_document_chunks
from app.retriever import search_chunks
from app.database import initialize_database
from app.database import save_chunks
from app.database import load_chunks
uploaded_chunks = []
app = FastAPI(
    title="RAG Evaluation Workbench",
    description="Evaluate and compare retrieval strategies for RAG systems.",
    version="0.1.0",
)
initialize_database()

@app.get("/")
def read_root():
    return {"message": "RAG Evaluation Workbench API is running"}
@app.get("/health")
def health_check():
    return {"status": "healthy"}
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