from fastapi import FastAPI, UploadFile, File, HTTPException
from app.document_processor import extract_text_from_pdf

app = FastAPI(
    title="RAG Evaluation Workbench",
    description="Evaluate and compare retrieval strategies for RAG systems.",
    version="0.1.0",
)


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
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "text_length": len(text),
        "text_preview": text[:200],
    }