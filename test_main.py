from fastapi.testclient import TestClient

from app.main import app
from io import BytesIO
from pypdf import PdfWriter

client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG Evaluation Workbench API is running"
    }
def test_health_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
def test_upload_document_returns_filename_and_content_type():
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=72, height=72)

    pdf_buffer = BytesIO()
    pdf_writer.write(pdf_buffer)
    pdf_buffer.seek(0)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.pdf",
        "content_type": "application/pdf",
        "text_length": 0,
        "text_preview": "",
    }
def test_upload_document_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"Not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }