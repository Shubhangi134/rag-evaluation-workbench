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

def test_upload_document_rejects_blank_pdf():
    """A PDF without readable text should be rejected."""

    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=72, height=72)

    pdf_buffer = BytesIO()
    pdf_writer.write(pdf_buffer)
    pdf_buffer.seek(0)

    # Create a temporary session first.
    session_response = client.post("/sessions")
    assert session_response.status_code == 200

    session_id = session_response.json()["session_id"]

    # Upload the blank PDF using the session ID.
    response = client.post(
        "/documents/upload",
        headers={"x-session-id": session_id},
        files={
            "file": (
                "sample.pdf",
                pdf_buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    # Blank PDFs do not contain searchable text.
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "No readable text found in the PDF."
    )

    # Clean up the test session.
    client.delete(
        "/sessions",
        headers={"x-session-id": session_id},
    )

def test_upload_document_rejects_blank_pdf():
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=72, height=72)

    pdf_buffer = BytesIO()
    pdf_writer.write(pdf_buffer)

    session_response = client.post("/sessions")
    session_id = session_response.json()["session_id"]

    response = client.post(
        "/documents/upload",
        headers={"x-session-id": session_id},
        files={
            "file": (
                "sample.pdf",
                pdf_buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "No readable text found in the PDF."
    )

def test_upload_document_rejects_non_pdf():
    """Check that the API rejects non-PDF files."""

    # Create a temporary session.
    session_response = client.post("/sessions")
    assert session_response.status_code == 200

    session_id = session_response.json()["session_id"]

    # Attempt to upload a text file.
    response = client.post(
        "/documents/upload",
        headers={"x-session-id": session_id},
        files={
            "file": (
                "sample.txt",
                b"This is not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are allowed."

    # Remove the temporary session.
    client.delete(
        "/sessions",
        headers={"x-session-id": session_id},
    )
