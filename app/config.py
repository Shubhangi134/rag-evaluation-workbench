
"""Configuration settings for the RAG Workbench."""

import os


# Maximum number of PDFs a user can upload
MAX_PDFS_PER_SESSION = 3

# Maximum PDF size: 10 MB
MAX_PDF_SIZE_MB = 10
MAX_PDF_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

# Maximum pages allowed per PDF
MAX_PDF_PAGES = 50

# Delete inactive sessions after 30 minutes
SESSION_TIMEOUT_MINUTES = 30
SESSION_TIMEOUT_SECONDS = SESSION_TIMEOUT_MINUTES * 60

# Local LLM configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
