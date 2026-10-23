# RAG Evaluation Workbench

A Retrieval-Augmented Generation (RAG) evaluation project that compares retrieval strategies using a real-world scientific information retrieval benchmark.

## Overview

The RAG Evaluation Workbench explores how different retrieval methods identify relevant information for a given query. It includes a document-based question-answering application and a benchmark evaluation pipeline.

The project compares three retrieval strategies:
- **Keyword Retrieval** — matches query terms against document text.
- **TF-IDF Retrieval** — ranks documents using term frequency and inverse document frequency.
- **Semantic Retrieval** — uses sentence embeddings to identify semantically similar content.

## Technology Stack

- Python
- FastAPI
- React, TypeScript, and Vite
- SQLite
- Sentence Transformers
- BEIR SciFact benchmark

## Evaluation Results

The retrieval strategies were evaluated on the BEIR SciFact test split using 300 queries and a corpus of 5,183 scientific documents.

### Top-5 Retrieval Results

Evaluated on the BEIR SciFact test split using 300 queries and 5,183 scientific documents.

| Retrieval Method | Precision@5 | Recall@5 | nDCG@5 |
|---|---:|---:|---:|
| Keyword | 0.0940 | 0.4397 | 0.3774 |
| TF-IDF | 0.1413 | 0.6466 | 0.5547 |
| Semantic | **0.1640** | **0.7379** | **0.6293** |

**Key finding:** Semantic retrieval achieved the highest Precision@5, Recall@5, and nDCG@5 among the three methods in this evaluation run.

*These results use the project's custom evaluation script and should be considered preliminary until the metric calculations and relevance mappings are validated. Retrieval metrics assess document ranking and relevance, not generated-answer correctness.*

## Installation and Setup

### Prerequisites

- Python 3.12+
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Shubhangi134/rag-evaluation-workbench.git
cd rag-evaluation-workbench
```

### 2. Set up the Python environment

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn app.main:app --reload
```

The FastAPI server should be available at `http://127.0.0.1:8000`.

### 4. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your terminal.

### 5. Run the SciFact evaluation

From the project root, with the virtual environment activated:

```bash
python download_scifact.py
python evaluate_scifact.py
```

The first command downloads the SciFact dataset into `datasets/`. The evaluation compares Keyword, TF-IDF, and Semantic retrieval using Precision@5, Recall@5, and nDCG@5.

**Note:** The SciFact dataset is downloaded separately and is not included in the Git repository.  The PDF documents uploaded through the application and local environment configuration are also excluded from version control.