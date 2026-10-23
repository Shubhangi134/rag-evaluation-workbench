from app.models import DocumentChunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from functools import lru_cache
def search_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[DocumentChunk]:
    """Return the most relevant chunks using simple keyword matching."""
    if top_k <= 0 or not query.strip():
        return []

    query_words = set(query.lower().split())
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(chunk.text.lower().split())
        score = len(query_words & chunk_words)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:top_k]]

def search_chunks_tfidf(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[DocumentChunk]:
    """Retrieve relevant chunks using TF-IDF similarity."""
    if top_k <= 0 or not query.strip() or not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = (document_vectors @ query_vector.T).toarray().ravel()
    ranked_indices = scores.argsort()[::-1]

    return [
        chunks[i]
        for i in ranked_indices[:top_k]
        if scores[i] > 0
    ]
@lru_cache(maxsize=1)
def get_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@lru_cache(maxsize=1)
def get_semantic_embeddings(texts: tuple[str, ...]):
    model = get_semantic_model()
    return model.encode(list(texts), normalize_embeddings=True)
def search_chunks_semantic(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[DocumentChunk]:
    """Retrieve chunks using cached semantic embeddings."""
    if top_k <= 0 or not query.strip() or not chunks:
        return []

    model = get_semantic_model()

    texts = tuple(chunk.text for chunk in chunks)
    embeddings = get_semantic_embeddings(texts)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding[0]
    ranked_indices = scores.argsort()[::-1]

    return [chunks[i] for i in ranked_indices[:top_k]]