
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from app.retriever import (
    search_chunks,
    search_chunks_tfidf,
    search_chunks_semantic,
)
from app.models import DocumentChunk


def main():
    corpus, queries, qrels = GenericDataLoader(
        data_folder="datasets/scifact"
    ).load(split="test")

    # Convert each SciFact document into one DocumentChunk.
    chunks = [
        DocumentChunk(
            document_id=doc_id,
            chunk_index=index,
            text=f"{doc.get('title', '')}\n{doc.get('text', '')}",
        )
        for index, (doc_id, doc) in enumerate(corpus.items())
    ]

    strategies = {
        "Keyword": search_chunks,
        "TF-IDF": search_chunks_tfidf,
        "Semantic": search_chunks_semantic,
    }

    totals = {
        name: {"precision": 0.0, "recall": 0.0, "queries": 0}
        for name in strategies
    }

    # BEIR expects query IDs mapped to document IDs and ranking scores.
    results_by_strategy = {name: {} for name in strategies}

    for query_id, query_text in queries.items():
        relevant_doc_ids = {
            doc_id
            for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0
        }

        if not relevant_doc_ids:
            continue

        for name, retrieve in strategies.items():
            retrieved = retrieve(query_text, chunks, top_k=5)

            # Preserve original SciFact document IDs for BEIR.
            results_by_strategy[name][query_id] = {
                chunk.document_id: 1.0 / rank
                for rank, chunk in enumerate(retrieved, start=1)
            }

            retrieved_doc_ids = [
                chunk.document_id for chunk in retrieved
            ]

            hits = sum(
                1
                for doc_id in retrieved_doc_ids
                if doc_id in relevant_doc_ids
            )

            totals[name]["precision"] += hits / 5
            totals[name]["recall"] += (
                hits / len(relevant_doc_ids)
            )
            totals[name]["queries"] += 1

    # Calculate nDCG@5 using SciFact relevance judgments.
    ndcg_results = {}

    for name, results in results_by_strategy.items():
        ndcg, _, _, _ = EvaluateRetrieval.evaluate(
            qrels,
            results,
            k_values=[5],
        )
        ndcg_results[name] = ndcg.get("NDCG@5", 0.0)

    print("\nSciFact Retrieval Evaluation (Top 5)")
    print("-" * 55)

    for name, values in totals.items():
        count = values["queries"]

        if count == 0:
            print(f"{name}: No queries evaluated")
            continue

        print(f"\n{name}")
        print(f"Queries evaluated: {count}")
        print(f"Mean Precision@5: {values['precision'] / count:.4f}")
        print(f"Mean Recall@5:    {values['recall'] / count:.4f}")
        print(f"Mean nDCG@5:      {ndcg_results[name]:.4f}")


if __name__ == "__main__":
    main()
