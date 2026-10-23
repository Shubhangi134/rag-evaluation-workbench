
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

            results_by_strategy[name][query_id] = {
                chunk.document_id: 1.0 / rank
                for rank, chunk in enumerate(retrieved, start=1)
            }

    evaluated_qrels = {
        query_id: qrels[query_id]
        for query_id in results_by_strategy["Keyword"]
    }

    metrics_by_strategy = {}

    for name, results in results_by_strategy.items():
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            evaluated_qrels,
            results,
            k_values=[5],
        )

        metrics_by_strategy[name] = {
            "precision": precision["P@5"],
            "recall": recall["Recall@5"],
            "ndcg": ndcg["NDCG@5"],
        }

    print("\nSciFact Retrieval Evaluation (Top 5)")
    print("-" * 55)

    for name, metrics in metrics_by_strategy.items():

        print(f"\n{name}")
        print(f"Queries evaluated: {len(results_by_strategy[name])}")
        print(f"Mean Precision@5: {metrics['precision']:.4f}")
        print(f"Mean Recall@5:    {metrics['recall']:.4f}")
        print(f"Mean nDCG@5:      {metrics['ndcg']:.4f}")


if __name__ == "__main__":
    main()
