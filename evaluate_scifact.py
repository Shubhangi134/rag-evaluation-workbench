
"""Compare BM25, TF-IDF, and Semantic retrieval on SciFact."""

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from app.evaluator import reciprocal_rank_at_k
from app.models import DocumentChunk
from app.retriever import (
    search_chunks_bm25,
    search_chunks_tfidf,
    search_chunks_semantic,
)


TOP_K = 5

# Start with 10 queries to check the evaluation pipeline.
# Set to None later to evaluate the entire test split.
MAX_QUERIES = None


def main():
    """Run SciFact document retrieval evaluation."""

    print("Loading SciFact dataset...")

    corpus, queries, qrels = GenericDataLoader(
        data_folder="datasets/scifact"
    ).load(split="test")

    # Represent each SciFact document as one chunk.
    # Keep the original document ID for relevance evaluation.
    chunks = [
        DocumentChunk(
            document_id=doc_id,
            chunk_index=index,
            text=f"{doc.get('title', '')}\n{doc.get('text', '')}",
        )
        for index, (doc_id, doc) in enumerate(corpus.items())
    ]

    strategies = {
        "BM25": search_chunks_bm25,
        "TF-IDF": search_chunks_tfidf,
        "Semantic": search_chunks_semantic,
    }

    # Only evaluate queries with ground-truth relevance labels.
    evaluation_queries = [
        (query_id, query_text)
        for query_id, query_text in queries.items()
        if any(
            score > 0
            for score in qrels.get(query_id, {}).values()
        )
    ]

    if MAX_QUERIES is not None:
        evaluation_queries = evaluation_queries[:MAX_QUERIES]

    print(f"Documents: {len(chunks)}")
    print(f"Queries to evaluate: {len(evaluation_queries)}")

    if not evaluation_queries:
        print("No evaluation queries found.")
        return

    results_by_strategy = {
        name: {} for name in strategies
    }

    reciprocal_ranks = {
        name: [] for name in strategies
    }

    for query_number, (query_id, query_text) in enumerate(
        evaluation_queries,
        start=1,
    ):
        relevant_doc_ids = {
            doc_id
            for doc_id, score in qrels[query_id].items()
            if score > 0
        }

        print(
            f"Evaluating query "
            f"{query_number}/{len(evaluation_queries)}"
        )

        for name, retrieve in strategies.items():
            retrieved = retrieve(
                query_text,
                chunks,
                top_k=TOP_K,
            )

            retrieved_ids = [
                chunk.document_id
                for chunk in retrieved
            ]

            # BEIR expects document IDs mapped to ranking scores.
            results_by_strategy[name][query_id] = {
                doc_id: 1.0 / rank
                for rank, doc_id in enumerate(
                    retrieved_ids,
                    start=1,
                )
            }

            # Reciprocal rank for this individual query.
            rr = reciprocal_rank_at_k(
                retrieved_ids,
                relevant_doc_ids,
                TOP_K,
            )

            reciprocal_ranks[name].append(rr)

    # Use the same ground-truth queries for every method.
    evaluated_qrels = {
        query_id: qrels[query_id]
        for query_id, _ in evaluation_queries
    }

    print("\nSciFact Retrieval Evaluation")
    print(f"Top K: {TOP_K}")
    print(f"Queries evaluated: {len(evaluation_queries)}")
    print("-" * 55)

    for name, results in results_by_strategy.items():
        ndcg, _map, recall, precision = (
            EvaluateRetrieval.evaluate(
                evaluated_qrels,
                results,
                k_values=[TOP_K],
            )
        )

        # MRR is the mean of reciprocal ranks across queries.
        mrr = (
            sum(reciprocal_ranks[name])
            / len(reciprocal_ranks[name])
        )

        print(f"\n{name}")
        print(f"Precision@{TOP_K}: {precision[f'P@{TOP_K}']:.4f}")
        print(f"Recall@{TOP_K}:    {recall[f'Recall@{TOP_K}']:.4f}")
        print(f"nDCG@{TOP_K}:      {ndcg[f'NDCG@{TOP_K}']:.4f}")
        print(f"MRR@{TOP_K}:       {mrr:.4f}")


if __name__ == "__main__":
    main()
