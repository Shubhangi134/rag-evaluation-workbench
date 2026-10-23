from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset_name = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_path = "datasets"

corpus, queries, qrels = GenericDataLoader(
    data_folder=f"{data_path}/{dataset_name}"
).load(split="test")

print(f"Documents: {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Relevance judgments: {sum(len(items) for items in qrels.values())}")
print("\nExample query:")
first_query_id = next(iter(queries))
print(queries[first_query_id])
print("Relevant documents:", qrels[first_query_id])