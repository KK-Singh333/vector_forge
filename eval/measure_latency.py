import json
import requests
import numpy as np
from tqdm import tqdm
import random
import time
GROUND_TRUTH_PATH = r"E:\Agmentis\Scalable_FAISS_Store\eval\ground_truth.json"
CHAT_URL = "http://127.0.0.1:8069/chat"

USER_ID = 1
TOP_K = 3


def mean(values):
    return float(np.mean(values)) if values else 0


def p95(values):
    return float(np.percentile(values, 95)) if values else 0


# ------------------------------------------
# Load queries
# ------------------------------------------

with open(GROUND_TRUTH_PATH, "r") as f:
    dataset = json.load(f)

queries = [d["query"] for d in dataset]

random.shuffle(queries)
embedding_times = []
retrieval_times = []
rerank_times = []
generation_times = []


# ------------------------------------------
# Run benchmark
# ------------------------------------------
i=0
for q in tqdm(queries):
    i+=1
    if i==20:
        break
    time.sleep(5)
    response = requests.post(
        CHAT_URL,
        json={
            "user_id": USER_ID,
            "query": q,
            "k": TOP_K
        },
        timeout=60
    ).json()
    # print(response)
    # -------------------------
    # Embedding time
    # -------------------------

    sources = response.get("sources", [])

    if sources:
        emb = sources[0].get("embedding_time")
        if emb is not None:
            embedding_times.append(emb)

    # -------------------------
    # Retrieval
    # -------------------------

    retrieval_times.append(response.get("vector_db_time", 0))

    # -------------------------
    # Reranking
    # -------------------------

    rerank_times.append(response.get("reranking_time", 0))

    # -------------------------
    # Generation
    # -------------------------

    generation_times.append(response.get("generation_time", 0))


# ------------------------------------------
# Report
# ------------------------------------------

print("\n================ LATENCY BREAKDOWN ================\n")

print("Embedding Time")
print(f"Mean: {-1*mean(embedding_times):.4f}s")
print(f"P95 : {-1*p95(embedding_times):.4f}s\n")

print("Retrieval Time")
print(f"Mean: {mean(retrieval_times):.4f}s")
print(f"P95 : {p95(retrieval_times):.4f}s\n")

print("Re-ranking Time")
print(f"Mean: {mean(rerank_times):.4f}s")
print(f"P95 : {p95(rerank_times):.4f}s\n")

print("Generation Time")
print(f"Mean: {mean(generation_times):.4f}s")
print(f"P95 : {p95(generation_times):.4f}s\n")