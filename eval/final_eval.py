import json
import numpy as np
from tqdm import tqdm
import requests
import asyncio
from typing import List, Dict, Any
from llm_server.services.llm_client import LLMClient
import time
llm=LLMClient()
########################################
# CONFIG
########################################

TOP_K = 5
RECALL_K = 3
API_URL = "http://127.0.0.1:8068/search"

########################################
# LOAD DATASET
########################################

with open("E:\Agmentis\Scalable_FAISS_Store\eval\ground_truth.json", "r") as f:
    dataset = json.load(f)


def _parse_rerank_scores(
    response: str,
    expected_chunks: int
) -> Dict[int, int]:

    scores = {}

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("Chunk"):
            try:
                parts = line.split(":")
                idx = int(parts[0].split()[1])
                score = int(parts[1].strip())
                scores[idx] = score
            except Exception:
                continue

    for i in range(expected_chunks):
        if i not in scores:
            scores[i] = 0

    return scores


async def _rerank_chunks(
        query: str,
        chunks: List[Dict]
) -> List[Dict]:

    chunk_blocks = []
    for idx, c in enumerate(chunks):
        chunk_blocks.append(
            f"Chunk {idx}:\n{c['text'][:800]}\n"
        )

    rerank_prompt = f"""
You are a relevance scoring system.

Score each chunk from 0 to 100 based on relevance to the question.

Return strictly:

Chunk 0: <score>
Chunk 1: <score>
...

Question:
{query}

Chunks:
{chr(10).join(chunk_blocks)}
"""

    try:
        response = await asyncio.wait_for(
            llm.generate(rerank_prompt),
            timeout=10.0
        )

        scores = _parse_rerank_scores(response, len(chunks))

        for i, c in enumerate(chunks):
            c["rerank_score"] = scores.get(i, 0)

        return sorted(
            chunks,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )

    except Exception:
        return sorted(
            chunks,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
########################################
# YOUR RETRIEVER
########################################

def retrieve(query, k=TOP_K):

    response = requests.post(
        API_URL,
        json={
            "user_id": 1,
            "query": query,
            "k": k
        }
    ).json()

    chunks = response["results"]

    # run async reranker
    reranked = asyncio.run(_rerank_chunks(query, chunks))
    results = []

    for c in reranked[:k]:
        results.append({
            "chunk_id": c["chunk_id"],
            "score": c.get("rerank_score", c.get("score", 0))
        })

    return results


########################################
# METRIC HELPERS
########################################

def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg(retrieved_ids, relevant_ids, k=5):

    relevances = [1 if r in relevant_ids else 0 for r in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)

    if dcg(ideal) == 0:
        return 0

    return dcg(relevances) / dcg(ideal)


########################################
# METRICS
########################################

recall_at_k = 0
top1 = 0
top3 = 0
mrr = 0
ndcg_total = 0
false_positive = 0
negative_queries = 0


########################################
# EVALUATION LOOP
########################################

for sample in tqdm(dataset):

    query = sample["query"]
    relevant = set(sample["ground_truth_chunk_ids"])

    retrieved = retrieve(query, TOP_K)

    retrieved_ids = [r["chunk_id"] for r in retrieved]


    # ---------------------------
    # NEGATIVE QUERY TEST
    # ---------------------------

    if len(relevant) == 0:
        negative_queries += 1

        if len(retrieved_ids) > 0:
            false_positive += 1

        continue


    # ---------------------------
    # RECALL@K
    # ---------------------------

    if any(r in relevant for r in retrieved_ids[:RECALL_K]):
        recall_at_k += 1


    # ---------------------------
    # TOP 1 ACCURACY
    # ---------------------------

    if retrieved_ids and retrieved_ids[0] in relevant:
        top1 += 1


    # ---------------------------
    # TOP 3 ACCURACY
    # ---------------------------

    if any(r in relevant for r in retrieved_ids[:3]):
        top3 += 1


    # ---------------------------
    # MRR
    # ---------------------------

    for rank, r in enumerate(retrieved_ids, start=1):

        if r in relevant:
            mrr += 1 / rank
            break


    # ---------------------------
    # nDCG
    # ---------------------------

    ndcg_total += compute_ndcg(retrieved_ids, relevant)
    time.sleep(1.5)


########################################
# FINAL RESULTS
########################################

total_queries = len(dataset)
positive_queries = total_queries - negative_queries

results = {

    "Recall@3": recall_at_k / positive_queries,
    "Top1 Accuracy": top1 / positive_queries,
    "Top3 Accuracy": top3 / positive_queries,
    "MRR": mrr / positive_queries,
    "nDCG": ndcg_total / positive_queries,
    "False Positive Rate": false_positive / negative_queries if negative_queries else 0

}

print("\nEvaluation Results\n")

for k, v in results.items():
    print(f"{k}: {v:.4f}")