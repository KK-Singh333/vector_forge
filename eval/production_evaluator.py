"""
Production-Ready Evaluation Framework with Comprehensive Metrics
Includes: Caching strategy, latency tracking, and mandatory reporting
"""

import json
import numpy as np
from tqdm import tqdm
import requests
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict, defaultdict
from llm_server.services.llm_client import LLMClient
import statistics

# ============================================================================
# CONFIG
# ============================================================================

TOP_K = 5
RECALL_K = 3
API_URL = "http://127.0.0.1:8068/search"
GROUND_TRUTH_PATH = r"E:\Agmentis\Scalable_FAISS_Store\eval\ground_truth.json"


# ============================================================================
# CACHING LAYER
# ============================================================================

class EmbeddingCache:

    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[Any]:
        if query in self.cache:
            self.cache.move_to_end(query)
            self.hits += 1
            return self.cache[query]
        self.misses += 1
        return None

    def put(self, query: str, embedding: Any):
        if query in self.cache:
            self.cache.move_to_end(query)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[query] = embedding

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class QueryResultCache:

    def __init__(self, max_size: int = 500):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def make_key(self, query: str, k: int, user_id: int = 1):
        return f"{user_id}:{k}:{hash(query)}"

    def get(self, query: str, k: int, user_id: int = 1):
        key = self.make_key(query, k, user_id)

        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, query: str, k: int, results: List[Dict], user_id: int = 1):
        key = self.make_key(query, k, user_id)

        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = results

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# ============================================================================
# LATENCY TRACKER
# ============================================================================

class LatencyTracker:

    def __init__(self):
        self.stages = defaultdict(list)
        self.end_to_end = []

    def record_stage(self, stage: str, duration_ms: float):
        self.stages[stage].append(duration_ms)

    def record_total(self, duration_ms: float):
        self.end_to_end.append(duration_ms)

    def get_stage_stats(self, stage: str):

        if stage not in self.stages or not self.stages[stage]:
            return {}

        values = self.stages[stage]

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

    def get_all_stats(self):

        stats = {}

        for stage in self.stages:
            stats[stage] = self.get_stage_stats(stage)

        if self.end_to_end:
            stats["end_to_end"] = {
                "mean": statistics.mean(self.end_to_end),
                "median": statistics.median(self.end_to_end),
                "p95": np.percentile(self.end_to_end, 95),
                "p99": np.percentile(self.end_to_end, 99),
                "min": min(self.end_to_end),
                "max": max(self.end_to_end),
                "count": len(self.end_to_end)
            }

        return stats

    def clear(self):
        self.stages.clear()
        self.end_to_end.clear()


# ============================================================================
# RE-RANKER
# ============================================================================

class ReRanker:

    def __init__(self, llm_client):
        self.llm = llm_client

    def _parse_rerank_scores(self, response: str, expected_chunks: int):

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

    async def rerank(self, query: str, chunks: List[Dict], tracker=None):

        start_time = time.time()

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
                self.llm.generate(rerank_prompt),
                timeout=10.0
            )

            scores = self._parse_rerank_scores(response, len(chunks))

            for i, c in enumerate(chunks):
                c["rerank_score"] = scores.get(i, 0)

            reranked = sorted(
                chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

        except Exception:

            reranked = sorted(
                chunks,
                key=lambda x: x.get("score", 0),
                reverse=True
            )

        latency_ms = (time.time() - start_time) * 1000

        if tracker:
            tracker.record_stage("reranking", latency_ms)

        return reranked, latency_ms


# ============================================================================
# RETRIEVER (FIXED chunk_id TYPE)
# ============================================================================

class CachedRetriever:

    def __init__(self, query_cache, embedding_cache, use_reranking=False, reranker=None):

        self.query_cache = query_cache
        self.embedding_cache = embedding_cache
        self.use_reranking = use_reranking
        self.reranker = reranker

    def retrieve(self, query, k=TOP_K, user_id=1, tracker=None):

        latency_breakdown = {}

        cached_results = self.query_cache.get(query, k, user_id)

        if cached_results is not None:
            latency_breakdown["cache_hit"] = True

            if tracker:
                tracker.record_total(0.1)

            return cached_results, latency_breakdown

        latency_breakdown["cache_hit"] = False

        start_time = time.time()

        try:
            # print(query)
            response = requests.post(
                API_URL,
                json={
                    "user_id": 1,
                    "query": query,
                    "k": k
                },
                timeout=30
            ).json()

            api_latency = (time.time() - start_time) * 1000
            latency_breakdown["api_call"] = api_latency

            if tracker:
                tracker.record_stage("api_call", api_latency)
            # print(response)
            chunks = response.get("results", [])
            # print(chunks)

            results = []

            for c in chunks[:k]:

                try:
                    chunk_id = int(c["chunk_id"])
                    # print(chunk_id)
                except Exception:
                    continue

                results.append({
                    "chunk_id": chunk_id,
                    "score": c.get("score", 0),
                    "text": c.get("text", "")
                })

            self.query_cache.put(query, k, results, user_id)

            # Optional re-ranking step (may be async) — runs only if reranker provided and enabled
            if self.use_reranking and self.reranker is not None and results:
                try:
                    # ReRanker.rerank is async; run it synchronously here
                    reranked, rerank_latency = asyncio.run(self.reranker.rerank(query, results, tracker=tracker))
                    results = reranked
                    latency_breakdown["reranking"] = rerank_latency
                except Exception as e:
                    # If reranking fails, keep original ordering
                    print(f"Re-ranking failed: {e}")

            total_latency = (time.time() - start_time) * 1000
            latency_breakdown["total"] = total_latency

            if tracker:
                tracker.record_total(total_latency)

            return results, latency_breakdown

        except Exception as e:

            print(f"Retrieval error: {e}")
            return [], latency_breakdown


# ============================================================================
# METRICS
# ============================================================================

def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg(retrieved_ids, relevant_ids, k=5):

    relevances = [1 if r in relevant_ids else 0 for r in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)

    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0

    return dcg(relevances) / ideal_dcg


def compute_mrr(retrieved_ids, relevant_ids):

    for rank, r in enumerate(retrieved_ids, start=1):

        if r in relevant_ids:
            return 1 / rank

    return 0


def compute_recall_at_k(retrieved_ids, relevant_ids, k):

    return any(r in relevant_ids for r in retrieved_ids[:k])


# ============================================================================
# EVALUATOR
# ============================================================================

class ProductionEvaluator:

    def __init__(
        self,
        ground_truth_path,
        use_caching=True,
        use_reranking=False
    ):

        self.use_caching = use_caching
        self.use_reranking = use_reranking

        with open(ground_truth_path, "r") as f:
            self.dataset = json.load(f)

        self.query_cache = QueryResultCache()
        self.embedding_cache = EmbeddingCache()

        self.retriever = CachedRetriever(
            self.query_cache,
            self.embedding_cache
        )

        self.tracker = LatencyTracker()

    def evaluate(self,verbose=True):

        positive_queries = 0
        negative_queries = 0
        false_positives = 0

        recall_at_k_count = 0
        top1_count = 0
        top3_count = 0
        mrr_total = 0
        ndcg_total = 0

        query_details = []

        # Use the shared tracker when calling the retriever so per-stage latencies are recorded
        for sample in tqdm(self.dataset):

            query = sample.get("query")
            relevant_ids = set(sample.get("ground_truth_chunk_ids", []))

            results, latency = self.retriever.retrieve(query, k=TOP_K, tracker=self.tracker)

            retrieved_ids = [r.get("chunk_id") for r in results]

            # Query-level record
            qrec = {
                "query": query,
                "relevant_ids": list(relevant_ids),
                "retrieved_ids": retrieved_ids,
                "cache_hit": bool(latency.get("cache_hit", False)),
                "latency": latency,
                "top_1_correct": False,
                "top_3_correct": False
            }

            # Derive a single latency_ms value for reporting (prefer total, fallback to api_call)
            try:
                latency_ms = float(
                    latency.get("total") if latency.get("total") is not None else (
                        latency.get("api_call") if latency.get("api_call") is not None else (
                            0.1 if latency.get("cache_hit") else 0.0
                        )
                    )
                )
            except Exception:
                latency_ms = 0.0

            qrec["latency_ms"] = latency_ms

            if len(relevant_ids) == 0:
                negative_queries += 1

                if len(retrieved_ids) > 0:
                    false_positives += 1

                query_details.append(qrec)
                continue

            positive_queries += 1

            # Metrics
            if compute_recall_at_k(retrieved_ids, relevant_ids, RECALL_K):
                recall_at_k_count += 1
                qrec["top_3_correct"] = True

            if retrieved_ids and retrieved_ids[0] in relevant_ids:
                top1_count += 1
                qrec["top_1_correct"] = True

            if compute_recall_at_k(retrieved_ids, relevant_ids, 3):
                top3_count += 1

            mrr_total += compute_mrr(retrieved_ids, relevant_ids)

            ndcg_total += compute_ndcg(retrieved_ids, relevant_ids, TOP_K)

            query_details.append(qrec)

        # Aggregate metrics
        metrics = {
            "recall_at_3": (recall_at_k_count / positive_queries) if positive_queries else 0.0,
            "top_1_accuracy": (top1_count / positive_queries) if positive_queries else 0.0,
            "top_3_accuracy": (top3_count / positive_queries) if positive_queries else 0.0,
            "mrr": (mrr_total / positive_queries) if positive_queries else 0.0,
            "ndcg": (ndcg_total / positive_queries) if positive_queries else 0.0,
            "false_positive_rate": (false_positives / negative_queries) if negative_queries else 0.0
        }

        # Latency and cache stats
        latency_stats = self.tracker.get_all_stats()

        caching_stats = {
            "query_cache": self.query_cache.stats(),
            "embedding_cache": self.embedding_cache.stats()
        }

        config = {
            "use_caching": self.use_caching,
            "use_reranking": self.use_reranking,
            "top_k": TOP_K,
            "recall_k": RECALL_K,
            "api_url": API_URL
        }

        return {
            "metrics": metrics,
            "latency": latency_stats,
            "caching": caching_stats,
            "query_details": query_details,
            "config": {
                **config,
                "total_queries": len(self.dataset),
                "positive_queries": positive_queries,
                "negative_queries": negative_queries
            }
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    evaluator = ProductionEvaluator(GROUND_TRUTH_PATH)

    results = evaluator.evaluate()

    print("\nEvaluation Results\n")

    for k, v in results.items():
        print(f"{k}: {v:.4f}")