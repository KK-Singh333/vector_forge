"""
Comprehensive Metrics Reporter and Analysis Framework
Generates detailed reports with architecture overview, trade-off analysis, and recommendations
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import statistics


class MetricsReporter:
    """Generates comprehensive evaluation reports"""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
    
    def format_metric(self, value: float, as_percentage: bool = False) -> str:
        """Format metric for display"""
        if as_percentage:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    
    def format_latency(self, ms: float) -> str:
        """Format latency for display"""
        return f"{ms:.2f}ms"
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate concise summary report"""
        
        metrics = results["metrics"]
        latency = results["latency"]
        config = results["config"]
        caching = results["caching"]
        
        report = []
        report.append("\n" + "="*80)
        report.append("EVALUATION SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Timestamp: {self.timestamp}\n")
        
        # Configuration
        report.append("CONFIGURATION")
        report.append("-" * 80)
        report.append(f"Total Queries:       {config['total_queries']}")
        report.append(f"Positive Queries:    {config['positive_queries']}")
        report.append(f"Negative Queries:    {config['negative_queries']}")
        report.append(f"Caching Enabled:     {config['use_caching']}")
        report.append(f"Re-ranking Enabled:  {config['use_reranking']}")
        report.append(f"Top-K Parameter:     {config['top_k']}")
        report.append("")
        
        # Core Metrics
        report.append("CORE METRICS (MANDATORY REPORTING)")
        report.append("-" * 80)
        report.append(f"Recall@3:            {self.format_metric(metrics['recall_at_3'], True)}")
        report.append(f"  → Target: ≥90%")
        report.append(f"  → Status: {'✓ PASS' if metrics['recall_at_3'] >= 0.90 else '✗ FAIL'}")
        report.append("")
        
        report.append(f"Top-1 Accuracy:      {self.format_metric(metrics['top_1_accuracy'], True)}")
        report.append(f"  → Target: ≥75%")
        report.append(f"  → Status: {'✓ PASS' if metrics['top_1_accuracy'] >= 0.75 else '✗ FAIL'}")
        report.append("")
        
        report.append(f"Top-3 Accuracy:      {self.format_metric(metrics['top_3_accuracy'], True)}")
        report.append("")
        
        report.append(f"Mean Reciprocal Rank (MRR): {self.format_metric(metrics['mrr'], False)}")
        report.append("  → Measures how high first correct result appears (higher = better)")
        report.append("")
        
        report.append(f"nDCG (Ranking Quality):     {self.format_metric(metrics['ndcg'], False)}")
        report.append("  → Measures ranking quality with multiple relevant items (higher = better)")
        report.append("")
        
        report.append(f"False Positive Rate: {self.format_metric(metrics['false_positive_rate'], True)}")
        report.append("  → For negative queries (no correct answer)")
        report.append("")
        
        # Latency Analysis
        report.append("LATENCY ANALYSIS")
        report.append("-" * 80)
        
        if "end_to_end" in latency:
            e2e = latency["end_to_end"]
            report.append(f"End-to-End Latency (total pipeline):")
            report.append(f"  Mean:    {self.format_latency(e2e['mean'])}")
            report.append(f"  Median:  {self.format_latency(e2e['median'])}")
            report.append(f"  P95:     {self.format_latency(e2e['p95'])}")
            report.append(f"  P99:     {self.format_latency(e2e['p99'])}")
            report.append(f"  Min/Max: {self.format_latency(e2e['min'])} / {self.format_latency(e2e['max'])}")
            report.append("")
        
        report.append("Stage-wise Breakdown:")
        for stage in sorted(latency.keys()):
            if stage == "end_to_end":
                continue
            stats = latency[stage]
            report.append(f"\n  {stage.upper().replace('_', ' ')}:")
            report.append(f"    Mean:   {self.format_latency(stats['mean'])}")
            report.append(f"    P95:    {self.format_latency(stats['p95'])}")
            report.append(f"    Count:  {int(stats['count'])}")
        
        report.append("")
        
        # Caching Performance
        report.append("CACHING PERFORMANCE")
        report.append("-" * 80)
        
        query_cache_stats = caching["query_cache"]
        report.append(f"Query Result Cache:")
        report.append(f"  Hit Rate:     {self.format_metric(query_cache_stats['hit_rate'], True)}")
        report.append(f"  Hits:         {query_cache_stats['hits']}")
        report.append(f"  Misses:       {query_cache_stats['misses']}")
        report.append(f"  Cache Size:   {query_cache_stats['cache_size']}")
        report.append("")
        
        embedding_cache_stats = caching["embedding_cache"]
        report.append(f"Embedding Cache:")
        report.append(f"  Hit Rate:     {self.format_metric(embedding_cache_stats['hit_rate'], True)}")
        report.append(f"  Hits:         {embedding_cache_stats['hits']}")
        report.append(f"  Misses:       {embedding_cache_stats['misses']}")
        report.append(f"  Cache Size:   {embedding_cache_stats['cache_size']}")
        report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed analysis report"""
        
        report = []
        report.append("\n" + "="*80)
        report.append("DETAILED ANALYSIS REPORT")
        report.append("="*80 + "\n")
        
        # Query-level Analysis
        query_details = results["query_details"]
        total_correct_top1 = sum(1 for q in query_details if q["top_1_correct"])
        total_correct_top3 = sum(1 for q in query_details if q["top_3_correct"])
        cache_hits = sum(1 for q in query_details if q["cache_hit"])
        
        report.append("QUERY-LEVEL ANALYSIS")
        report.append("-" * 80)
        report.append(f"Total Queries Analyzed: {len(query_details)}")
        report.append(f"Correct Top-1: {total_correct_top1}/{len(query_details)}")
        report.append(f"Correct Top-3: {total_correct_top3}/{len(query_details)}")
        report.append(f"Cache Hits: {cache_hits}/{len(query_details)}")
        
        # Latency distribution by cache hit
        cached_latencies = [
            q["latency_ms"] for q in query_details if q["cache_hit"]
        ]
        non_cached_latencies = [
            q["latency_ms"] for q in query_details if not q["cache_hit"]
        ]
        
        if cached_latencies:
            report.append(f"\nCached Query Latency:")
            report.append(f"  Mean: {self.format_latency(statistics.mean(cached_latencies))}")
            report.append(f"  Median: {self.format_latency(statistics.median(cached_latencies))}")
        
        if non_cached_latencies:
            report.append(f"\nNon-Cached Query Latency:")
            report.append(f"  Mean: {self.format_latency(statistics.mean(non_cached_latencies))}")
            report.append(f"  Median: {self.format_latency(statistics.median(non_cached_latencies))}")
        
        if cached_latencies and non_cached_latencies:
            speedup = statistics.mean(non_cached_latencies) / statistics.mean(cached_latencies)
            report.append(f"\nCaching Speedup: {speedup:.2f}x")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def generate_architecture_overview(self) -> str:
        """Generate architecture overview document"""
        
        overview = """
================================================================================
ARCHITECTURE OVERVIEW - PRODUCTION EVALUATION FRAMEWORK
================================================================================

1. SYSTEM COMPONENTS
================================================================================

1.1 RETRIEVAL PIPELINE
    ┌────────────────────────────────────────────────────────────┐
    │ Query Input                                                 │
    └────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────┐
    │ Query Result Cache (LRU - 500 entries)                      │
    │ → Caches complete retrieval results                         │
    │ → O(1) lookup, eliminates downstream processing            │
    └────────────────────────────────────────────────────────────┘
                              │
                     ┌────────┴────────┐
              Cache Hit              Cache Miss
                │                       │
                ▼                       ▼
          Return Results      ┌─────────────────────────────┐
              (0.1ms)         │ API Call to Search Service   │
                              │ - Query Embedding           │
                              │ - FAISS Vector Search       │
                              │ - Metadata Retrieval        │
                              └─────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────────────────┐
                              │ Re-ranking (Optional)        │
                              │ - LLM Scoring               │
                              │ - Result Reordering         │
                              └─────────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────────┐
                         │ Cache Results & Return       │
                         └──────────────────────────────┘


1.2 CACHING STRATEGY (Multi-Level)
    
    Level 1: Query Result Cache
    ├─ Type: LRU (Least Recently Used)
    ├─ Size: 500 entries
    ├─ Key: [user_id]:[k]:[hash(query)]
    ├─ Hit Rate Target: >70% in typical usage
    ├─ Latency Improvement: ~99% reduction (cache hit in <1ms)
    └─ Use Case: Repeated queries within session
    
    Level 2: Embedding Cache
    ├─ Type: LRU (Least Recently Used)
    ├─ Size: 1000 entries
    ├─ Key: Query text
    ├─ Hit Rate Target: >60% across sessions
    ├─ Latency Improvement: ~80% reduction (embedding generation skipped)
    └─ Use Case: Semantically similar queries
    
    Expected Cache Benefits:
    ├─ Repeated queries: 0.1ms (vs. 200-500ms without cache)
    ├─ Similar queries: 150-300ms (vs. 200-500ms without cache)
    └─ First query: 200-500ms (no cache benefit)


1.3 LATENCY BREAKDOWN

    Stage 1: Query Embedding Generation (~100-200ms)
    ├─ Model: Sentence Transformers
    ├─ Approach: Multi-threaded embedding service
    ├─ Optimization: Can be cached for repeated queries
    ├─ Cache Impact: Skip if embedding cache hit
    
    Stage 2: Vector Search (~50-150ms)
    ├─ Engine: FAISS
    ├─ Operation: Similarity search
    ├─ Complexity: O(log N) with index optimization
    ├─ Result: Top-K vectors + distances
    ├─ Data Movement: GPU to CPU (if GPU-accelerated)
    
    Stage 3: Metadata Retrieval (~10-50ms)
    ├─ Source: Database/In-memory store
    ├─ Operation: Batch lookup by chunk IDs
    ├─ Optimization: Indexed by chunk_id
    ├─ Result: Chunk text, PDF info, page number
    
    Stage 4: Re-ranking (Optional) (~500-2000ms)
    ├─ Method: LLM-based relevance scoring
    ├─ Cost: High latency for quality improvement
    ├─ Trade-off: Quality vs Speed
    ├─ Recommendation: Use only when accuracy > latency priority
    
    TOTAL PIPELINE LATENCY
    ├─ Without Caching: 160-2400ms
    ├─ With Query Cache Hit: 0.1ms
    ├─ With Embedding Cache Hit: 50-200ms
    ├─ With Re-ranking: +500-2000ms


2. PERFORMANCE TARGETS & BENCHMARKS
================================================================================

2.1 RECALL METRICS (Mandatory)

    Recall@3: ≥90%
    ├─ Definition: % queries where correct chunk in top-3
    ├─ Target: ≥90% for production deployment
    ├─ Current Strategy: Focus on embedding quality
    ├─ Improvement Path: Re-ranking can help
    
    Top-1 Accuracy: ≥75%
    ├─ Definition: % queries with correct chunk ranked first
    ├─ Target: ≥75% for good user experience
    ├─ Current Strategy: Optimize semantic matching
    └─ Improvement Path: Hybrid search (keyword + semantic)
    
    Top-3 Accuracy: >85%
    ├─ Definition: % queries with correct chunk in top-3
    ├─ Target: >85% for reliable retrieval
    └─ Natural progression from Top-1
    
    MRR (Mean Reciprocal Rank): >0.80
    ├─ Definition: Average 1/rank of first correct result
    ├─ Target: >0.80 (avg rank of correct result < 1.25)
    └─ Improvement: Re-ranking specifically targets this
    
    nDCG (Normalized Discounted Cumulative Gain): >0.85
    ├─ Definition: Ranking quality with weighted penalty for lower ranks
    ├─ Target: >0.85 for high-quality rankings
    └─ Improvement: Multi-relevance ranking scenarios


2.2 LATENCY TARGETS (50% Reduction Goal)

    Current Baseline (Without Optimization):
    ├─ Mean: 300-500ms
    ├─ P95:  800-1200ms
    ├─ With Re-ranking: +800-2000ms
    
    Target After Optimization (50% reduction):
    ├─ Mean: 150-250ms (without re-ranking)
    ├─ P95:  400-600ms
    ├─ With Cache:  0.1-50ms (99% reduction)
    
    Optimization Strategies:
    ├─ 1. Query result caching (50-70% queries cached)
    ├─ 2. Embedding cache for similar queries
    ├─ 3. Batch processing for non-real-time use cases
    ├─ 4. GPU acceleration for embeddings (if available)
    ├─ 5. Async re-ranking (non-blocking)
    └─ 6. Connection pooling for API calls


3. TRADE-OFF ANALYSIS
================================================================================

3.1 RE-RANKING: Quality vs Speed

    WITH RE-RANKING (LLM-based scoring)
    ├─ Pros:
    │  ├─ +5-15% improvement in Recall@3
    │  ├─ +10-20% improvement in MRR
    │  └─ Better handling of complex queries
    ├─ Cons:
    │  ├─ +800-2000ms latency
    │  └─ 4-10x slower end-to-end
    ├─ Decision: Use when accuracy > latency priority
    ├─ Recommendation: Enable for batch/offline evaluation
    └─ NOT Recommended: For real-time user-facing queries
    
    WITHOUT RE-RANKING (Vector similarity only)
    ├─ Pros:
    │  ├─ Fast (150-250ms mean latency)
    │  ├─ Predictable performance
    │  └─ Scalable to millions of queries
    ├─ Cons:
    │  ├─ -5-15% accuracy vs re-ranked
    │  └─ Less nuanced relevance scoring
    ├─ Decision: Good for production serving
    └─ Recommendation: Default for real-time serving


3.2 CACHING DEPTH: Memory vs Hit Rate

    Shallow Cache (500 entries)
    ├─ Memory: ~50-100MB
    ├─ Hit Rate: 60-70% in typical usage
    └─ Recommendation: DEFAULT for single-user or small teams
    
    Deep Cache (2000+ entries)
    ├─ Memory: ~200-400MB
    ├─ Hit Rate: 80-90%
    └─ Recommendation: For high-query-volume scenarios
    
    Distributed Cache (Redis)
    ├─ Memory: Unlimited (external)
    ├─ Hit Rate: 80-95% cross-session
    └─ Recommendation: For multi-instance deployments


4. EVALUATION METRICS DASHBOARD
================================================================================

Core Metrics to Track:
├─ Recall@3          [CRITICAL] - Target ≥90%
├─ Top-1 Accuracy    [CRITICAL] - Target ≥75%
├─ Top-3 Accuracy    [IMPORTANT] - Target >85%
├─ MRR               [IMPORTANT] - Target >0.80
├─ nDCG              [IMPORTANT] - Target >0.85
├─ False Positive    [CRITICAL] - Target <10%
├─ Latency P95       [CRITICAL] - Target <600ms
├─ Cache Hit Rate    [IMPORTANT] - Target >70%
└─ Hallucination Rate [NEW]     - Target <5%


5. NEXT ITERATION ROADMAP
================================================================================

Phase 1: Caching Optimization (Week 1-2)
├─ ✓ Implement multi-level caching
├─ ✓ Add cache statistics
├─ ○ Measure real-world hit rates
└─ ○ Optimize cache eviction policy

Phase 2: Latency Analysis (Week 2-3)
├─ ✓ Stage-wise latency tracking
├─ ✓ Add P95/P99 percentiles
├─ ○ Profile embedding generation
└─ ○ Optimize vector search

Phase 3: Advanced Re-ranking (Week 3-4)
├─ Async re-ranking pipeline
├─ Conditional re-ranking (only for ambiguous queries)
├─ Weighted re-ranking with embedding scores
└─ Multi-stage re-ranking

Phase 4: Robustness & Quality (Week 4-5)
├─ Entity coverage measurement
├─ Paraphrase robustness testing
├─ Hallucination detection
└─ Negative query handling

Phase 5: Production Deployment (Week 5+)
├─ Load testing & scaling
├─ A/B testing framework
├─ Monitoring & alerting
└─ Documentation & runbooks


6. TECHNICAL SPECIFICATIONS
================================================================================

Query Result Cache:
├─ Implementation: OrderedDict (thread-safe with lock)
├─ Eviction: LRU (Least Recently Used)
├─ Size Limit: 500 entries
├─ TTL: None (session-based)
└─ Hit Rate Monitoring: Enabled

Embedding Cache:
├─ Implementation: OrderedDict (thread-safe)
├─ Eviction: LRU
├─ Size Limit: 1000 entries
├─ TTL: None
└─ Hit Rate Monitoring: Enabled

Latency Tracking:
├─ Granularity: Per-stage breakdown
├─ Metrics: Min, Max, Mean, Median, P95, P99
├─ Storage: In-memory list
└─ Export: JSON format

Metrics Reporting:
├─ Format: Interactive JSON + Text Reports
├─ Frequency: Per evaluation run
├─ Storage: eval/reports/ directory
└─ Archival: Timestamped snapshots

================================================================================
"""
        return overview
    
    def generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on results"""
        
        metrics = results["metrics"]
        latency = results["latency"]
        config = results["config"]
        
        recommendations = []
        recommendations.append("\n" + "="*80)
        recommendations.append("RECOMMENDATIONS & ACTION ITEMS")
        recommendations.append("="*80 + "\n")
        
        # Recall recommendations
        if metrics["recall_at_3"] < 0.90:
            recommendations.append(f"[PRIORITY-HIGH] Recall@3: {metrics['recall_at_3']:.2%} (Target: 90%)")
            recommendations.append("  → Action: Enable re-ranking for better relevance scoring")
            recommendations.append("  → Action: Improve embedding model quality")
            recommendations.append("  → Action: Add hybrid search (keyword + semantic)")
            recommendations.append("")
        else:
            recommendations.append(f"[✓ PASS] Recall@3: {metrics['recall_at_3']:.2%} - Meets target")
            recommendations.append("")
        
        # Top-1 accuracy
        if metrics["top_1_accuracy"] < 0.75:
            recommendations.append(f"[PRIORITY-HIGH] Top-1 Accuracy: {metrics['top_1_accuracy']:.2%} (Target: 75%)")
            recommendations.append("  → Action: Optimize embedding quality or similarity metric")
            recommendations.append("  → Action: Consider re-ranking to boost top-1 performance")
            recommendations.append("")
        else:
            recommendations.append(f"[✓ PASS] Top-1 Accuracy: {metrics['top_1_accuracy']:.2%} - Meets target")
            recommendations.append("")
        
        # Latency analysis
        if "end_to_end" in latency:
            e2e = latency["end_to_end"]
            if e2e["mean"] > 300:
                recommendations.append(f"[PRIORITY-MEDIUM] Latency: {e2e['mean']:.0f}ms mean (Optimize)")
                recommendations.append("  → Action: Verify caching is enabled and working")
                recommendations.append("  → Action: Consider GPU acceleration for embeddings")
                recommendations.append("  → Action: Profile each stage to identify bottleneck")
                recommendations.append("")
            else:
                recommendations.append(f"[✓ PASS] Latency: {e2e['mean']:.0f}ms mean - Good performance")
                recommendations.append("")
        
        # Cache hit rate
        query_cache_hit = results["caching"]["query_cache"]["hit_rate"]
        if query_cache_hit < 50:
            recommendations.append(f"[PRIORITY-LOW] Query Cache Hit Rate: {query_cache_hit:.1f}%")
            recommendations.append("  → Action: Increase cache size for better hit rate")
            recommendations.append("  → Action: Consider distributed cache (Redis) for multi-instance")
            recommendations.append("")
        else:
            recommendations.append(f"[✓ GOOD] Query Cache Hit Rate: {query_cache_hit:.1f}%")
            recommendations.append("")
        
        # Re-ranking recommendation
        if config["use_reranking"]:
            if latency.get("reranking", {}).get("mean", 0) > 1000:
                recommendations.append("[CONSIDERATION] Re-ranking adds >1000ms latency")
                recommendations.append("  → Evaluate if accuracy gain justifies the cost")
                recommendations.append("  → Consider async re-ranking or selective re-ranking")
                recommendations.append("")
        else:
            recommendations.append("[CONSIDERATION] Re-ranking disabled")
            if metrics["recall_at_3"] < 0.95:
                recommendations.append("  → Enable re-ranking to improve recall metrics")
            recommendations.append("")
        
        # False positive analysis
        if metrics["false_positive_rate"] > 0.10:
            recommendations.append(f"[PRIORITY-MEDIUM] False Positive Rate: {metrics['false_positive_rate']:.2%}")
            recommendations.append("  → Action: Add confidence thresholding")
            recommendations.append("  → Action: Implement query authenticity checking")
            recommendations.append("")
        
        recommendations.append("="*80)
        
        return "\n".join(recommendations)


def generate_production_report(results: Dict[str, Any]) -> None:
    """Generate and save all reports"""
    
    reporter = MetricsReporter()
    
    # Summary Report
    summary = reporter.generate_summary_report(results)
    print(summary)
    
    with open("eval/reports/summary_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    # Detailed Report
    detailed = reporter.generate_detailed_report(results)
    print(detailed)
    
    with open("eval/reports/detailed_report.txt", "w", encoding="utf-8") as f:
        f.write(detailed)
    
    # Architecture Overview
    overview = reporter.generate_architecture_overview()
    with open("eval/reports/architecture_overview.txt", "w", encoding="utf-8") as f:
        f.write(overview)
    
    # Recommendations
    recommendations = reporter.generate_recommendations(results)
    print(recommendations)
    
    with open("eval/reports/recommendations.txt", "w", encoding="utf-8") as f:
        f.write(recommendations)
    
    # JSON Report (for programmatic access)
    json_report = {
        "timestamp": reporter.timestamp,
        "metrics": results["metrics"],
        "latency": results["latency"],
        "caching": results["caching"],
        "config": results["config"],
        "summary": {
            "total_queries": len(results["query_details"]),
            "correct_top1": sum(1 for q in results["query_details"] if q["top_1_correct"]),
            "correct_top3": sum(1 for q in results["query_details"] if q["top_3_correct"]),
            "cache_hits": sum(1 for q in results["query_details"] if q["cache_hit"])
        }
    }
    
    with open("eval/reports/metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    
    print("\n✓ Reports saved:")
    print("  - eval/reports/summary_report.txt")
    print("  - eval/reports/detailed_report.txt")
    print("  - eval/reports/architecture_overview.txt")
    print("  - eval/reports/recommendations.txt")
    print("  - eval/reports/metrics_report.json")
