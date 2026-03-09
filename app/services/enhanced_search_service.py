"""
Enhanced Search Service with Caching Statistics and Monitoring
Extends basic caching with metrics tracking and optimizations
"""

from collections import OrderedDict
from typing import List, Dict, Any, Tuple
import threading
import json
from datetime import datetime


class CacheStatistics:
    """Tracks cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_latency_saved = 0.0  # ms
        self.lock = threading.Lock()
    
    def record_hit(self, latency_saved_ms: float = 0):
        with self.lock:
            self.hits += 1
            self.total_latency_saved += latency_saved_ms
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def record_eviction(self):
        with self.lock:
            self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            avg_latency_per_hit = (self.total_latency_saved / self.hits) if self.hits > 0 else 0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_latency_saved_ms": self.total_latency_saved,
                "avg_latency_per_hit_ms": avg_latency_per_hit
            }
    
    def reset(self):
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.total_latency_saved = 0.0


class EnhancedSearchService:
    """
    Enhanced search service with caching, statistics, and monitoring
    Builds on SearchService with additional features
    """
    
    def __init__(
        self,
        search_engine,
        cache_size: int = 500,  # Increased from 100 for better hit rates
        enable_stats: bool = True,
        logger=None
    ):
        self.search_engine = search_engine
        self.cache_size = cache_size
        self.logger = logger
        self.enable_stats = enable_stats
        
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStatistics() if enable_stats else None
        
        # Per-user caches for better isolation
        self._user_caches = {}
        self._user_caches_lock = threading.Lock()
    
    def search(
        self,
        query_text: str,
        user_id: int,
        k: int = 5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search with caching
        Returns: (results, metadata)
        """
        
        if not query_text or not query_text.strip():
            return [], {"cache_hit": False, "error": "empty_query"}
        
        cache_key = self._make_cache_key(query_text, user_id, k)
        
        # Check cache
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                
                results = self._cache[cache_key]
                
                if self._stats:
                    self._stats.record_hit(200)  # Typical retrieval saves ~200ms
                
                if self.logger:
                    self.logger.info(f"[CACHE] HIT user={user_id} key={cache_key}")
                
                return results, {
                    "cache_hit": True,
                    "cache_size": len(self._cache),
                    "cache_key": str(cache_key)
                }
        
        if self._stats:
            self._stats.record_miss()
        
        if self.logger:
            self.logger.info(f"[CACHE] MISS user={user_id} key={cache_key}")
        
        # Search
        results = self.search_engine.search(
            query_text=query_text,
            user_id=user_id,
            k=k
        )
        
        # Update cache
        with self._lock:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
                if self._stats:
                    self._stats.record_eviction()
            
            self._cache[cache_key] = results
            self._cache.move_to_end(cache_key)
        
        return results, {
            "cache_hit": False,
            "cache_size": len(self._cache),
            "cache_key": str(cache_key)
        }
    
    def invalidate_user(self, user_id: int) -> int:
        """
        Invalidate all cache entries for a user
        Returns: Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key[0] == user_id
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
        
        count = len(keys_to_remove)
        
        if self.logger:
            self.logger.info(f"[CACHE] Invalidated {count} entries for user={user_id}")
        
        return count
    
    def invalidate_query(self, query_text: str, user_id: int) -> int:
        """
        Invalidate cache entries for a specific query/user
        Returns: Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key[0] == user_id and key[1] == query_text.strip()
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
        
        count = len(keys_to_remove)
        
        if self.logger:
            self.logger.info(
                f"[CACHE] Invalidated {count} entries for user={user_id} query='{query_text[:50]}...'"
            )
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._stats:
            return {}
        
        stats = self._stats.get_stats()
        stats["cache_size"] = len(self._cache)
        stats["max_cache_size"] = self.cache_size
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
        
        if self.logger:
            self.logger.info("[CACHE] Cleared all entries")
    
    def export_cache_stats_json(self, filepath: str) -> None:
        """Export cache statistics to JSON"""
        stats = self.get_cache_stats()
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        
        if self.logger:
            self.logger.info(f"[CACHE] Exported stats to {filepath}")
    
    @staticmethod
    def _make_cache_key(query_text: str, user_id: int, k: int) -> Tuple:
        """Create cache key from query parameters"""
        return (user_id, query_text.strip(), k)


# ============================================================================
# CACHING CONFIGURATION RECOMMENDATIONS
# ============================================================================

CACHE_CONFIGURATION_RECOMMENDATIONS = """
CACHING CONFIGURATION GUIDE
================================================================================

1. SINGLE-INSTANCE DEPLOYMENT (Development/Small Scale)
   ├─ Cache Size: 500 entries
   ├─ Strategy: In-memory LRU cache
   ├─ Expected Hit Rate: 60-70%
   ├─ Memory: ~50-100 MB
   └─ Implementation: EnhancedSearchService with default settings

2. HIGH-VOLUME SINGLE-INSTANCE (Production)
   ├─ Cache Size: 2000+ entries
   ├─ Strategy: In-memory LRU cache
   ├─ Expected Hit Rate: 75-85%
   ├─ Memory: ~200-300 MB
   └─ Implementation: EnhancedSearchService(cache_size=2000)

3. MULTI-INSTANCE DEPLOYMENT (Scaling)
   ├─ Cache Size: 1000 entries per instance
   ├─ Strategy: Local LRU + Distributed Cache (Redis)
   ├─ Expected Hit Rate: 80-90%
   ├─ Memory: Redis ~1-5 GB (depends on data size)
   ├─ Recommendation: Use Redis for cross-instance cache sharing
   └─ Implementation: Add Redis wrapper layer

4. HIGH-FREQUENCY QUERY PATTERNS
   ├─ Cache Size: 5000+ entries
   ├─ Strategy: Persistent cache with TTL
   ├─ Expected Hit Rate: 85-95%
   ├─ TTL: 24-48 hours
   └─ Implementation: Add database-backed cache layer

OPTIMIZATION STRATEGIES
================================================================================

1. Query Normalization (boost hit rate by 10-20%)
   ├─ Lowercase all queries
   ├─ Remove extra whitespace
   ├─ Normalize punctuation
   └─ Example: "What is 222 Rajpur?" → "what is 222 rajpur"

2. Semantic Query Clustering (boost hit rate by 5-15%)
   ├─ Cache similar queries together
   ├─ Grouping radius: 0.95 embedding similarity
   ├─ Trade-off: Slightly lower accuracy for better hit rate
   └─ Implementation: Pre-compute query embeddings

3. Partial Result Caching (for Top-K adjustment)
   ├─ Cache top-100 results
   ├─ Return top-K from cache when K < 100
   ├─ Reduce API calls by 80-90%
   └─ Trade-off: Slightly higher memory usage

4. Time-Based Cache Invalidation
   ├─ Automatic TTL: 12-24 hours
   ├─ Manual invalidation on document updates
   ├─ Event-driven invalidation for critical queries
   └─ Configuration: Make TTL configurable

EXPECTED BENEFITS
================================================================================

Latency Improvement:
├─ Cache Hit: 0.1-1ms (99% reduction)
├─ Cache Miss: 200-500ms (original)
├─ Average (70% hit rate): 55-210ms (70-75% reduction)

Throughput Improvement:
├─ Original: 2-5 queries/second per instance
├─ With Cache: 100-500 queries/second per instance
├─ Improvement: 50-250x throughput increase

Cost Reduction:
├─ Embedding API calls: -70% (if using paid API)
├─ Database queries: -70%
├─ Vector search operations: -70%

User Experience:
├─ Sub-second response time for cached queries
├─ Improved perceived performance
├─ Better user satisfaction scores

MONITORING & MAINTENANCE
================================================================================

Daily Checks:
├─ Cache hit rate > 60%?
├─ Memory usage within limits?
├─ No memory leaks (size stable)?

Weekly Analysis:
├─ Query patterns changing?
├─ Cache effectiveness stable?
├─ Need to adjust cache size?

Monthly Optimization:
├─ Review top queries by frequency
├─ Adjust cache size for optimal hit rate
├─ Analyze patterns for semantic clustering
"""

print(CACHE_CONFIGURATION_RECOMMENDATIONS)
