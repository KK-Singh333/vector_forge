from collections import OrderedDict
from typing import List, Dict, Any
import threading


class SearchService:

    def __init__(
        self,
        search_engine,
        cache_size: int = 100,
        logger=None
    ):
        self.search_engine = search_engine
        self.cache_size = cache_size
        self.logger = logger

        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def search(
        self,
        query_text: str,
        user_id: int,
        k: int = 5
    ) -> List[Dict[str, Any]]:

        if not query_text or not query_text.strip():
            return []

        cache_key = self._make_cache_key(query_text, user_id, k)

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)

                if self.logger:
                    self.logger.info(
                        f"[CACHE] HIT user={user_id}"
                    )

                return self._cache[cache_key]

        if self.logger:
            self.logger.info(
                f"[CACHE] MISS user={user_id}"
            )

        results = self.search_engine.search(
            query_text=query_text,
            user_id=user_id,
            k=k
        )
        with self._lock:
            self._cache[cache_key] = results
            self._cache.move_to_end(cache_key)

            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return results

    def invalidate_user(self, user_id: int):
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key[0] == user_id
            ]

            for key in keys_to_remove:
                del self._cache[key]

        if self.logger:
            self.logger.info(
                f"[CACHE] Invalidated for user={user_id}"
            )

    def clear(self):
        with self._lock:
            self._cache.clear()

        if self.logger:
            self.logger.info("[CACHE] Cleared")


    def _make_cache_key(self, query_text: str, user_id: int, k: int):
        return (user_id, query_text.strip(), k)
