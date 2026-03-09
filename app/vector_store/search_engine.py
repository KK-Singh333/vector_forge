import numpy as np
from typing import List, Dict, Any
from utils.helper import cosine_similarity
import time
class SearchEngine:

    def __init__(
        self,
        vector_store_manager,
        metadata_store,
        embedder,
        logger=None
    ):
        self.vector_store = vector_store_manager
        self.metadata_store = metadata_store
        self.embedder = embedder
        self.logger = logger

    def search(
        self,
        query_text: str,
        user_id: int,
        k: int = 5
    ) -> List[Dict[str, Any]]:

        try:
            if not query_text or not query_text.strip():
                return []
            time2=time.time()
            query_vector = self._embed_query(query_text)
            embedding_time=time2-time.time()
            D, I = self.vector_store.search(
                query_vector=query_vector,
                user_id=user_id,
                k=k
            )

            if I is None or I.size == 0:
                return []

            ids = I[0].tolist()
            scores = D[0].tolist()

            filtered = [
                (int(vid), float(score))
                for vid, score in zip(ids, scores)
                if vid != -1
            ]

            if not filtered:
                return []

            chunk_ids = [vid for vid, _ in filtered]

            rows = self.metadata_store.get_chunk_metadata_by_chunk_id(chunk_ids)

            id_to_row = {
                row["chunk_id"]: row
                for row in rows
            }

            ordered_results = []
            for vid, score in filtered:
                if vid in id_to_row:
                    row = id_to_row[vid]

                    ordered_results.append({
                        "chunk_id": vid,  
                        "pdf_id": row["pdf_id"],
                        "page_no": row["page_no"],
                        "text": row["chunk_text"],
                        "score": cosine_similarity(query_vector,self._embed_query(row["chunk_text"])),
                        "embedding_time":embedding_time
                    })

            return ordered_results

        except Exception as e:
            if self.logger:
                self.logger.exception(f"[SEARCH ENGINE] ERROR: {e}")
            return []

    def _embed_query(self, query_text: str) -> np.ndarray:
        vector = self.embedder.model.encode([query_text])
        return np.asarray(vector, dtype=np.float32)
