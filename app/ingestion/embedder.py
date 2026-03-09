import numpy as np
from typing import List, Optional, Tuple
from pydantic import BaseModel


class MetaDataSchema(BaseModel):
    user_id: int
    pdf_id: int
    page_no: int
    chunk_text: str




class Embedder:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(
        self,
        chunks: List[str],
        chunk_metadatas: List[MetaDataSchema],
    ) -> Optional[Tuple[np.ndarray,List[MetaDataSchema]]]:

        if not chunks:
            self.logger.warning("[EMBEDDING] EMPTY CHUNKS")
            return None

        try:
            # print(len(chunks))
            embeddings = self.model.encode(chunks)
            # print(embeddings)
            # print(len(embeddings))
            # print(len(chunk_metadatas))
            if embeddings is None or len(embeddings) == 0:
                self.logger.warning("[EMBEDDING] EMPTY EMBEDDINGS GENERATED")
                return None

            if len(embeddings) != len(chunk_metadatas):
                self.logger.error("[EMBEDDING] METADATA LENGTH MISMATCH")
                return None

            embeddings = np.asarray(embeddings, dtype=np.float32)

            self.logger.info(
                f"[EMBEDDING] SUCCESS "
                f"count={len(embeddings)} "
                f"user_ids={list({m.user_id for m in chunk_metadatas})}"
            )

            return embeddings,chunk_metadatas

        except Exception:
            self.logger.exception("[EMBEDDING] ERROR")
            return None
