import numpy as np
from typing import List
from utils.helper import group_embeddings_by_user
class StoreServant:
    def __init__(self,database_manager,logger):
        self.database_manger=database_manager
        self.logger=logger
    def __call__(self,embeddings:tuple[np.ndarray,list]):
        try:
            grouped_batches = group_embeddings_by_user(embeddings)
            if not grouped_batches:
                return
            for user_vectors, user_chunks in grouped_batches:
                global_ids=self.database_manger.allocate_vectors(
                    (user_vectors, user_chunks)
                )
                if not global_ids:
                    self.logger.warning(f'[INGESTION] BATCH INSERT FAILED')
        except Exception as e:
            self.logger.exception(f"[INGESTION] ERROR DURING BATCH INSERT: {e}")
            raise

