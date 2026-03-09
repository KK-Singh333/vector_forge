import os
from typing import Optional
from ingestion.embedder import MetaDataSchema

class IngestionService:

    def __init__(
        self,
        parser,
        chunker,
        embedder,
        store_servant,
        logger=None
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store_servant=store_servant
        self.logger = logger

    

    def ingest_pdf(
        self,
        file_path: str,
        user_id: int,
        pdf_id: str
    ) -> bool:

        try:
            if not os.path.exists(file_path):
                if self.logger:
                    self.logger.error(f"[INGESTION] File not found: {file_path}")
                return False

            pages = self.parser(file_path)

            if not pages:
                if self.logger:
                    self.logger.warning("[INGESTION] Empty parsed pages")
                return False

            total_chunks = 0

            for page_no, page in enumerate(pages):

                page_chunks = self.chunker(page)

                if not page_chunks:
                    continue

                metadatas = [
                    MetaDataSchema(
                        user_id=user_id,
                        pdf_id=pdf_id,
                        page_no=page_no,
                        chunk_text=text
                    )
                    for text in page_chunks
                ]

                embeddings_tuple = self.embedder(page_chunks, metadatas)

                if not embeddings_tuple:
                    if self.logger:
                        self.logger.error(
                            f"[INGESTION] Embedding failed for page {page_no}"
                        )
                    return False

                self.store_servant(embeddings_tuple)

                total_chunks += len(page_chunks)

            if self.logger:
                self.logger.info(
                    f"[INGESTION] Success | user={user_id} | pdf={pdf_id} | chunks={total_chunks}"
                )

            return True

        except Exception as e:
            if self.logger:
                self.logger.exception(f"[INGESTION] Fatal error: {e}")
            return False