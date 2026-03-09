from services.ingestion_service import IngestionService
from ingestion.pdf_parser import PdfParser
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from vector_store.id_allocator import StoreServant
from vector_store.active_segment_manager import VectorStoreManager
from storage.database_manager import DataBaseStoreManager
from utils.logger import setup_logger
from sentence_transformers import SentenceTransformer
import time
logger=setup_logger()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pdf_parser=PdfParser(logger,max_pages=50)
# text=pdf_parser(r"C:\Users\Lenovo\Downloads\max-towers-brochure.pdf")
# print(type(text))
chunker=Chunker(chunk_size=250,chunk_overlap=50,logger=logger)
embedder=Embedder(model=model,logger=logger)
vector_store_manager=VectorStoreManager(
    dim=384,
    segment_dir="segments",
    upgrade_threshold=5000,
    max_vectors_per_segment=100_000,
    nlist=32,
    logger=logger
)

database_manager=DataBaseStoreManager(db_path=r"E:\Agmentis\Scalable_FAISS_Store\data\metadata.db",logger=logger)
database_manager.conn.execute('''
    INSERT INTO users (user_id, current_chunk_count)
    VALUES (1, 0);
''')
database_manager.conn.commit()
database_manager.register_vector_store_manager(vector_store_manager)
store_servant=StoreServant(database_manager=database_manager,logger=logger)
ingestion_service=IngestionService(
    parser=pdf_parser,
    chunker=chunker,
    embedder=embedder,
    store_servant=store_servant,
    logger=logger
)
time1=time.time()
ingestion_service.ingest_pdf(
    file_path=r"C:\Users\Lenovo\Downloads\max-towers-brochure.pdf",
    user_id=1,
    pdf_id=1
)
print(time.time()-time1)