from services.ingestion_service import IngestionService
from vector_store.search_engine import SearchEngine
from services.search_service import SearchService
from vector_store.id_allocator import StoreServant
from vector_store.active_segment_manager import VectorStoreManager
from storage.database_manager import DataBaseStoreManager
from ingestion.pdf_parser import PdfParser
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from utils.logger import setup_logger
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

db_path = DATA_DIR / "metadata.db"

logger=setup_logger()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

database_manager=DataBaseStoreManager(db_path=db_path,logger=logger)

pdf_parser=PdfParser(logger,max_pages=50)
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
database_manager.register_vector_store_manager(vector_store_manager)
store_servant=StoreServant(database_manager=database_manager,logger=logger)
search_engine = SearchEngine(
    vector_store_manager=vector_store_manager,
    metadata_store=database_manager,
    embedder=embedder,
    logger=logger
)
cached_search_service = SearchService(
    search_engine=search_engine,
    cache_size=200
)

ingestion_service=IngestionService(
    parser=pdf_parser,
    chunker=chunker,
    embedder=embedder,
    store_servant=store_servant,
    logger=logger
)
