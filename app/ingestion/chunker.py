from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List,Optional
from utils.helper import is_noise
class Chunker:
    def __init__(self,chunk_size:int,chunk_overlap:int,logger):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.logger=logger
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    def __call__(self,text:str)->Optional[List[str]]:
        '''Chunks the given string into smaller strings of size chunk_size, Returns a list of strings'''
        if not text:
            self.logger.warning('[CHUNKING] WARNING EMPTY TEXT')
            return []
        try:
            chunks=self.text_splitter.split_text(text)
            filtered_chunks=[]
            for chunk in chunks:
                if not is_noise(chunk):
                    filtered_chunks.append(chunk)
                else:
                    self.logger.warning(f'[CHUNKING] {chunk} DROPPED')
            if not chunks:
                self.logger.warning('[CHUNKING] WARNING EMPTY CHUNKS')
                return []
            self.logger.info(
    f"[CHUNKING] SUCCESS chunks={len(chunks)} "
    f"chunk_size={self.chunk_size} "
    f"overlap={self.chunk_overlap}"
)
            return filtered_chunks
        except Exception as e:
            self.logger.exception(f'[CHUNKING] ERROR {e}')
            return []
        
        