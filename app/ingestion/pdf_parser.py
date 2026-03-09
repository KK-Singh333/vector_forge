import pymupdf
import os
from utils.helper import clean_text
from typing import Optional
class PdfParser:
    def __init__(self,logger,max_pages:int):
        self.logger=logger
        self.max_pages=max_pages
    def __call__(self,path:str)->Optional[str]:
        if not os.path.exists(path):
            self.logger.error(f"[PARSING] FILE NOT FOUND : {path}")
            return None
        pdf_content=[]
        try:
            with pymupdf.open(path) as pdf:
                num_pages=len(pdf)
                if num_pages>self.max_pages:
                    self.logger.warning(f'[PARSING] WARNING PDF {path} TRIMMMED TO {self.max_pages} PAGES')
                processed_pages=min(num_pages,self.max_pages)
                for i in range(processed_pages):
                    text=pdf[i].get_text()
                    if text:
                        pdf_content.append(clean_text(text))
                self.logger.info(
    f"[PARSING] SUCCESS path={path} "
    f"total_pages={num_pages} | processed_pages={processed_pages}"
)
        except Exception as e:
            self.logger.exception(f"[PARSING] ERROR {path} | {e}")
            return None
        return pdf_content


