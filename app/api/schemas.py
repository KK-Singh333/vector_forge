from pydantic import BaseModel


class IngestRequest(BaseModel):
    user_id: int
    pdf_path: str
    pdf_id: str


class SearchRequest(BaseModel):
    user_id: int
    query: str
    k: int = 5

class RegisterRequest(BaseModel):
    user_id : int 