from pydantic import BaseModel

class RegisterRequest(BaseModel):
    username : str

class ChatRequest(BaseModel):
    user_id : int
    query : str
    k : int =5