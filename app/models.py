from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    query: str

class SearchRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5