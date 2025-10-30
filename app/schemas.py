from typing import Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    stream: bool = False
    metadata: Optional[Dict[str, str]] = None


class Source(BaseModel):
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[Source]
    used_reranking: bool
    topic_similarity: float
    latency_ms: int


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionResetRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    llm: str

