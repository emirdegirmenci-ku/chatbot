import json
import time

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionCreateResponse,
    SessionResetRequest,
    Source,
)
from app.services import ChatResult, get_rag_service


app = FastAPI(title="Türkçe RAG Chatbot API")


async def _ensure_session_id(req: ChatRequest, rag_service) -> str:
    if req.session_id:
        return req.session_id
    return await rag_service.create_session_id()


async def _handle_chat_completion(req: ChatRequest) -> ChatResponse:
    rag_service = await get_rag_service()
    session_id = await _ensure_session_id(req, rag_service)
    start_time = time.perf_counter()
    result = await rag_service.chat(session_id=session_id, message=req.message)
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return _build_chat_response(result, latency_ms)


def _build_chat_response(result: ChatResult, latency_ms: int) -> ChatResponse:
    sources = [
        Source(source=document.source, page=document.page, score=document.score)
        for document in result.documents
    ]
    return ChatResponse(
        session_id=result.session_id,
        response=result.response,
        sources=sources,
        used_reranking=result.used_reranking,
        topic_similarity=result.topic_similarity,
        latency_ms=latency_ms,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream özelliği için /api/chat/stream kullanın")
    return await _handle_chat_completion(req)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    rag_service = await get_rag_service()
    session_id = await _ensure_session_id(req, rag_service)
    start_time = time.perf_counter()
    stream = rag_service.stream_chat(session_id=session_id, message=req.message, start_time=start_time)
    return StreamingResponse(stream, media_type="application/json")


@app.post("/api/session/create", response_model=SessionCreateResponse)
async def session_create() -> SessionCreateResponse:
    rag_service = await get_rag_service()
    session_id = await rag_service.create_session_id()
    return SessionCreateResponse(session_id=session_id)


@app.post("/api/session/reset")
async def session_reset(req: SessionResetRequest) -> Response:
    rag_service = await get_rag_service()
    await rag_service.reset_session(req.session_id)
    payload = json.dumps({"status": "reset", "session_id": req.session_id})
    return Response(content=payload, media_type="application/json")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rag_service = await get_rag_service()
    result = await rag_service.health()
    return HealthResponse(**result)


@app.get("/api/config")
async def config() -> dict:
    settings = get_settings()
    return {
        "vector_store_path": settings.vector_store_path,
        "reranker_enabled": settings.enable_reranker,
        "top_k_initial": settings.top_k_initial,
        "top_k_context": settings.top_k_context,
    }

