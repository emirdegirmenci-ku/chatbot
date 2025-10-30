import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import CrossEncoder

from app.config import get_settings


@dataclass
class VectorDocument:
    content: str
    source: str
    page: int
    score: float


@dataclass
class SessionState:
    history: List[Dict[str, str]]
    topic_embedding: Optional[List[float]]
    last_context: Optional[str]
    last_sources: List[Dict[str, Any]]
    summary: Optional[str]

    @classmethod
    def empty(cls) -> "SessionState":
        return cls(history=[], topic_embedding=None, last_context=None, last_sources=[], summary=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "topic_embedding": self.topic_embedding,
            "last_context": self.last_context,
            "last_sources": self.last_sources,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionState":
        if not data:
            return cls.empty()
        history = data.get("history") or []
        topic_embedding = data.get("topic_embedding")
        last_context = data.get("last_context")
        last_sources = data.get("last_sources") or []
        summary = data.get("summary")
        return cls(
            history=history,
            topic_embedding=topic_embedding,
            last_context=last_context,
            last_sources=last_sources,
            summary=summary,
        )


class AsyncSessionStore:
    async def load(self, session_id: str) -> SessionState:
        raise NotImplementedError

    async def save(self, session_id: str, state: SessionState) -> None:
        raise NotImplementedError

    async def reset(self, session_id: str) -> None:
        raise NotImplementedError


class SQLiteSessionStore(AsyncSessionStore):
    def __init__(self, dsn: str) -> None:
        import aiosqlite

        self._dsn = dsn
        self._aiosqlite = aiosqlite
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self._aiosqlite.connect(self._dsn) as db:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                await db.commit()
            self._initialized = True

    async def load(self, session_id: str) -> SessionState:
        await self._ensure_initialized()
        async with self._aiosqlite.connect(self._dsn) as db:
            db.row_factory = lambda cursor, row: row[0]
            async with db.execute("SELECT payload FROM sessions WHERE session_id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
        if not row:
            return SessionState.empty()
        data = json.loads(row)
        return SessionState.from_dict(data)

    async def save(self, session_id: str, state: SessionState) -> None:
        await self._ensure_initialized()
        payload = json.dumps(state.to_dict())
        async with self._aiosqlite.connect(self._dsn) as db:
            await db.execute(
                "INSERT INTO sessions(session_id, payload) VALUES(?, ?) ON CONFLICT(session_id) DO UPDATE SET payload = excluded.payload",
                (session_id, payload),
            )
            await db.commit()

    async def reset(self, session_id: str) -> None:
        await self._ensure_initialized()
        async with self._aiosqlite.connect(self._dsn) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()


class JSONSessionStore(AsyncSessionStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def _read_all(self) -> Dict[str, Any]:
        if not os.path.exists(self._path):
            return {}
        return await asyncio.to_thread(self._read_file)

    def _read_file(self) -> Dict[str, Any]:
        with open(self._path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    async def _write_all(self, data: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._write_file, data)

    def _write_file(self, data: Dict[str, Any]) -> None:
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    async def load(self, session_id: str) -> SessionState:
        async with self._lock:
            data = await self._read_all()
            payload = data.get(session_id)
        return SessionState.from_dict(payload)

    async def save(self, session_id: str, state: SessionState) -> None:
        async with self._lock:
            data = await self._read_all()
            data[session_id] = state.to_dict()
            await self._write_all(data)

    async def reset(self, session_id: str) -> None:
        async with self._lock:
            data = await self._read_all()
            if session_id in data:
                data.pop(session_id)
                await self._write_all(data)


class EmbeddingProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> List[float]:
        response = await self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._lock = asyncio.Lock()

    async def ensure_collection(self, vector_size: int) -> None:
        async with self._lock:
            collections = await self._client.get_collections()
            names = {collection.name for collection in collections.collections}
            if self._collection_name in names:
                return
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    async def collection_exists(self) -> bool:
        collections = await self._client.get_collections()
        names = {collection.name for collection in collections.collections}
        return self._collection_name in names

    async def query(self, embedding: Sequence[float], top_k: int) -> List[VectorDocument]:
        search_result = await self._client.search(
            collection_name=self._collection_name,
            query_vector=list(embedding),
            limit=top_k,
            with_payload=True,
        )
        documents: List[VectorDocument] = []
        for point in search_result:
            payload = point.payload or {}
            source = str(payload.get("source") or "")
            page = int(payload.get("page") or 0)
            content = str(payload.get("content") or "")
            score = float(point.score or 0.0)
            documents.append(VectorDocument(content=content, source=source, page=page, score=score))
        return documents


class Reranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> CrossEncoder:
        if self._model:
            return self._model
        async with self._lock:
            if self._model:
                return self._model
            model = await asyncio.to_thread(lambda: CrossEncoder(self._model_name))
            self._model = model
        return self._model

    async def rerank(self, query: str, documents: List[VectorDocument], top_n: int) -> List[VectorDocument]:
        if not documents:
            return []
        model = await self._get_model()
        pairs = [(query, document.content) for document in documents]
        scores = await asyncio.to_thread(model.predict, pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        limited = scored[:top_n]
        reranked = []
        for document, score in limited:
            reranked.append(VectorDocument(content=document.content, source=document.source, page=document.page, score=float(score)))
        return reranked


class LLMClient:
    def __init__(self, client: AsyncOpenAI, model: str, timeout: int) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        async def _call() -> str:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            return content

        return await asyncio.wait_for(_call(), timeout=self._timeout)

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        async for piece in stream:
            if piece.choices and piece.choices[0].delta and piece.choices[0].delta.content:
                yield piece.choices[0].delta.content


@dataclass
class ContextSelection:
    session_state: SessionState
    query_embedding: List[float]
    context_text: str
    documents: List[VectorDocument]
    used_reranking: bool
    topic_similarity: float
    rewritten_query: str


@dataclass
class ChatResult:
    session_id: str
    response: str
    documents: List[VectorDocument]
    used_reranking: bool
    topic_similarity: float


class RAGService:
    def __init__(
        self,
        session_store: AsyncSessionStore,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Optional[Reranker],
        llm: LLMClient,
    ) -> None:
        self._session_store = session_store
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._llm = llm
        self._settings = get_settings()
        self._timezone = self._resolve_timezone(self._settings.conversation_timezone)

    async def chat(self, session_id: str, message: str) -> ChatResult:
        prepared = await self._prepare(session_id, message)
        messages = self._build_messages(prepared.session_state.history, prepared.context_text, message)
        response = await self._llm.generate(messages)
        await self._finalize(session_id, prepared, message, response)
        return ChatResult(
            session_id=session_id,
            response=response,
            documents=prepared.documents,
            used_reranking=prepared.used_reranking,
            topic_similarity=prepared.topic_similarity,
        )

    async def stream_chat(self, session_id: str, message: str, start_time: float) -> AsyncIterator[str]:
        prepared = await self._prepare(session_id, message)
        messages = self._build_messages(prepared.session_state.history, prepared.context_text, message)
        chunks: List[str] = []
        async for part in self._llm.stream(messages):
            chunks.append(part)
            yield json.dumps({"type": "token", "data": part}) + "\n"
        response = "".join(chunks)
        await self._finalize(session_id, prepared, message, response)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        payload = {
            "type": "summary",
            "session_id": session_id,
            "response": response,
            "sources": [
                {
                    "source": document.source,
                    "page": document.page,
                    "score": document.score,
                }
                for document in prepared.documents
            ],
            "used_reranking": prepared.used_reranking,
            "topic_similarity": prepared.topic_similarity,
            "latency_ms": latency_ms,
        }
        yield json.dumps(payload) + "\n"

    async def create_session_id(self) -> str:
        return str(uuid.uuid4())

    async def reset_session(self, session_id: str) -> None:
        await self._session_store.reset(session_id)

    async def health(self) -> Dict[str, str]:
        settings = get_settings()
        status = "ok" if (settings.azure_chat_key and settings.azure_chat_endpoint) else "degraded"
        vector_status = "ready"
        llm_status = "ready" if (settings.azure_chat_key and settings.azure_chat_deployment) else "missing_key"
        try:
            vector_status = "ready" if await self._vector_store.collection_exists() else "missing"
        except Exception:
            vector_status = "error"
        return {"status": status, "vector_store": vector_status, "llm": llm_status}

    async def _prepare(self, session_id: str, message: str) -> ContextSelection:
        session_state = await self._session_store.load(session_id)
        rewritten_query = await self._rewrite_query(
            session_state.history, session_state.summary, message
        )
        query_embedding = await self._embeddings.embed(rewritten_query)
        topic_similarity = 0.0
        documents: List[VectorDocument] = []
        used_reranking = False
        context_text = ""
        if session_state.topic_embedding and session_state.last_context:
            topic_similarity = self._cosine_similarity(query_embedding, session_state.topic_embedding)
        if topic_similarity >= self._settings.topic_similarity_threshold and session_state.last_context:
            context_text = session_state.last_context
            documents = [
                VectorDocument(
                    content=doc.get("content") or "",
                    source=str(doc.get("source") or ""),
                    page=int(doc.get("page") or 0),
                    score=float(doc.get("score") or 0.0),
                )
                for doc in session_state.last_sources
            ]
        else:
            await self._vector_store.ensure_collection(vector_size=len(query_embedding))
            retrieved = await self._vector_store.query(query_embedding, self._settings.top_k_initial)
            if self._settings.enable_reranker and self._reranker:
                documents = await self._reranker.rerank(message, retrieved, self._settings.top_k_context)
                used_reranking = True
            else:
                documents = retrieved[: self._settings.top_k_context]
            context_text = self._join_context(documents)
            topic_similarity = topic_similarity if session_state.topic_embedding else 0.0
        return ContextSelection(
            session_state=session_state,
            query_embedding=query_embedding,
            context_text=context_text,
            documents=documents,
            used_reranking=used_reranking,
            topic_similarity=topic_similarity,
            rewritten_query=rewritten_query,
        )

    async def _finalize(
        self,
        session_id: str,
        prepared: ContextSelection,
        message: str,
        response: str,
    ) -> None:
        history = prepared.session_state.history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        summary = prepared.session_state.summary
        if len(history) > 20:
            excess = history[:-20]
            history = history[-20:]
            new_summary = await self._summarize_history(excess)
            summary = self._merge_summaries(summary, new_summary)
        stored_sources = [
            {
                "content": document.content,
                "source": document.source,
                "page": document.page,
                "score": document.score,
            }
            for document in prepared.documents
        ]
        session_state = SessionState(
            history=history,
            topic_embedding=prepared.query_embedding,
            last_context=prepared.context_text,
            last_sources=stored_sources,
            summary=summary,
        )
        await self._session_store.save(session_id, session_state)
        await self._append_conversation_log(
            session_id,
            message,
            response,
            stored_sources,
            prepared.rewritten_query,
        )

    def _join_context(self, documents: List[VectorDocument]) -> str:
        contents = [document.content for document in documents if document.content]
        return "\n\n".join(contents)

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        vec_a = np.array(a)
        vec_b = np.array(b)
        denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denominator)

    def _build_messages(self, history: List[Dict[str, str]], context_text: str, message: str) -> List[Dict[str, str]]:
        system_prompt = (
"""
Adı: KU Hub Assistant.
Görev: Kullanıcı sorularını yalnızca sağlanan belgelerdeki bilgilere dayanarak yanıtla ve kullanıcılarla dostane bir şekilde iletişim kur.

Yalnızca sağlanan PDF veya belge bağlamındaki doğrulanabilir bilgilere dayanarak yanıt ver.
Kısa, net, insani ve yardımsever bir üslup kullan; gereksiz ayrıntıya girme.
Uzun paragrafları kopyalama, gerekli bilgiyi öz olarak ver.
Gerekli olduğunda PDF adı ve sayfa numarasını belirt (örnek: 'PDF s.4').
Varsayım yapma, dış kaynak kullanma.

Kullanıcı belirli bir öğe, koşul veya banka sorarsa bilgiyi yorumlayıp doğrudan EVET/HAYIR + tek cümlelik kısa gerekçe ile yanıt ver.
Tüm listeyi yalnızca açıkça istenirse ve spesifik bilgi gerekiyorsa ver.
Bağlamda bilgi yoksa 'Üzgünüm, bu bilgi mevcut değil.' de.

Kapsam dışı (üniversiteyle ilgisiz) kompleks sorularda kibarca kapsamını belirt. Basit, kişisel sohbetlere (adın ne, nasılsın gibi) katılabilirsin.
Eğer kullanıcı adını söylerse, ona adıyla hitap etmeye çalış.

Kullanıcı ifadesi belirsizse veya hatalıysa, anlamını yorumla ve kısa bir netleştirme sorusu sor.
Talep edilirse bağlantı veya işlem adımlarını açık ve adım adım aktar.
Gerekirse basit hesaplamaları yap.
Kendini tekrar etme, aynı bilgiyi farklı cümlelerle uzatma.

Kullanıcı hangi dili kullanıyorsa, aynı dilde yanıt ver.
Günlük konuşmalara (örnek: merhaba, nasılsın vb.) katılabilirsin, ancak yanlış bilgi verme.
Belgeye dayalı yanıt ver; doğrudan olmasa bile dolaylı bilgilerden çıkarım yapabilirsin.
Belgeye erişimin olduğunu ima etme.
Her zaman kibar, nazik ve yardımsever ol.
Kullanıcının önceki sorularını ve bağlamı hatırla, tutarlı yanıt ver.
Yanıtlarını kısa, açık ve anlaşılır tut; gereksiz uzatma yapma.
Belgeden yanıt verilemiyorsa kullanıcıdan sorusunu detaylandırmasını iste.
Eğer kullanıcı sadece sohbet ediyorsa belgeye bakmadan genel bilgiyle yanıt ver; ancak belgeyle ilgili sorularda belgeye dayalı yanıt ver.
Belgedeki bağlantı veya yönergeleri açık, net ve adım adım sun; kullanıcıyı belgeyi incelemeye yönlendirme.
Belgeyle ilgili hesaplama istenirse yap ve sonucu belirt.
Kullanıcının yazım hatalarını görmezden gel, anlamı doğru yorumla.
"""
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if context_text:
            prompt = "Bağlam:\n" + context_text + "\n\nSoru:\n" + message
        else:
            prompt = message
        recent_history = history[-8:]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": prompt})
        # --- DEV LOGGING START ---
        print("\n" + "=" * 50)
        print("NIHAI CEVAP ICIN PROMPT OLUSTURULDU")
        print("=" * 50)
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            print(f"--- ROLE: {role} ---")
            print(content)
        print("=" * 50 + "\n")
        # --- DEV LOGGING END ---
        return messages

    async def _append_conversation_log(
        self,
        session_id: str,
        message: str,
        response: str,
        sources: List[Dict[str, Any]],
        rewritten_query: str,
    ) -> None:
        log_dir = Path(self._settings.conversation_log_dir)
        entries = [
            {
                "timestamp": self._current_time(),
                "session_id": session_id,
                "role": "user",
                "content": message,
            },
            {
                "timestamp": self._current_time(),
                "session_id": session_id,
                "role": "assistant",
                "content": response,
                "sources": sources,
            },
        ]
        if rewritten_query and rewritten_query.strip() and rewritten_query.strip() != message.strip():
            entries[0]["rewritten_query"] = rewritten_query.strip()
        await asyncio.to_thread(self._write_log_entries, log_dir, session_id, entries)

    def _write_log_entries(self, log_dir: Path, session_id: str, entries: List[Dict[str, Any]]) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{session_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")

    def _current_time(self) -> str:
        return datetime.now(self._timezone).isoformat()

    def _resolve_timezone(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except Exception:
            return ZoneInfo("UTC")

    async def _rewrite_query(
        self, history: List[Dict[str, str]], summary: Optional[str], message: str
    ) -> str:
        if not history and not summary:
            return message
        formatted_history = self._format_history_for_rewrite(history)
        if not formatted_history and not summary:
            return message
        system_prompt = (
            "Senin görevin, bir kullanıcı ve asistan arasındaki konuşma geçmişini ve uzun vadeli özeti analiz ederek, "
            "kullanıcının son mesajını tam bağlamıyla birlikte, tek başına anlaşılır bir soruya veya ifadeye dönüştürmektir.\n\n"
            "- Konuşma geçmişindeki önemli detayları (örneğin: isimler, yerler, tarihler, belirli miktarlar, daha önce bahsedilen konular) dikkate al.\n"
            "- Kullanıcının niyetini ve sorduğu sorunun kök nedenini anlamaya çalış.\n"
            "- Eğer kullanıcının son mesajı zaten tek başına anlaşılır ise, onu değiştirmeden aynen geri döndür.\n"
            "- Sadece ve sadece yeniden yapılandırılmış soruyu veya ifadeyi döndür. Kesinlikle açıklama, giriş veya yorum ekleme."
        )
        # --- DEV LOGGING START ---
        print("\n" + "=" * 50)
        print("SORGU YENIDEN YAZMA PROMPTU OLUSTURULDU")
        print("=" * 50)
        print(f"SYSTEM: {system_prompt}")
        print("-" * 50)
        if formatted_history:
            print("KISA VADELİ HAFIZA (Son 10 mesaj):")
            print(formatted_history)
        if summary:
            print("UZUN VADELİ HAFIZA (Özet):")
            print(summary)
        print(f"\nKULLANICININ SON MESAJI: {message}")
        print("=" * 50 + "\n")
        # --- DEV LOGGING END ---
        content_parts: List[str] = []
        if formatted_history:
            content_parts.append("Konuşma geçmişi:\n" + formatted_history)
        if summary:
            content_parts.append("Özet:\n" + summary)
        user_content = "\n\n".join(content_parts) + "\n\n"
        user_content += "Son kullanıcı sorusu: " + message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            rewritten = await self._llm.generate(messages)
        except Exception:
            return message
        cleaned = (rewritten or "").strip()
        if not cleaned:
            return message
        return cleaned

    def _format_history_for_rewrite(self, history: List[Dict[str, str]], max_messages: int = 6) -> str:
        relevant = history[-max_messages:]
        lines: List[str] = []
        for item in relevant:
            role = item.get("role", "")
            content = item.get("content", "")
            if not content:
                continue
            prefix = "Kullanıcı" if role == "user" else "Asistan"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    async def _summarize_history(self, history: List[Dict[str, str]]) -> Optional[str]:
        if not history:
            return None
        formatted = self._format_history_for_summary(history)
        if not formatted:
            return None
        system_prompt = (
            "Senin görevin, aşağıda verilen konuşma geçmişini analiz ederek, gelecekteki konuşmalarda bağlamı korumak için kritik bilgileri içeren profesyonel bir özet oluşturmaktır.\n\n"
            "Özeti oluştururken şu kurallara uy:\n"
            "1. **Anahtar Bilgileri Çıkar:** Kullanıcı hakkında öğrenilen kişisel bilgileri (adı, tercihleri, durumu vb.), konuşulan ana konuları, varılan sonuçları, verilen sayısal verileri (tarih, para miktarı vb.) ve önemli kararları mutlaka özetine ekle.\n"
            "2. **Yapılandırılmış Ol:** Bilgileri maddeler halinde veya kısa paragraflar şeklinde düzenle.\n"
            "3. **Kısa ve Net Ol:** Gereksiz diyalogları, selamlaşmaları ve dolgu ifadeleri çıkar. Sadece gelecekte referans olarak kullanılabilecek temel gerçekleri ve bağlamı koru.\n"
            "4. **Nötr Bir Dil Kullan:** Özeti objektif bir dille yaz.\n\n"
            "Bu özet, asistanın konuşmanın ilerleyen kısımlarında tutarlı ve bilgili kalmasını sağlayacaktır. Sadece özeti döndür."
        )
        user_prompt = "Konuşma:\n" + formatted
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            summary = await self._llm.generate(messages)
        except Exception:
            return None
        cleaned = (summary or "").strip()
        return cleaned or None

    def _merge_summaries(self, existing: Optional[str], new_summary: Optional[str]) -> Optional[str]:
        if not new_summary:
            return existing
        if not existing:
            return new_summary
        return existing + "\n" + new_summary

    def _format_history_for_summary(self, history: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            if not content:
                continue
            prefix = "Kullanıcı" if role == "user" else "Asistan"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)


_service_instance: Optional[RAGService] = None
_service_lock = asyncio.Lock()


async def _create_service() -> RAGService:
    settings = get_settings()
    session_backend = settings.session_backend.lower()
    if session_backend == "sqlite":
        session_store = SQLiteSessionStore(dsn="sessions.db")
    elif session_backend == "json":
        session_store = JSONSessionStore(path="sessions.json")
    else:
        session_store = SQLiteSessionStore(dsn="sessions.db")
    embed_client = AsyncOpenAI(api_key=settings.azure_embed_key, base_url=settings.azure_embed_endpoint or None)
    chat_client = AsyncOpenAI(api_key=settings.azure_chat_key, base_url=settings.azure_chat_endpoint or None)
    embed_model = settings.azure_embed_deployment
    chat_model = settings.azure_chat_deployment
    embeddings = EmbeddingProvider(client=embed_client, model=embed_model)
    vector_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    vector_store = VectorStore(client=vector_client, collection_name="documents")
    reranker = Reranker(model_name=settings.reranker_model) if settings.enable_reranker else None
    llm = LLMClient(client=chat_client, model=chat_model, timeout=settings.response_timeout)
    return RAGService(session_store=session_store, embeddings=embeddings, vector_store=vector_store, reranker=reranker, llm=llm)


async def get_rag_service() -> RAGService:
    global _service_instance
    if _service_instance:
        return _service_instance
    async with _service_lock:
        if _service_instance:
            return _service_instance
        _service_instance = await _create_service()
    return _service_instance

