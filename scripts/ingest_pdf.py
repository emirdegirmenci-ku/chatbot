import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings


@dataclass
class Chunk:
    content: str
    source: str
    page: int


def load_pdf_chunks(path: Path, max_tokens: int = 230) -> List[Chunk]:
    reader = PdfReader(str(path))
    chunks: List[Chunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        tokens = text.split()
        buffer: List[str] = []
        token_count = 0
        for token in tokens:
            buffer.append(token)
            token_count += 1
            if token_count >= max_tokens:
                content = " ".join(buffer).strip()
                if content:
                    chunks.append(Chunk(content=content, source=path.name, page=page_index))
                buffer = []
                token_count = 0
        if buffer:
            content = " ".join(buffer).strip()
            if content:
                chunks.append(Chunk(content=content, source=path.name, page=page_index))
    return chunks


async def embed_chunks(chunks: Sequence[Chunk], client: AsyncOpenAI, model: str, batch_size: int = 8) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for index in range(0, len(chunks), batch_size):
        batch = chunks[index : index + batch_size]
        texts = [chunk.content for chunk in batch]
        response = await client.embeddings.create(model=model, input=texts)
        for item in response.data:
            embeddings.append(list(item.embedding))
    return embeddings


async def ingest_pdf(path: Path) -> None:
    settings = get_settings()
    base_url = settings.azure_embed_endpoint or None
    api_key = settings.azure_embed_key
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    chunks = load_pdf_chunks(path)
    if not chunks:
        print(f"Boş içerik atlandı: {path}")
        return
    model_name = settings.azure_embed_deployment
    vectors = await embed_chunks(chunks, client, model_name)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    vector_size = len(vectors[0]) if vectors else 0
    if vector_size == 0:
        print("Embedding üretilemedi, yükleme atlandı.")
        return
    collections = await qdrant_client.get_collections()
    names = {collection.name for collection in collections.collections}
    if "documents" not in names:
        await qdrant_client.create_collection(
            collection_name="documents",
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload={
                    "source": chunk.source,
                    "page": chunk.page,
                    "content": chunk.content,
                },
            )
        )
    await qdrant_client.upsert(collection_name="documents", points=points)
    print(f"{path.name} için {len(points)} parça Qdrant'e eklendi.")


async def main() -> None:
    settings = get_settings()
    pdf_dir = Path("data/pdf")
    if not pdf_dir.exists():
        raise FileNotFoundError("data/pdf klasörü bulunamadı")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("PDF dosyası bulunamadı")
    for path in pdf_files:
        await ingest_pdf(path)


if __name__ == "__main__":
    asyncio.run(main())

