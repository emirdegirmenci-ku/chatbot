import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        # Azure Chat
        self.azure_chat_endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT", "")
        self.azure_chat_key = os.getenv("AZURE_OPENAI_CHAT_KEY", "")
        self.azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
        # Azure Embedding
        self.azure_embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT", "")
        self.azure_embed_key = os.getenv("AZURE_OPENAI_EMBED_KEY", "")
        self.azure_embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
        # RAG params
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        self.top_k_initial = int(os.getenv("TOP_K_INITIAL", "10"))
        self.top_k_context = int(os.getenv("TOP_K_CONTEXT", "3"))
        self.topic_similarity_threshold = float(os.getenv("TOPIC_SIM_THRESHOLD", "0.8"))
        self.reranker_model = os.getenv("RERANKER_MODEL", "seroe/bge-reranker-v2-m3-turkish-triplet")
        self.session_backend = os.getenv("SESSION_BACKEND", "sqlite")
        self.enable_reranker = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
        self.response_timeout = int(os.getenv("RESPONSE_TIMEOUT_S", "12"))
        # Vector DB
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.conversation_log_dir = os.getenv("CONVERSATION_LOG_DIR", "./logs/conversations")
        self.conversation_timezone = os.getenv("CONVERSATION_TIMEZONE", "Europe/Istanbul")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

