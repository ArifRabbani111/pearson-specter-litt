from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    chroma_db_path: str = "./db/chroma"
    sqlite_db_path: str = "./db/edits.db"

    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k_retrieve: int = 10
    top_k_rerank: int = 4

    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "models/embedding-001"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    groq_model: str = "llama-3.1-70b-versatile"

    class Config:
        env_file = ".env"

settings = Settings()