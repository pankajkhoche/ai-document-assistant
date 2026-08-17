"""Environment-driven app configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "AI Document Assistant"
    debug: bool = False

    # LLM
    llm_api_key: str = ""
    llm_model: str = ""
    max_tokens: int = 1024

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4

    # Storage
    upload_dir: str = "data/uploads"
    vectorstore_dir: str = "data/vectorstore"


settings = Settings()
