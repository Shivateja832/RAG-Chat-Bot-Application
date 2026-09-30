from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_keep_alive: str
    ollama_num_predict: int
    context_max_chars: int
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    retrieval_top_k: int
    retrieval_min_score: float
    chunk_size: int
    chunk_overlap: int
    ebook_pdf_url: str
    ebook_pdf_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/"),
            ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b").strip(),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip(),
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip(),
            ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "180")),
            context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "1600")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook-nomic").strip(),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "agentic-ai-ebook").strip(),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "140")),
            ebook_pdf_url=os.getenv(
                "EBOOK_PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
            ).strip(),
            ebook_pdf_path=Path(os.getenv("EBOOK_PDF_PATH", str(ROOT_DIR / "data" / "Agentic-AI.pdf"))),
        )

    def validate(self) -> None:
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP or HTTPS URL.")
        if not self.ollama_chat_model or not self.ollama_embedding_model:
            raise ValueError("Configure both Ollama model names.")
        if self.ollama_num_predict < 32:
            raise ValueError("OLLAMA_NUM_PREDICT must be at least 32.")
        if self.context_max_chars < 300:
            raise ValueError("CONTEXT_MAX_CHARS must be at least 300.")
        if not self.pinecone_api_key:
            raise ValueError("Set PINECONE_API_KEY in .env to use Pinecone.")
        if not self.pinecone_index_name or not self.pinecone_namespace:
            raise ValueError("PINECONE_INDEX_NAME and PINECONE_NAMESPACE must not be empty.")
        if not self.pinecone_cloud or not self.pinecone_region:
            raise ValueError("Set a Pinecone serverless cloud and region.")
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be positive.")
        if not 0 <= self.retrieval_min_score < 1:
            raise ValueError("RETRIEVAL_MIN_SCORE must be between 0 and 1.")
        if self.chunk_size < 100 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
