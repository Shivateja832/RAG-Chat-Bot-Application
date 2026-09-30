from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT_DIR, Settings
from app.ingestion import ingest_ebook
from app.ollama_client import OllamaClient, OllamaError
from app.rag import build_rag_graph
from app.vector_store import PineconeStore

app = FastAPI(
    title="Fieldnotes | Agentic AI eBook",
    description="A document-grounded research assistant for the Agentic AI eBook.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_clients() -> tuple[OllamaClient, PineconeStore]:
    settings = get_settings()
    settings.validate()
    return OllamaClient(settings), PineconeStore(settings)


def require_services() -> tuple[Settings, OllamaClient, PineconeStore]:
    settings = get_settings()
    try:
        client, store = get_clients()
        missing = client.missing_models()
        if missing:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Download the required local models before using the ebook assistant.",
                    "missing_models": missing,
                },
            )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return settings, client, store


def provider_error_message(exc: Exception) -> str:
    if isinstance(exc, OllamaError):
        return str(exc)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "Could not reach a required service. Check that Ollama is open and Pinecone is reachable, then retry."
    return "The request could not be completed. Check that Ollama is running, its models are installed, and Pinecone settings are valid."


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(ROOT_DIR / "app" / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
async def status() -> dict[str, object]:
    settings = get_settings()
    result: dict[str, object] = {
        "state": "checking",
        "missing_models": [],
        "models": {
            "chat": settings.ollama_chat_model,
            "embedding": settings.ollama_embedding_model,
        },
        "vector_store": "Pinecone",
        "index_name": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "source": settings.ebook_pdf_url,
        "vector_count": 0,
    }
    if not settings.pinecone_api_key:
        result["state"] = "missing_pinecone_key"
        return result
    try:
        client, store = get_clients()
        result["vector_count"] = (await asyncio.to_thread(store.stats))["vector_count"]
        missing = await asyncio.to_thread(client.missing_models)
        result["missing_models"] = missing
        if missing:
            result["state"] = "missing_models"
        else:
            result["state"] = "ready" if result["vector_count"] else "needs_ingestion"
    except OllamaError as exc:
        result["state"] = "ollama_offline"
        result["error"] = str(exc)
    except Exception as exc:
        message = str(exc).lower()
        if "unauthorized" in message or "api key" in message or "authentication" in message or "401" in message:
            result["state"] = "pinecone_auth_error"
            result["error"] = "Pinecone rejected its API key. Check PINECONE_API_KEY in .env."
        else:
            result["state"] = "connection_error"
            result["error"] = "Could not connect to Pinecone. Check network access and Pinecone settings."
    return result


@app.post("/api/ingest")
async def ingest() -> dict[str, object]:
    settings, client, store = require_services()
    try:
        return await asyncio.to_thread(ingest_ebook, settings, store, client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=provider_error_message(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings, client, store = require_services()
    try:
        stats = await asyncio.to_thread(store.stats)
    except LookupError as exc:
        raise HTTPException(
            status_code=409,
            detail="The ebook has not been indexed yet. Build the index before asking questions.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=provider_error_message(exc)) from exc
    if stats["vector_count"] == 0:
        raise HTTPException(
            status_code=409,
            detail="The ebook has not been indexed yet. Build the index before asking questions.",
        )

    try:
        graph = build_rag_graph(settings, store, client)
        state = await asyncio.to_thread(
            graph.invoke,
            {
                "query": request.query.strip(),
                "history": [turn.model_dump() for turn in request.history],
            },
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=409,
            detail="The ebook has not been indexed yet. Run ingestion first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=provider_error_message(exc)) from exc

    chunks = state.get("retrieved", [])
    return ChatResponse(
        query=request.query.strip(),
        final_answer=state.get("final_answer", ""),
        retrieved_context_chunks=[
            f"[Page {item['page_number']}] {item['text']}" for item in chunks
        ],
        confidence_score=state.get("confidence_score", 0.0),
    )
