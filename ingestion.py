from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings
from app.ollama_client import OllamaClient
from app.vector_store import PineconeStore, make_vector_id

MAX_PDF_BYTES = 60 * 1024 * 1024


def download_pdf(settings: Settings) -> Path:
    destination = settings.ebook_pdf_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        response = client.get(settings.ebook_pdf_url)
        response.raise_for_status()
    content = response.content
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("The source PDF is larger than the 60 MB safety limit.")
    if not content.startswith(b"%PDF"):
        raise ValueError("The configured ebook URL did not return a PDF file.")
    destination.write_bytes(content)
    return destination


def extract_chunks(pdf_path: Path, settings: Settings) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    source = settings.ebook_pdf_url
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        for item in splitter.create_documents([page_text]):
            text = item.page_content.strip()
            if len(text) < 30:
                continue
            start_index = int(item.metadata.get("start_index", 0))
            chunks.append(
                {
                    "id": make_vector_id(source, page_number, start_index, text),
                    "text": text,
                    "page_number": page_number,
                    "source": source,
                    "start_index": start_index,
                }
            )
    if not chunks:
        raise ValueError("No readable text was extracted. The PDF may be image-only or empty.")
    return chunks


def embed_chunks(chunks: list[dict[str, Any]], client: OllamaClient) -> list[dict[str, Any]]:
    vectors = []
    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        result = client.embed_many([item["text"] for item in batch])
        for item, embedding in zip(batch, result, strict=True):
            vectors.append(
                {
                    "id": item["id"],
                    "embedding": embedding,
                    "text": item["text"],
                    "metadata": {
                        "text": item["text"],
                        "page_number": item["page_number"],
                        "source": item["source"],
                        "start_index": item["start_index"],
                    },
                }
            )
    return vectors


def ingest_ebook(settings: Settings, store: PineconeStore, client: OllamaClient) -> dict[str, Any]:
    settings.validate()
    pdf_path = download_pdf(settings)
    chunks = extract_chunks(pdf_path, settings)
    vectors = embed_chunks(chunks, client)
    stored_count = store.replace_documents(vectors)
    return {
        "source": settings.ebook_pdf_url,
        "pages_read": len(PdfReader(str(pdf_path)).pages),
        "chunks_indexed": stored_count,
        "vector_store": "Pinecone",
        "index_name": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "embedding_model": settings.ollama_embedding_model,
    }
