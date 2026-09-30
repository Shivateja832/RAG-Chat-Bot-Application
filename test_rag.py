from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import app
from app.rag import REFUSAL, build_rag_graph, retrieval_confidence
from app.vector_store import PineconeStore


def make_settings() -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="llama3.2:3b",
        ollama_embedding_model="nomic-embed-text",
        ollama_keep_alive="30m",
        ollama_num_predict=180,
        context_max_chars=1600,
        pinecone_api_key="test-key",
        pinecone_index_name="test-ebook-nomic",
        pinecone_namespace="test-namespace",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        retrieval_top_k=3,
        retrieval_min_score=0.25,
        chunk_size=500,
        chunk_overlap=50,
        ebook_pdf_url="https://example.test/book.pdf",
        ebook_pdf_path=Path("book.pdf"),
    )


class FakeStore:
    def __init__(self, matches):
        self.matches = matches

    def search(self, _vector, _top_k):
        return self.matches


class FakeOllama:
    def __init__(self, content):
        self.content = content

    def embed(self, _text):
        return [1.0, 0.0]

    def chat_json(self, _messages):
        return self.content


def test_out_of_scope_query_refuses_without_calling_generation():
    graph = build_rag_graph(make_settings(), FakeStore([]), FakeOllama("{}"))

    result = graph.invoke({"query": "What is the capital of France?", "history": []})

    assert result["final_answer"] == REFUSAL
    assert result["confidence_score"] == 0.0
    assert result["retrieved"] == []


def test_supported_answer_cites_only_retrieved_pages():
    store = FakeStore([
        {"text": "Agentic systems can plan tasks and use tools to act toward a goal.", "page_number": 5, "source": "book.pdf", "score": 0.68},
        {"text": "A paragraph with weak relevance.", "page_number": 9, "source": "book.pdf", "score": 0.1},
    ])
    client = FakeOllama('{"answer":"They plan and act toward a goal.","cited_pages":[5,999]}')
    graph = build_rag_graph(make_settings(), store, client)

    result = graph.invoke({"query": "What can agentic systems do?", "history": []})

    assert result["final_answer"] == "They plan and act toward a goal. (Source: p. 5)"
    assert result["confidence_score"] == retrieval_confidence(0.68)
    assert len(result["retrieved"]) == 1


def test_generation_without_valid_citation_fails_closed():
    store = FakeStore([
        {"text": "Agentic systems can plan tasks.", "page_number": 5, "source": "book.pdf", "score": 0.7}
    ])
    graph = build_rag_graph(
        make_settings(),
        store,
        FakeOllama('{"answer":"Maybe.","cited_pages":[999]}'),
    )

    result = graph.invoke({"query": "Explain agents", "history": []})

    assert result["final_answer"] == REFUSAL
    assert result["confidence_score"] == 0.0
    assert result["retrieved"] == []


def test_retrieval_confidence_is_bounded():
    assert retrieval_confidence(0.0) == 0.0
    assert retrieval_confidence(1.0) == 0.97


def test_pinecone_store_creates_dimensioned_index_and_upserts(monkeypatch):
    class FakeIndex:
        def __init__(self):
            self.upserted = []

        def delete(self, **_kwargs):
            raise RuntimeError("Namespace not found")

        def upsert(self, vectors, **_kwargs):
            self.upserted.extend(vectors)

    class FakePinecone:
        def __init__(self):
            self.index = FakeIndex()
            self.created = None

        def list_indexes(self):
            return SimpleNamespace(names=lambda: ["test-ebook-nomic"] if self.created else [])

        def create_index(self, **kwargs):
            self.created = kwargs

        def describe_index(self, _name):
            return SimpleNamespace(dimension=self.created["dimension"], status={"ready": True})

        def Index(self, _name):
            return self.index

    client = FakePinecone()
    store = PineconeStore(make_settings(), client=client)
    count = store.replace_documents([
        {"id": "chunk-1", "embedding": [0.1, 0.2, 0.3], "metadata": {"text": "evidence", "page_number": 2}}
    ])

    assert count == 1
    assert client.created["dimension"] == 3
    assert client.created["metric"] == "cosine"
    assert client.index.upserted[0]["metadata"]["page_number"] == 2


def test_chat_api_refuses_before_model_call_when_index_is_empty(monkeypatch):
    class EmptyStore:
        def stats(self):
            return {"vector_count": 0}

    monkeypatch.setattr(
        main_module,
        "require_services",
        lambda: (make_settings(), object(), EmptyStore()),
    )
    monkeypatch.setattr(
        main_module,
        "build_rag_graph",
        lambda *_: (_ for _ in ()).throw(AssertionError("model graph must not run")),
    )

    response = TestClient(app).post("/api/chat", json={"query": "What is Agentic AI?"})

    assert response.status_code == 409
    assert "Build the index" in response.json()["detail"]
