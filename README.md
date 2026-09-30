# Fieldnotes: Agentic AI eBook RAG

A custom Python retrieval-augmented chatbot for the [Konverge Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). LangGraph coordinates retrieval, context grading, grounded generation, and refusal. Ollama runs generation and embeddings locally; Pinecone stores and searches the vectors. OpenAI is not used.

## Requirements

- Windows 10/11 (Ollama is also available on macOS and Linux)
- Python 3.11+
- Ollama installed and running
- About 3 GB free disk space for the default chat and embedding models; more RAM improves response speed
- A Pinecone account/API key. Free serverless availability depends on Pinecone's current plan and region.
- Internet access once to download Ollama models and the ebook PDF

## Setup

1. Install Ollama from [ollama.com/download](https://ollama.com/download), then open the Ollama app. The local API should be available at `http://127.0.0.1:11434`.
2. In PowerShell, download the models used by this project:

```powershell
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama list
```

`llama3.2:1b` generates grounded answers; `nomic-embed-text` creates document and query embeddings. `llama3.2:3b` is an optional, slower chat-model alternative. Ollama serves these locally after download; model use does not call OpenAI. Pinecone receives the vectors and passage metadata.

3. Create a Pinecone API key in the Pinecone console. It is used only for vector storage/search; Ollama inference remains local.

4. From the project root, create/activate the Python environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` if it does not exist, then set `PINECONE_API_KEY`. Confirm the Pinecone cloud and region are available to your account. No OpenAI key or OpenAI billing is required.

## Run

Make sure Ollama is open and `PINECONE_API_KEY` is set in `.env`, then run from the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Wait for the models and Pinecone to be ready, then click **Build / refresh index**. This downloads the ebook if it is not cached, extracts page-preserving chunks, embeds them with Ollama, creates a compatible Pinecone serverless index if needed, and stores vectors in the configured namespace. When indexing finishes, the chat box and suggested prompts unlock. API documentation is at <http://127.0.0.1:8000/docs>.

The first indexing run downloads 127 document chunks based on the current source PDF and model. Local inference speed depends on your computer. If Ollama reports that a model is unavailable, run the corresponding `ollama pull <model-name>` command, then refresh the browser.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama service |
| `OLLAMA_CHAT_MODEL` | `llama3.2:1b` | Faster local grounded answer generation; `llama3.2:3b` is an optional slower alternative |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Local document/query embeddings |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep models loaded to reduce repeat-request startup delay |
| `OLLAMA_NUM_PREDICT` | `180` | Maximum generated tokens per response |
| `CONTEXT_MAX_CHARS` | `1600` | Bound retrieved text sent to the local model |
| `PINECONE_API_KEY` | — | Pinecone vector database authentication |
| `PINECONE_INDEX_NAME` | `agentic-ai-ebook-nomic` | Pinecone index, created at the embedding model's dimension |
| `PINECONE_NAMESPACE` | `agentic-ai-ebook` | Namespace replaced on each full ingestion refresh |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Pinecone serverless location |
| `RETRIEVAL_TOP_K` | `5` | Maximum passages retrieved |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Minimum cosine similarity accepted |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `140` | Page-local chunk size and overlap in characters |
| `EBOOK_PDF_URL` | Konverge ebook URL | Source document |
| `EBOOK_PDF_PATH` | `data/Agentic-AI.pdf` | Local PDF cache |

Only the Pinecone API key is required. Keep `.env` private and never commit it. The downloaded PDF is ignored by Git.

## API

- `GET /api/health` — application process health.
- `GET /api/status` — Ollama availability, model readiness, and local vector count.
- `POST /api/ingest` — download, parse, embed, and replace the local collection.
- `POST /api/chat` — grounded question answering, including the query, answer, evidence chunks, and confidence score.

A question without relevant evidence receives a refusal. Source page citations are checked against pages in the retrieved passages. The confidence score is derived from retrieval similarity and is not a calibrated probability.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Tests use mocked Ollama responses and a mocked Pinecone index. No Ollama model download or Pinecone credentials are needed to run the test suite.

## Project layout

```text
app/
  config.py        local settings and paths
  ollama_client.py local model API client
  ingestion.py     PDF download, extraction, chunking, and local embeddings
  vector_store.py  Pinecone index lifecycle and top-k retrieval
  rag.py           LangGraph workflow and grounded generation
  main.py          FastAPI routes and response models
  static/          browser application
 tests/             offline workflow and vector-store tests
 data/              cached ebook PDF (not committed)
```
