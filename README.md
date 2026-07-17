# RAG Engine

Upload your documents, chat with them, and get answers with citations — built with FastAPI, LangChain, and FAISS, with a Next.js dashboard. Runs with one `docker compose up`.

## Features

- **Multi-format document ingestion**: PDF, TXT, DOCX, Markdown, HTML, CSV, JSON
- **Local-first retrieval stack**: FAISS vector store + BAAI/bge-small-en-v1.5 embeddings (no external services, no embedding API costs)
- **LLM providers**: OpenAI, Anthropic (Claude), OpenRouter (one key for many models, free tier available), Ollama (fully local)
- **Retrieval**: similarity search and hybrid search (dense + BM25 with reciprocal rank fusion), optional cross-encoder reranking (BAAI/bge-reranker-base)
- **Query pipeline**: preprocessing (normalization, metadata-filter extraction), query-type classification, retrieval confidence scoring, answer validation
- **Conversational chat**: session memory, follow-up condensing, streaming responses (SSE)
- **Answer citations**: every answer includes source documents with page numbers
- **Auth & roles**: JWT authentication with admin/user roles (single-tenant: all users share one document index)
- **Operations**: structured logging, rate limiting, security headers, health checks, per-provider cost tracking, background job queue

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Next.js Frontend (dashboard)            │
│      chat · documents · settings · users             │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│                    FastAPI (API)                     │
│  /auth  /upload  /ingest  /query  /chat  /settings   │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│              RAG Service (Orchestration)             │
│  Preprocess → Classify → Retrieve (similarity ∥      │
│  hybrid) → Merge → Confidence → Rerank → Context     │
│  → Generate → Validate → Respond with citations      │
└───┬───────────────────┬──────────────────────────────┘
    │                   │
┌───▼────────┐   ┌──────▼──────────────────────┐
│ LLM Factory │   │  FAISS + BM25 (hybrid)      │
│ openai      │   │  bge-small embeddings       │
│ anthropic   │   │  persisted to ./data        │
│ openrouter  │   └─────────────────────────────┘
│ ollama      │
└─────────────┘

Ingestion: Load → Clean → Chunk (recursive) → Embed → Store
```

## Project Structure

```
rag/
├── app/
│   ├── api/                 # FastAPI routers: auth, chat history, settings, RAG routes
│   ├── config/              # Pydantic settings, constants, prompt templates
│   ├── embeddings/          # Embedding factory (bge-small) + index versioning
│   ├── ingestion/           # Loaders, recursive chunking, ingestion pipeline
│   ├── jobs/                # In-process background job queue
│   ├── llm/                 # LLM provider factory (openai/anthropic/openrouter/ollama)
│   ├── memory/              # Chat session store
│   ├── middleware/          # Rate limiting, security headers, request tracing
│   ├── models/              # Pydantic schemas + SQLAlchemy models (users, settings)
│   ├── retriever/           # Pipeline: preprocessing, classifier, hybrid/BM25,
│   │                        #   confidence, reranker, context builder, validator,
│   │                        #   evaluation metrics (Recall@k, MRR, nDCG)
│   ├── services/            # RAG orchestration, auth service, DI
│   ├── utils/               # Logging, audit, cache, helpers
│   ├── vectorstore/         # FAISS store (persistence, filtering, hybrid)
│   └── main.py              # FastAPI app entry point
├── frontend/                # Next.js dashboard (chat, documents, settings, users)
├── tests/                   # Pytest suite (chunking, ingestion, retriever, api, …)
├── scripts/seed_data.py     # Sample data ingestion
├── data/                    # Runtime data (index, uploads, SQLite) — not committed
├── docker-compose.yml       # api + frontend + nginx
└── .env.example
```

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env         # add your API key (or set LLM_PROVIDER=ollama)
docker compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:3000
```

### Local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/seed_data.py    # optional sample data
uvicorn app.main:app --reload --port 8000
```

**No API key?** Set `LLM_PROVIDER=ollama` (with [Ollama](https://ollama.com) running locally) or use an OpenRouter free-tier model.

## API Reference

Base URL: `http://localhost:8000/api/v1` — interactive docs at `/docs`.

```bash
# Upload
curl -X POST http://localhost:8000/api/v1/upload -F "file=@document.pdf"

# Ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"filenames": ["document.pdf"], "chunk_size": 512, "chunk_overlap": 128}'

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval-Augmented Generation?", "top_k": 5}'

# Chat (session memory; add "stream": true for SSE)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG in simple terms", "session_id": "user-123"}'

# Documents
curl http://localhost:8000/api/v1/documents
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}

# Health & stats
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/stats
```

## Configuration

All configuration is via environment variables or `.env` (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `openrouter`, `ollama` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` | – | API key for the chosen provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name (per-provider `*_MODEL` vars) |
| `EMBEDDING_MODEL` | `bge-small` | BAAI/bge-small-en-v1.5, local |
| `VECTOR_STORE` | `faiss` | FAISS, persisted to `FAISS_INDEX_DIR` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `128` | Recursive chunking parameters |
| `ENABLE_HYBRID_SEARCH` | `true` | Dense + BM25 hybrid retrieval |
| `ENABLE_RERANKING` | `false` | Cross-encoder reranking |
| `ENABLE_MEMORY` | `true` | Chat session memory |
| `ENABLE_AUTH` | `false` | JWT auth (requires non-default `SECRET_KEY`) |

Cost estimates use per-provider pricing (`*_COST_PER_1K_*_TOKENS`); Ollama is always $0.

## Running Tests

```bash
pytest                    # full suite
pytest -m "not slow"      # skip model-download tests
pytest --cov=app          # with coverage
```

Tests also run automatically in CI (GitHub Actions) on every push and pull request.

> macOS note: if you hit a `Fatal Python error: Aborted` from a faiss/torch
> OpenMP conflict, run tests with `KMP_DUPLICATE_LIB_OK=TRUE pytest`.

## Design Decisions

- **Deliberately narrow scope**: one vector store (FAISS), one embedding model (bge-small), one chunking strategy (recursive), two retrieval modes (similarity, hybrid). Every configurable path is tested; breadth can be reintroduced behind the existing factory interfaces when needed.
- **LLM flexibility is kept** because it is cheap: every provider is ~15 lines behind LangChain's `BaseChatModel`, and OpenRouter covers most other model families.
- **Single-tenant by design**: authentication controls access, but all users share one document index.
- **Factory pattern + Pydantic settings**: swapping any backend is one env-var change with startup-time validation.
- **Local-first**: embeddings, BM25, reranking, and the vector index all run in-process — the only external call is the LLM.

## License

MIT
