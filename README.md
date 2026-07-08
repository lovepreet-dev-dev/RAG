# RAG Engine

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and pluggable vector stores.

## Features

- **Multi-format document ingestion**: PDF, TXT, DOCX, Markdown, HTML, CSV, JSON
- **Pluggable vector stores**: FAISS (default), Chroma, Qdrant, Pinecone
- **Multiple embedding models**: BAAI/bge-small-en-v1.5, bge-large, all-MiniLM-L6-v2, OpenAI, Nomic
- **Multiple LLM providers**: OpenAI, Ollama, Anthropic, Gemini, LM Studio, OpenRouter
- **Advanced retrieval**: Similarity, MMR, Hybrid Search, Multi-Query, Context Compression
- **Optional reranking**: BAAI/bge-reranker-base cross-encoder
- **Conversational chat**: Session memory, history-aware retrieval, streaming responses
- **Answer citations**: Every answer includes source documents with page numbers
- **Incremental indexing**: Duplicate detection, batch ingestion
- **Production features**: Structured logging, rate limiting, health checks, statistics

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI (API)                     │
│  /upload  /ingest  /query  /chat  /health  /stats   │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│              RAG Service (Orchestration)             │
│    Query → Retrieve → Rerank → Generate → Respond   │
└───┬───────────────┬──────────────────┬──────────────┘
    │               │                  │
┌───▼───┐   ┌───────▼──────┐   ┌──────▼──────┐
│  LLM  │   │   Retriever  │   │  Reranker   │
│Factory│   │  (Strategies)│   │(CrossEncoder)│
└───────┘   └──────┬───────┘   └─────────────┘
                   │
          ┌────────▼────────┐
          │  Vector Store   │
          │  (FAISS/Chroma/ │
          │   Qdrant/Pine)  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │   Embeddings    │
          │ (SentenceTrans) │
          └─────────────────┘

┌─────────────────────────────────────────────────────┐
│               Ingestion Pipeline                    │
│  Load → Clean → Chunk → Embed → Store               │
│  (PDF, TXT, DOCX, MD, HTML, CSV, JSON)              │
│  (Recursive / Semantic / Sliding Window)            │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
rag/
├── app/
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   ├── config/
│   │   ├── settings.py          # Pydantic settings from .env
│   │   └── constants.py         # App-wide constants
│   ├── embeddings/
│   │   └── factory.py           # Embedding model factory
│   ├── ingestion/
│   │   ├── loaders.py           # Document loaders by file type
│   │   ├── chunking.py          # Chunking strategies
│   │   └── pipeline.py          # Ingestion orchestrator
│   ├── llm/
│   │   └── factory.py           # LLM provider factory
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── retriever/
│   │   ├── rag_retriever.py     # Multi-strategy retriever
│   │   └── reranker.py          # Cross-encoder reranker
│   ├── services/
│   │   ├── rag_service.py       # Core RAG orchestration
│   │   └── dependencies.py      # Dependency injection
│   ├── utils/
│   │   ├── logging.py           # Structured logging
│   │   └── helpers.py           # Utility functions
│   ├── vectorstore/
│   │   ├── base.py              # Abstract vector store interface
│   │   ├── faiss_store.py       # FAISS implementation
│   │   ├── chroma_store.py      # Chroma implementation
│   │   ├── qdrant_store.py      # Qdrant implementation
│   │   ├── pinecone_store.py    # Pinecone implementation
│   │   └── factory.py           # Vector store factory
│   └── main.py                  # FastAPI app entry point
├── tests/
│   ├── test_chunking/
│   ├── test_embedding/
│   ├── test_retriever/
│   ├── test_api/
│   ├── test_vectorstore/
│   └── test_ingestion/
├── scripts/
│   └── seed_data.py             # Sample data ingestion
├── data/                        # Runtime data (indexes, uploads)
├── docker/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-engine.git
cd rag-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Seed sample data (optional)
python scripts/seed_data.py

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

```bash
# Build and run
docker-compose up --build

# Or with docker directly
docker build -t rag-engine .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data rag-engine
```

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Upload Document
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@document.pdf"
```

### Ingest Documents
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "filenames": ["document.pdf"],
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 128
  }'
```

### Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Retrieval-Augmented Generation?",
    "top_k": 5,
    "retrieval_mode": "similarity"
  }'
```

### Chat
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain RAG in simple terms",
    "session_id": "user-123"
  }'
```

### List Documents
```bash
curl http://localhost:8000/api/v1/documents
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Statistics
```bash
curl http://localhost:8000/api/v1/stats
```

### Swagger UI

Visit: `http://localhost:8000/docs`

## Configuration

All configuration is via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend: openai, ollama, anthropic, gemini, lmstudio, openrouter |
| `OPENAI_API_KEY` | - | API key for OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_MODEL` | `bge-small` | Embedding model: bge-small, bge-large, minilm, openai, nomic |
| `VECTOR_STORE` | `faiss` | Vector DB: faiss, chroma, qdrant, pinecone |
| `CHUNKING_STRATEGY` | `recursive` | Strategy: recursive, semantic, sliding_window |
| `CHUNK_SIZE` | `512` | Max chunk size in characters |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |
| `TOP_K` | `5` | Number of retrieved documents |
| `ENABLE_MMR` | `true` | Enable Maximal Marginal Relevance |
| `ENABLE_RERANKING` | `false` | Enable cross-encoder reranking |
| `ENABLE_MEMORY` | `true` | Enable chat session memory |
| `LOG_LEVEL` | `INFO` | Logging level |

## How RAG Works

1. **Ingest**: Documents are loaded, text is extracted, cleaned, and split into semantic chunks.
2. **Embed**: Each chunk is converted to a vector embedding using a sentence-transformer model.
3. **Store**: Vectors are indexed in a vector database (FAISS by default).
4. **Retrieve**: When a query arrives, it's embedded and the most similar chunks are retrieved.
5. **Rerank** (optional): Retrieved chunks are re-scored by a cross-encoder for precision.
6. **Generate**: The top chunks are injected as context into an LLM prompt to generate an answer.
7. **Cite**: Sources (filename, page, chunk) are returned alongside the answer.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test suite
pytest tests/test_chunking/
pytest tests/test_api/
pytest tests/test_vectorstore/

# Skip slow tests (model downloads)
pytest -m "not slow"
```

## Design Decisions

- **Factory pattern** for vector stores and embeddings allows swapping backends with one config change.
- **Pydantic Settings** provides type-safe, validated configuration from `.env`.
- **Abstract base class** for vector stores ensures all backends implement the same interface.
- **Lazy initialization** of heavy components (embeddings, LLM) avoids startup overhead.
- **Dependency injection** via FastAPI's system keeps the service layer testable.
- **Structured logging** with `structlog` enables machine-readable log analysis.

## License

MIT
