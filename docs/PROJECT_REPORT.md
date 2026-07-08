# RAG Engine — Project Report

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What the Project Does](#2-what-the-project-does)
3. [Architecture & How It Works](#3-architecture--how-it-works)
4. [Technologies Used & Their Purposes](#4-technologies-used--their-purposes)
5. [How to Operate It](#5-how-to-operate-it)
6. [API Reference](#6-api-reference)
7. [Configuration Options](#7-configuration-options)
8. [Deployment](#8-deployment)
9. [Project Structure](#9-project-structure)

---

## 1. Executive Summary

The **RAG Engine** is a production-ready Retrieval-Augmented Generation platform that allows users to upload documents, index them into a vector database, and query them using natural language. It combines the power of large language models (LLMs) with precise document retrieval to deliver accurate, source-cited answers.

**Key Capabilities:**
- Ingest documents in 7 formats (PDF, TXT, DOCX, Markdown, HTML, CSV, JSON)
- Store document embeddings in 4 vector backends (FAISS, Chroma, Qdrant, Pinecone)
- Query with 4 retrieval strategies (Similarity, MMR, Hybrid, Multi-Query)
- Chat with memory and streaming support
- Switch LLM providers with a single environment variable change
- Full REST API with OpenAPI documentation

---

## 2. What the Project Does

### 2.1 Document Ingestion Pipeline

```
Upload → Load → Clean → Chunk → Embed → Store
```

1. **Upload**: Users upload files via the `/api/v1/upload` endpoint. Files are validated for type and size, then saved to disk with sanitized filenames.

2. **Load**: The ingestion pipeline reads files using format-specific loaders:
   - PDF → PyPDFLoader (one Document per page)
   - TXT/Markdown → Direct text reading
   - DOCX → Docx2txtLoader
   - HTML → BSHTMLLoader (BeautifulSoup-based)
   - CSV → Row-by-row conversion to JSON Documents
   - JSON → Nested structure flattening

3. **Clean**: Text is normalized — excess whitespace stripped, control characters removed, encoding issues resolved.

4. **Chunk**: Documents are split into smaller pieces using one of three strategies:
   - **Recursive** (default): Splits on paragraph → sentence → word boundaries using `RecursiveCharacterTextSplitter`
   - **Semantic**: Groups sentences by embedding similarity using `SemanticChunker`
   - **Sliding Window**: Fixed-size windows with configurable overlap

5. **Embed**: Each chunk is converted to a vector (numerical representation) using one of 5 embedding models:
   - `bge-small` → BAAI/bge-small-en-v1.5 (384 dimensions, local)
   - `bge-large` → BAAI/bge-large-en-v1.5 (1024 dimensions, local)
   - `minilm` → all-MiniLM-L6-v2 (384 dimensions, local)
   - `nomic` → nomic-embed-text-v1.5 (768 dimensions, local)
   - `openai` → text-embedding-3-small (1536 dimensions, API)

6. **Store**: Vectors are persisted in the configured vector store with metadata (filename, page number, chunk ID, document ID).

### 2.2 Query & Retrieval

```
Question → Retrieve → (Rerank) → Format Context → Generate Answer → Return with Sources
```

When a user asks a question:

1. **Retrieve**: The question is embedded and searched against the vector store using one of 4 modes:
   - **Similarity**: Direct cosine/distance search for the k most similar chunks
   - **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity to avoid redundant results
   - **Hybrid**: Combines semantic vector search with keyword matching
   - **Multi-Query**: Uses the LLM to generate 3 alternative phrasings, searches with all, merges de-duplicated results

2. **Score Threshold**: Results below the configured similarity threshold are filtered out.

3. **Rerank** (optional): A cross-encoder model (`BAAI/bge-reranker-base`) re-scores `(query, document)` pairs for higher precision.

4. **Format Context**: Retrieved chunks are formatted into a context string with source citations.

5. **Generate**: The LLM receives the context + question and generates an answer. The system prompt enforces:
   - Answer ONLY from provided context
   - Cite sources by filename and page number
   - Say "I don't have enough information" when context is insufficient

6. **Return**: The response includes the answer, source documents with scores, confidence score, token usage, cost estimate, and latency.

### 2.3 Conversational Chat

The `/api/v1/chat` endpoint adds session memory:

- Each session maintains a conversation history (configurable window, default 10 message pairs)
- Follow-up questions are rephrased as standalone questions using `CONDENSE_QUESTION_PROMPT`
- The rephrased question is retrieved and answered like a normal query
- Streaming mode (`stream=true`) returns Server-Sent Events (SSE) for real-time token delivery

### 2.4 Confidence Scoring

After retrieval, a confidence score is computed based on the average similarity scores of retrieved documents. Scores range from 0.0 (no confidence) to 1.0 (high confidence).

### 2.5 Cost Tracking

Every query tracks:
- Token count (estimated via tiktoken)
- Cost estimate based on configured per-1k-token pricing
- Cumulative statistics across all queries

---

## 3. Architecture & How It Works

### 3.1 Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                            │
│              app/api/routes.py                           │
│         (FastAPI Router, Pydantic Validation)            │
├─────────────────────────────────────────────────────────┤
│                   Service Layer                          │
│            app/services/rag_service.py                   │
│       (Orchestration, Business Logic, Stats)             │
├──────────────┬──────────────┬───────────────────────────┤
│  Ingestion   │  Retriever   │       LLM Layer           │
│  Pipeline    │  Module      │    app/llm/factory.py     │
│              │              │   (6 Provider Adapters)    │
├──────────────┼──────────────┼───────────────────────────┤
│  Loaders     │  Strategies  │   Embeddings Factory      │
│  Chunking    │  Reranker    │  (5 Model Adapters)       │
├──────────────┴──────────────┴───────────────────────────┤
│                  Vector Store Layer                       │
│          app/vectorstore/factory.py                      │
│    (FAISS / Chroma / Qdrant / Pinecone Adapters)         │
├─────────────────────────────────────────────────────────┤
│                  Config & Utilities                       │
│     settings.py · constants.py · logging.py · helpers.py  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Factory** | `vectorstore/factory.py`, `embeddings/factory.py`, `llm/factory.py` | Create backend-specific instances from config |
| **Strategy** | `ingestion/chunking.py`, `retriever/rag_retriever.py` | Swap algorithms at runtime via env vars |
| **Adapter** | Each vector store, LLM provider, embedding model | Uniform interface over diverse implementations |
| **Repository** | `BaseVectorStore` ABC | Abstract data access behind a common interface |
| **Singleton** | `get_settings()`, `get_rag_service()` | Process-wide cached instances via `lru_cache` |
| **Dependency Injection** | `services/dependencies.py` | Lazy-loaded singletons for all components |
| **Lazy Loading** | All factories, `RAGService` properties | Heavy libraries loaded only when needed |

### 3.3 Request Flow

```
Client → FastAPI Router → Pydantic Validation → RAGService → Component
                                    ↓
                              Error Handler → Structured JSON Error Response
```

```
POST /api/v1/query
  → QueryRequest validation
  → RAGService.query()
    → RAGRetriever.retrieve()
      → VectorStore.similarity_search_with_score()
      → Score threshold filtering
      → (Optional) Reranker.rerank()
    → Context formatting
    → LLM.invoke()
    → Source building
    → Cost/token estimation
  → QueryResponse serialization
```

### 3.4 Streaming Flow

```
POST /api/v1/chat  (stream=true)
  → ChatRequest validation
  → RAGService.chat_stream()
    → Retriever.retrieve()
    → Context formatting
    → LLM.astream()  ← async iterator
      → yield StreamChunk(token=...)  ← SSE frame
      → yield StreamChunk(token=...)  ← SSE frame
      → yield StreamChunk(is_final=True, sources=...)  ← final frame
  → StreamingResponse (text/event-stream)
```

---

## 4. Technologies Used & Their Purposes

### 4.1 Core Framework

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Primary language |
| **FastAPI** | 0.115.6 | Async web framework, auto-generates OpenAPI docs |
| **Uvicorn** | 0.34.0 | ASGI server for production deployment |
| **Pydantic** | 2.10.4 | Data validation, serialization, settings management |
| **Pydantic Settings** | 2.7.1 | Environment variable loading with type coercion |

### 4.2 AI/ML Stack

| Technology | Purpose |
|-----------|---------|
| **LangChain** | Orchestration framework — chains, prompts, document loaders, output parsers |
| **LangChain Core** | Base abstractions (Document, Embeddings, BaseChatModel, VectorStore) |
| **LangChain Community** | Third-party integrations (HuggingFace, Chroma, FAISS, etc.) |
| **LangChain OpenAI** | OpenAI & OpenAI-compatible LLM/embedding wrappers |
| **LangChain Anthropic** | Anthropic Claude integration |
| **LangChain Google GenAI** | Google Gemini integration |
| **LangChain Text Splitters** | Recursive and semantic text chunking algorithms |
| **sentence-transformers** | Local embedding models (BGE, MiniLM, Nomic) and cross-encoder rerankers |
| **PyTorch** | Deep learning backend for sentence-transformers |
| **tiktoken** | OpenAI's fast tokenizer for token counting and cost estimation |

### 4.3 Vector Stores

| Technology | Type | Purpose |
|-----------|------|---------|
| **FAISS** | Local | Facebook's high-performance similarity search library. Best for single-machine, large-scale local indices. |
| **Chroma** | Local/Server | Developer-friendly vector database with built-in persistence. Good for prototyping and small-to-medium datasets. |
| **Qdrant** | Server | High-performance vector search engine with advanced filtering. Best for production workloads requiring scalability. |
| **Pinecone** | Cloud | Fully-managed cloud vector database. Best for zero-ops production deployments. |

### 4.4 Document Processing

| Technology | Purpose |
|-----------|---------|
| **PyPDF** | PDF text extraction (page-by-page) |
| **python-docx** | Microsoft Word document parsing |
| **BeautifulSoup4** | HTML content extraction and cleaning |

### 4.5 Infrastructure

| Technology | Purpose |
|-----------|---------|
| **structlog** | Structured JSON logging with correlation IDs |
| **Docker** | Containerized deployment with multi-stage builds |
| **Docker Compose** | Multi-service orchestration (API + optional databases) |
| **python-dotenv** | `.env` file loading for local development |
| **httpx** | Async HTTP client for health checks and external calls |

### 4.6 Development & Quality

| Technology | Purpose |
|-----------|---------|
| **pytest** | Test framework with async support |
| **pytest-cov** | Code coverage reporting |
| **pytest-asyncio** | Async test support |
| **Black** | Code formatting (100 char line length) |
| **Ruff** | Fast Python linter (E, F, I, N, UP, B, A, SIM rules) |
| **mypy** | Static type checking |

---

## 5. How to Operate It

### 5.1 Prerequisites

- Python 3.10 or higher
- pip or Docker
- An LLM provider API key (or local Ollama/LM Studio running)

### 5.2 Quick Start (Local)

```bash
# 1. Clone and enter the project
cd rag-engine

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# 5. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at:
- **API**: http://localhost:8000/api/v1
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 5.3 Docker Deployment

```bash
# Build and start
docker compose up --build -d

# Check logs
docker compose logs -f rag-api

# Stop
docker compose down
```

### 5.4 Basic Workflow

#### Step 1: Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@myfile.pdf"
```

Response:
```json
{
  "filename": "myfile.pdf",
  "file_size": 1048576,
  "message": "File 'myfile.pdf' uploaded successfully."
}
```

#### Step 2: Ingest the Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "filenames": ["myfile.pdf"],
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 128
  }'
```

Response:
```json
{
  "documents": [
    {
      "document_id": "a1b2c3d4",
      "filename": "myfile.pdf",
      "total_chunks": 42,
      "total_pages": 5
    }
  ],
  "total_chunks": 42,
  "message": "Successfully ingested 1 document(s) into 42 chunks."
}
```

#### Step 3: Query the System

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 5,
    "retrieval_mode": "similarity"
  }'
```

Response:
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "content": "Relevant text snippet...",
      "filename": "myfile.pdf",
      "page_number": 2,
      "chunk_id": 5,
      "score": 0.8734
    }
  ],
  "confidence_score": 0.8234,
  "model_used": "gpt-4o-mini",
  "tokens_used": 342,
  "cost_estimate": 0.000171,
  "latency_ms": 1234.56
}
```

#### Step 4: Chat with Memory

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the document",
    "session_id": "my-session-1"
  }'
```

#### Step 5: Stream a Response

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the key findings",
    "stream": true
  }'
```

Returns Server-Sent Events:
```
data: {"token": "The", "query_id": "...", "is_final": false}
data: {"token": " key", "query_id": "...", "is_final": false}
data: {"token": " findings", "query_id": "...", "is_final": false}
...
data: {"token": "", "query_id": "...", "is_final": true, "sources": [...]}
data: [DONE]
```

### 5.5 Switching Backends

The entire backend can be swapped with a single `.env` change:

```bash
# Switch from FAISS to Qdrant
VECTOR_STORE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Switch from OpenAI to Ollama (local, free)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# Switch from local embeddings to OpenAI
EMBEDDING_MODEL=openai
OPENAI_API_KEY=sk-...
```

No code changes required — the factory pattern handles all instantiation.

---

## 6. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload a document file |
| `POST` | `/api/v1/ingest` | Ingest uploaded files into vector store |
| `POST` | `/api/v1/query` | Ask a question (one-shot) |
| `POST` | `/api/v1/chat` | Conversational chat (with memory + streaming) |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document by ID |
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/stats` | Usage statistics |
| `POST` | `/api/v1/feedback` | Submit feedback on a response |

Full interactive documentation is auto-generated at `/docs` (Swagger UI).

---

## 7. Configuration Options

All settings are configured via environment variables or the `.env` file. See `.env.example` for the complete list.

| Category | Key Variables | Default |
|----------|--------------|---------|
| **LLM** | `LLM_PROVIDER`, `OPENAI_API_KEY`, `OLLAMA_MODEL` | `openai`, `gpt-4o-mini` |
| **Embeddings** | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE` | `bge-small`, `cpu` |
| **Vector Store** | `VECTOR_STORE`, `FAISS_INDEX_DIR` | `faiss`, `./data/faiss_index` |
| **Chunking** | `CHUNKING_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `recursive`, `512`, `128` |
| **Retrieval** | `TOP_K`, `SCORE_THRESHOLD`, `ENABLE_MMR` | `5`, `0.3`, `true` |
| **Server** | `HOST`, `PORT`, `LOG_LEVEL` | `0.0.0.0`, `8000`, `INFO` |
| **Uploads** | `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB` | `./data/uploads`, `50` |

---

## 8. Deployment

### 8.1 Docker (Recommended)

```bash
docker compose up --build -d
```

The `docker-compose.yml` includes:
- Multi-stage Docker build (build → production image)
- Health checks (`/api/v1/health` every 30s)
- Volume mounts for data persistence
- Optional Qdrant and Chroma services (commented out)

### 8.2 Production Checklist

Run the built-in production validator:

```python
from app.config.settings import get_settings

issues = get_settings().validate_for_production()
for issue in issues:
    print(f"⚠ {issue}")
```

Common production items:
- Set `DEBUG=false`
- Change `SECRET_KEY` from the default placeholder
- Set `ENABLE_AUTH=true` with proper JWT configuration
- Configure CORS origins (default: `["*"]`)
- Use a production vector store (Qdrant/Pinecone) over FAISS
- Set appropriate `LLM_TIMEOUT_SECONDS` and `LLM_MAX_RETRIES`

---

## 9. Project Structure

```
rag-engine/
├── app/
│   ├── api/
│   │   └── routes.py              # 9 API endpoints with validation
│   ├── config/
│   │   ├── constants.py            # Prompts, error messages, model maps
│   │   └── settings.py            # Pydantic settings with 50+ config options
│   ├── embeddings/
│   │   └── factory.py             # 5 embedding model adapters
│   ├── ingestion/
│   │   ├── chunking.py            # 3 chunking strategies
│   │   ├── loaders.py             # 7 file format loaders
│   │   └── pipeline.py            # Load→Clean→Chunk→Embed→Store
│   ├── llm/
│   │   └── factory.py             # 6 LLM provider adapters
│   ├── models/
│   │   └── schemas.py             # 20+ Pydantic request/response models
│   ├── retriever/
│   │   ├── rag_retriever.py       # 4 retrieval strategies + HyDE
│   │   └── reranker.py            # Cross-encoder reranking
│   ├── services/
│   │   ├── dependencies.py        # Dependency injection container
│   │   └── rag_service.py         # Central orchestrator
│   ├── utils/
│   │   ├── helpers.py             # Hashing, text cleaning, timing
│   │   └── logging.py             # Structured logging + latency decorator
│   ├── vectorstore/
│   │   ├── base.py                # Abstract vector store interface
│   │   ├── chroma_store.py        # ChromaDB adapter
│   │   ├── faiss_store.py         # FAISS adapter
│   │   ├── factory.py             # Vector store factory
│   │   ├── pinecone_store.py      # Pinecone adapter
│   │   └── qdrant_store.py        # Qdrant adapter
│   └── main.py                    # FastAPI app factory
├── tests/
│   ├── test_api/                  # API endpoint tests
│   ├── test_embedding/            # Embedding factory tests
│   ├── test_retriever/            # Retriever strategy tests
│   ├── test_vectorstore/          # Vector store tests
│   └── conftest.py                # Shared test fixtures
├── scripts/
│   └── seed_data.py               # Sample data generator
├── data/                          # Runtime data (uploads, indexes)
├── .env.example                   # Environment template
├── docker-compose.yml             # Container orchestration
├── Dockerfile                     # Multi-stage build
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Project metadata + tool config
```

**Total**: ~5,200 lines of production Python across 51 files.
