"""RAG Engine application package.

Sets OpenMP thread limits before any heavy library is imported.

FAISS and PyTorch each ship their own OpenMP runtime.  When both are loaded
into one process (which always happens here — FAISS for search, PyTorch for
embeddings and reranking), two things go wrong on macOS:

* Without ``KMP_DUPLICATE_LIB_OK``, the duplicate-runtime check aborts the
  process outright.
* With it, the runtimes coexist but their thread pools collide, and any FAISS
  search or torch forward pass made from a **non-main thread** segfaults.
  FastAPI runs every sync endpoint on a worker thread, so this crashed the
  server on the first query.

Pinning OpenMP to a single thread makes both libraries safe off the main
thread.  Search and single-query embedding are sub-second at this index size,
so the throughput cost is negligible.  Both values are ``setdefault`` so an
operator can still override them.

These must be set before ``faiss`` or ``torch`` is imported, which is why they
live here rather than in ``app.main``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
