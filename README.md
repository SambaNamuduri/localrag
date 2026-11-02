# Local AI Knowledge Manager

A modular local RAG starter project using Streamlit, Ollama, and Qdrant.

## Features

- Dynamic Ollama model discovery using `/api/tags`, `/api/ps`, and `/api/show`.
- Distinguishes installed models from models currently loaded in memory.
- Uses Ollama-advertised capabilities to identify chat and embedding models.
- PDF, DOCX, TXT, and Markdown ingestion.
- Chunking and batch embedding.
- Qdrant vector storage and semantic retrieval.
- RAG chat with source chunk inspection.
- SQLite inference audit trail, including retrieved context and failures.
- Document listing and deletion.
- Basic unit tests.

## Requirements

- Python 3.10+
- Ollama running locally
- Qdrant running locally

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Pull models. These are examples only; the application discovers installed models dynamically.

```bash
ollama pull nomic-embed-text
ollama pull gemma3
```

Start Qdrant:

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Run the application:

```bash
streamlit run app.py
```

## Architecture

```text
localrag/
├── app.py
├── config.py
├── services/
│   ├── container.py
│   ├── ollama_service.py
│   ├── document_service.py
│   ├── retrieval_service.py
│   └── audit_service.py
├── repositories/
│   └── qdrant_repository.py
├── exceptions/
│   └── app_exceptions.py
├── tests/
│   └── test_document_service.py
├── data/
└── requirements.txt
```

## Important implementation notes

1. **Embedding model consistency:** this starter stores the embedding model name in each point's payload, but does not yet enforce collection-level model locking. Before production, add collection metadata and reject searches using a different embedding model unless the collection is re-embedded.
2. **Capability metadata:** Ollama `/api/show` may return missing or incomplete capability metadata for some model/version combinations. Such models are shown as unsupported by the selector until capability probing is implemented. Do not assume every installed model is a chat model.
3. **Ingestion atomicity:** vectors are generated before Qdrant upsert, reducing partial writes. For very large documents, use a staging collection or ingestion manifest and explicit rollback/reconciliation.
4. **Audit privacy:** the audit database stores prompts, retrieved context, and model responses in plaintext. Restrict filesystem access and define retention policies before using sensitive documents.
5. **PDF extraction:** scanned PDFs need OCR; this starter reports no extractable text.
6. **Production hardening still required:** authentication/authorization, upload size enforcement, rate limiting, secrets management, backups, structured centralized logs, migrations, observability, and deployment hardening.

## Run tests

```bash
pytest -q
```


## Added hardening: embedding contract, upload limits, rollback

### Collection-level embedding contract

The SQLite `collection_registry` table stores one embedding model, vector dimension,
and distance metric per collection. The first successful ingestion registers the
contract. Later ingestion attempts using a different model or dimension are rejected.
Queries also verify the selected embedding model against the registered contract.
Use a new collection for a different embedding model, or implement a deliberate
re-embedding migration.

The registry is authoritative for this application. Do not modify collections
directly outside the application without reconciling the registry.

### Upload-size enforcement

- `.streamlit/config.toml` limits incoming uploads to 30 MB at the Streamlit server.
- `DocumentService.extract()` independently checks the reported upload size and
  actual byte length.
- Extracted text is capped at 50 MB before chunking.
- Change `MAX_UPLOAD_MB` in `.env` and the matching `maxUploadSize` in
  `.streamlit/config.toml` together.

### Ingestion rollback

The application computes all embeddings before writing points. It records a
manifest of intended Qdrant point IDs, performs one upsert, and marks the manifest
committed. If the upsert raises, it attempts a compensating delete by the exact
point IDs. A failed compensating delete is recorded as `ROLLBACK_FAILED` and
requires reconciliation. This is a saga/compensating transaction, not a true
distributed ACID transaction between SQLite and Qdrant.

### Operational caveats

- Keep the audit SQLite database on persistent storage and back it up.
- A process crash between Qdrant upsert and manifest status update can leave a
  `PREPARING` manifest. Add a startup reconciliation job before production use:
  inspect stale manifests, check Qdrant for their `document_id`, and either mark
  committed or delete orphaned points.
- Streamlit's server upload limit is fixed in `.streamlit/config.toml`; keep it
  aligned with `MAX_UPLOAD_MB`.
- `MAX_UPLOAD_MB` is enforced in application code after Streamlit accepts the
  request. The Streamlit server cap is what protects the server before the
  application reads the uploaded bytes.


## Startup recovery and manifest reconciliation

The `RecoveryService` reconciles stale ingestion manifests against the exact Qdrant
point IDs recorded before the write:

- If every expected point exists and its `document_id` matches the manifest, the
  manifest is marked `COMMITTED`.
- If the point set is partial or inconsistent, the service deletes the exact
  expected point IDs and verifies that none remain before marking `ROLLED_BACK`.
- If Qdrant is unavailable, deletion fails, or verification is inconclusive, the
  manifest becomes `ROLLBACK_FAILED` and can be retried.
- `RECOVERING` manifests are also eligible after the stale interval, so a crash
  during recovery is recoverable.
- Startup recovery runs when the application service container is first created.
  A manual **Recovery** tab action retries stale manifests without restarting.

`RECOVERY_STALE_SECONDS` defaults to 120 seconds. This grace period prevents
recovery from racing an ingestion that is still active in another worker. Keep
this value longer than the maximum expected ingestion duration in a multi-worker
deployment. The SQLite claim operation uses `BEGIN IMMEDIATE` and conditional
updates to reduce duplicate recovery work across workers.

This is a compensating recovery workflow, not a distributed transaction. Use
persistent SQLite storage, Qdrant backups, and operational monitoring. A
multi-instance production deployment should move manifests and claims to a
shared transactional database and use a lease/lock strategy.


## Code comments and tuning

Python modules, classes, and functions include purpose docstrings. Non-obvious operations retain inline comments. See `docs/TUNING_GUIDE.md` for exact settings and files to tune for upload limits, chunking, retrieval, timeouts, embedding batches, and recovery.
