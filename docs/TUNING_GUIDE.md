# Configuration and tuning guide

Change one setting at a time and measure its effect.

## Environment variables (`.env`)

| Variable | Default | Purpose | Tune when |
|---|---:|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint | Ollama runs on another host/container |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant API endpoint | Qdrant runs elsewhere |
| `AUDIT_DB_PATH` | `data/ai_activity.sqlite3` | Durable SQLite database | You need persistent mounted storage |
| `CONNECT_TIMEOUT` | `5` sec | HTTP connection timeout | Services are remote or slow to accept connections |
| `READ_TIMEOUT` | `180` sec | Ollama response timeout | Large models/prompts need longer |
| `MAX_UPLOAD_MB` | `30` MB | Application upload cap | Uploaded files are larger/smaller |
| `MAX_TEXT_MB` | `50` MB | Extracted text cap | Text expands after extraction |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding API request | Reduce memory use or benchmark higher throughput |
| `RECOVERY_STALE_SECONDS` | `120` sec | Age before recovery claims an unfinished manifest | Must exceed longest valid ingestion time |
| `DEFAULT_COLLECTION` | `technical_docs` | Default collection name | You want another default |

Restart Streamlit after changing `.env`.

## Streamlit upload cap

File: `.streamlit/config.toml`

```toml
[server]
maxUploadSize = 30
```

Keep this at or below `MAX_UPLOAD_MB`. This server limit protects the process before application code reads the uploaded file.

## Chunking

File: `app.py`

- `chunk_size`: maximum characters per chunk.
- `overlap`: repeated characters shared by adjacent chunks.
- Increase chunk size when answers need more surrounding context.
- Decrease it when documents contain many independent topics.
- Keep overlap smaller than chunk size. Excessive overlap creates duplicate vectors.

Character-based chunking is a baseline. Code, tables, and technical manuals may need structure-aware chunking.

## Retrieval

File: `app.py`

- `top_k` controls how many chunks are supplied to the LLM.
- Increase it when the answer needs evidence from multiple sections.
- Decrease it when irrelevant chunks distract the model or prompts grow too large.

File: `services/retrieval_service.py`

- Query embedding must use the same embedding model registered for the collection.
- Never switch embedding models in-place without re-embedding all stored content.

## Ollama

File: `services/ollama_service.py`

- Model discovery uses `/api/tags`, `/api/ps`, and `/api/show`.
- Capability metadata may be missing on older/custom models; unknown capability must not be treated as supported.
- Chat temperature is controlled where `OllamaService.chat()` is called.

## Recovery

File: `services/recovery_service.py`

- Complete expected point set with matching `document_id`: mark `COMMITTED`.
- Partial/inconsistent set: delete exact manifest point IDs and verify absence.
- Failed or inconclusive deletion: retain `ROLLBACK_FAILED` for retry.
- Do not set the recovery grace period lower than the longest expected ingestion duration.

## Deployment

The SQLite manifest claim mechanism is intended for a local/single-host starter.
For multi-instance deployment, use a shared transactional database and a lease/lock
strategy. Keep SQLite and Qdrant on persistent storage and back both up.
