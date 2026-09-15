# RAGbot local MVP

A small, testable document Q&A service built around the existing `RAGBot` core.

## What is included

- `POST /documents/upload` accepts UTF-8 `.txt` files.
- Uploaded text is split into ordered overlapping word chunks.
- Each chunk retains its filename and zero-based chunk index.
- Embeddings are deterministic local hashed bag-of-words vectors; no embedding API is required.
- `POST /ask` retrieves relevant chunks and answers through `RAGBot`.
- With no `GROQ_API_KEY`, the app uses a deterministic local extractive generator, so the MVP runs offline.
- With `GROQ_API_KEY`, generation uses the existing Groq adapter; embeddings remain local.
- A clean browser workspace is served at `/` with document upload, chat, loading states, errors, and source chips.
- When `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set, documents, chunks, embeddings, and chat messages are persisted in Supabase. Without them, the app falls back to in-memory storage.
- Unsupported PDF and DOCX uploads return `415`. They are intentionally not stubbed because they require additional extraction dependencies.

## Run on Windows

From `D:\project4` (or the equivalent project directory):

```text
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_server.py
```

The API is then available at `http://127.0.0.1:8000`. Interactive documentation is at `http://127.0.0.1:8000/docs`.
The browser app is at `http://127.0.0.1:8000/`.

## Deploy to Vercel

This repository includes `api/index.py` and `vercel.json` so Vercel can run the FastAPI app and serve the same browser UI. Import the repository into Vercel, set the `GROQ_API_KEY`, `SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY` environment variables in Vercel Project Settings, and deploy. Do not upload `.env`; it is ignored by Git.

## Supabase persistence

1. Create a Supabase project and run [`supabase_schema.sql`](supabase_schema.sql) in its SQL Editor.
2. Put the variables in `.env` (the current local setup also supports `.env.example`). Keep the service-role key server-side; never put it in `web/` or expose it to the browser.

```text
set SUPABASE_URL=https://your-project.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=your-server-only-service-role-key
```

3. Start with `python run_server.py`. Existing rows are loaded into the local retriever at startup, while new uploads and chats are written through Supabase REST.

Set `RAG_DEBUG=1` for exact prompt/context diagnostics. Normal INFO logs include chunk previews, embedding dimensions, retrieval scores, fallback decisions, and provider call status. The app module itself does not load dotenv files when imported, which keeps the offline test suite isolated from production credentials.

The current MVP stores local hashed embeddings as JSON and performs retrieval in the FastAPI process. This keeps the fallback deterministic; a production-scale deployment should move retrieval to a vector index such as `pgvector`.

If Python 3.11 is not installed, use an installed Python 3.10+ version consistently for the venv and commands above.

## API examples

Upload a text file with PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload `
  -F "file=@facts.txt;type=text/plain"
```

Ask a question:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"When is the launch?"}'
```

A successful answer has this shape:

```json
{
  "answer": "...",
  "sources": [{"filename":"facts.txt","chunk_index":0}],
  "context_found": true
}
```

- With the local hashed embedding fallback, retrieval uses lexical relevance plus vector similarity. `LOCAL_SIMILARITY_THRESHOLD=0.30` remains the no-lexical fallback and `LOCAL_TOP_K=2` keeps GPT-OSS context compact; the core production default remains `SIMILARITY_THRESHOLD=0.75`. Replace the local embedder with a semantic embedding model before relying on the production cutoff.

## Tests

```text
python -m pytest -q
```

The test suite covers the original RAG core plus chunking, TXT upload and metadata, unsupported file rejection, answering with a source, missing context, and empty questions.

## Current limitations

This is an in-memory local MVP when Supabase is not configured: uploaded documents disappear when the process restarts. With Supabase configured, documents and chats are persisted and chunks are reloaded at startup. Only UTF-8 TXT extraction is implemented. PDF/DOCX support is deliberately deferred until an extraction dependency and its validation policy are selected.
