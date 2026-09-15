# Supabase integration

The app uses Supabase only from the FastAPI server through PostgREST. The browser never receives the service-role key.

## Setup

1. Run [`supabase_schema.sql`](supabase_schema.sql) in the Supabase SQL Editor.
2. Set both server-side variables:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-server-only-service-role-key
```

Both variables must be present together. If neither is present, the app uses its offline in-memory fallback. If only one is present, startup fails closed.

## Stored data

- `documents`: filename, full UTF-8 text, SHA-256 content hash, and chunk count.
- `document_chunks`: document foreign key, chunk index, content, and the local 128-value hashed embedding as JSON.
- `chat_messages`: separate user and assistant rows, validated sources, and context status.

Duplicate document content is detected by SHA-256 and is not inserted twice. The current retriever remains local cosine similarity; persisted chunks are loaded into it at startup. A future scale-up can replace this with a `pgvector` RPC without changing the API contract.

Persistence failures are returned as HTTP 503 responses rather than being reported as successful durable writes.
