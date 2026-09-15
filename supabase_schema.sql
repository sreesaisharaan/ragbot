-- Run this migration in the Supabase SQL Editor.
create extension if not exists pgcrypto;

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null check (length(filename) between 1 and 1024),
  content text not null,
  content_sha256 text not null check (content_sha256 ~ '^[0-9a-f]{64}$'),
  chunk_count integer not null default 0 check (chunk_count >= 0),
  created_at timestamptz not null default now()
);
create unique index if not exists documents_content_sha256_uq on public.documents(content_sha256);

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  chunk_index integer not null check (chunk_index >= 0),
  content text not null,
  embedding jsonb not null,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);
create index if not exists document_chunks_document_idx on public.document_chunks(document_id, chunk_index);

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  sources jsonb not null default '[]'::jsonb,
  context_found boolean,
  created_at timestamptz not null default now()
);
create index if not exists chat_messages_created_at_idx on public.chat_messages(created_at desc);

alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;
alter table public.chat_messages enable row level security;
-- The backend uses the service-role key. No browser/anon policies are created.
