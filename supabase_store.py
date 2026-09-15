"""Optional server-side Supabase REST persistence.

The app stays offline when neither setting is configured. If only one setting is
present, startup fails closed so a typo cannot silently disable durability.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx


class PersistenceError(RuntimeError):
    """A configured persistence operation could not be completed."""


class SupabaseStore:
    def __init__(self, url: str | None = None, service_key: str | None = None):
        env_url = os.getenv("SUPABASE_URL", "").strip()
        env_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.url = (url if url is not None else env_url).strip().rstrip("/")
        self.service_key = (service_key if service_key is not None else env_key).strip()
        if bool(self.url) != bool(self.service_key):
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set together")
        self.enabled = bool(self.url and self.service_key)
        self._headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, table: str, **kwargs: Any) -> list[dict]:
        headers = {**self._headers, **kwargs.pop("headers", {})}
        try:
            response = httpx.request(
                method,
                f"{self.url}/rest/v1/{table}",
                headers=headers,
                timeout=10,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PersistenceError(f"Supabase {method} {table} failed") from exc
        if not response.content:
            return []
        data = response.json()
        return data if isinstance(data, list) else [data]

    def load_chunks(self) -> list[dict]:
        if not self.enabled:
            return []
        return self._request(
            "GET", "document_chunks",
            params={"select": "content,chunk_index,embedding,documents(filename)", "order": "document_id.asc,chunk_index.asc"},
        )

    def persist_document(self, filename: str, text: str, chunks: list[dict]) -> bool:
        """Persist a document; return False when the same content already exists."""
        if not self.enabled:
            return True
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._request("GET", "documents", params={"select": "id", "content_sha256": f"eq.{digest}", "limit": 1})
        if existing:
            return False
        rows = self._request(
            "POST", "documents",
            headers={"Prefer": "return=representation"},
            json={"filename": filename, "content": text, "content_sha256": digest, "chunk_count": len(chunks)},
        )
        if not rows or not rows[0].get("id"):
            raise PersistenceError("Supabase did not return the inserted document")
        document_id = rows[0]["id"]
        try:
            self._request(
                "POST", "document_chunks",
                json=[{
                    "document_id": document_id,
                    "chunk_index": item["chunk_index"],
                    "content": item["content"],
                    "embedding": item["embedding"],
                } for item in chunks],
            )
        except Exception:
            self._request("DELETE", "documents", params={"id": f"eq.{document_id}"})
            raise
        return True

    def persist_chat(self, question: str, result: dict) -> None:
        if not self.enabled:
            return
        self._request("POST", "chat_messages", json=[
            {"role": "user", "content": question, "sources": [], "context_found": None},
            {"role": "assistant", "content": result.get("answer", ""), "sources": result.get("sources", []), "context_found": bool(result.get("context_found"))},
        ])
