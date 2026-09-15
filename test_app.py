import io

import pytest
from fastapi.testclient import TestClient

from app import app, chunk_text, local_embedding


@pytest.fixture(autouse=True)
def clean_store():
    app.state.store.reset()
    yield
    app.state.store.reset()


def test_chunk_text_preserves_order_and_uses_word_overlap():
    chunks = chunk_text("one two three four five six", chunk_size=3, overlap=1)
    assert chunks == ["one two three", "three four five", "five six"]


def test_local_embedding_is_deterministic():
    assert local_embedding("Alpha beta") == local_embedding("Alpha beta")
    assert local_embedding("Alpha beta") != local_embedding("Gamma delta")


def test_upload_accepts_txt_and_retains_filename_and_chunk_metadata():
    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_added"] == 1
    assert body["chunks"][0] == {"filename": "notes.txt", "chunk_index": 0}


def test_upload_rejects_non_txt_files():
    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", b"not supported", "application/pdf")},
    )
    assert response.status_code == 415


def test_ask_returns_answer_and_uploaded_source():
    client = TestClient(app)
    client.post(
        "/documents/upload",
        files={"file": ("facts.txt", b"The launch is on Tuesday.", "text/plain")},
    )
    response = client.post("/ask", json={"question": "When is the launch?"})
    assert response.status_code == 200
    body = response.json()
    assert body["context_found"] is True
    assert body["sources"] == [{"filename": "facts.txt", "chunk_index": 0}]
    assert "Tuesday" in body["answer"]


def test_ask_reports_missing_context_without_generation():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "What is in the documents?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have enough information in the uploaded documents to answer that.",
        "sources": [],
        "context_found": False,
    }


def test_ask_requires_a_question():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "  "})
    assert response.status_code == 422
