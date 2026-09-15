"""Start RAGbot with local dotenv-style configuration loaded safely."""
from __future__ import annotations

import logging
import os
from pathlib import Path


BASE_DIR = Path(__file__).parent


def load_env() -> None:
    for filename in (".env", ".env.example"):
        path = BASE_DIR / filename
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value.strip('"').strip("'")
        return


if __name__ == "__main__":
    load_env()
    logging.basicConfig(level=logging.DEBUG if os.getenv("RAG_DEBUG", "").lower() in {"1", "true", "yes"} else logging.INFO)
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)
