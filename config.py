"""Runtime settings for the RAG bot."""

SIMILARITY_THRESHOLD = 0.75
LOCAL_SIMILARITY_THRESHOLD = 0.30
TOP_K = 5
LOCAL_TOP_K = 2
TEMPERATURE = 0.15
MAX_TOKENS = 500
GROQ_MODEL = "openai/gpt-oss-120b"


def validate_settings() -> None:
    if not 0.1 <= TEMPERATURE <= 0.2:
        raise ValueError("TEMPERATURE must be between 0.1 and 0.2")
    if TOP_K < 1:
        raise ValueError("TOP_K must be positive")
    if LOCAL_TOP_K < 1:
        raise ValueError("LOCAL_TOP_K must be positive")
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
