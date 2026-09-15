"""Groq adapter and strict parsing for the structured generation contract."""
import json
import logging
from typing import Any

from config import GROQ_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger("ragbot.generator")

class GroqGenerator:
    def __init__(self, api_key: str | None = None, client: Any = None, model: str = GROQ_MODEL):
        if client is None:
            from groq import Groq  # optional dependency, imported only when used
            client = Groq(api_key=api_key)
        self.client, self.model = client, model

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        logger.info("[RAG][5-llm-call] model=%s temperature=%.2f max_tokens=%d system_chars=%d user_chars=%d", self.model, TEMPERATURE, MAX_TOKENS, len(system_prompt), len(user_prompt))
        logger.debug("[RAG][5-llm-context] exact_user_prompt=%s", user_prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Groq returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Groq JSON must be an object")
        logger.info("[RAG][5-llm-response] keys=%s", sorted(parsed))
        return parsed
