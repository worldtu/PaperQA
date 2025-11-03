"""
Central settings for PaperQA tools.
Loads from environment with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # API keys
    gemini_api_key: str | None = None

    # Ask LLM model/config
    ask_model: str = "gemini-2.0-flash"
    ask_temperature: float = 0.5
    ask_max_tokens: int = 150
    ask_max_words: int = 50

    # Cache / retries
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    max_retry_attempts: int = 3
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            ask_model=os.getenv("ASK_MODEL", "gemini-2.0-flash"),
            ask_temperature=float(os.getenv("ASK_TEMPERATURE", "0.5")),
            ask_max_tokens=int(os.getenv("ASK_MAX_TOKENS", "150")),
            ask_max_words=int(os.getenv("ASK_MAX_WORDS", "50")),
            cache_enabled=os.getenv("ASK_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_hours=int(os.getenv("ASK_CACHE_TTL_HOURS", "24")),
            max_retry_attempts=int(os.getenv("ASK_MAX_RETRY_ATTEMPTS", "3")),
            timeout_seconds=int(os.getenv("ASK_TIMEOUT_SECONDS", "30")),
        )


