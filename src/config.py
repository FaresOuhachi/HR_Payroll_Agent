"""
Application Configuration
=============================================================================
CONCEPT: pydantic-settings (BaseSettings)
Instead of reading os.environ manually, we define a typed class that:
  1. Reads from .env file automatically
  2. Validates types (str, int, bool) at startup
  3. Provides autocomplete in IDEs
  4. Fails fast if required variables are missing

This is a production best practice — you catch config errors at startup,
not at runtime when a user triggers a code path that reads the variable.
=============================================================================
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables / .env file.

    HOW IT WORKS:
    - Each field maps to an environment variable (case-insensitive)
    - Field `database_url` reads env var `DATABASE_URL`
    - Default values are used if the env var is not set
    - The .env file is loaded automatically from the project root
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file in current directory
        env_file_encoding="utf-8",
        case_sensitive=False,       # DATABASE_URL == database_url
    )

    # --- Database ---
    database_url: str = "postgresql+asyncpg://hr_admin:hr_secret_2025@localhost:5432/hr_payroll"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Groq (LLM inference) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # --- OpenAI (embeddings only — Groq doesn't support embedding models) ---
    openai_api_key: str = ""

    # --- JWT ---
    jwt_secret_key: str = "change-me-to-a-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # --- LangSmith (LLM Observability) ---
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://eu.api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "RH Payroll Agent"

    # --- App ---
    app_name: str = "HR Payroll Agent"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"


# Singleton instance — import this everywhere
# CONCEPT: Having a single Settings instance ensures config is loaded once
# and shared across the entire application.
settings = Settings()

# Export LangSmith env vars so the SDK picks them up.
# pydantic-settings reads .env into Python fields but does NOT set os.environ,
# while the LangSmith SDK reads directly from os.environ.
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
