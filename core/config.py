"""Application settings, loaded from environment variables and a local .env file.

Nothing secret is hard-coded in the source. Secrets use SecretStr, so they do
not show up in logs, tracebacks or print() output by accident.

If a required setting is missing or invalid, the app fails immediately at
startup with a clear message, instead of failing later in some random place.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "postgres"
    postgres_password: SecretStr  # required: no default on purpose
    postgres_db: str = "postgres"

    # --- App security ---
    app_secret_key: SecretStr  # required: no default on purpose
    # Set COOKIE_SECURE=true when the app is served over HTTPS (production). False allows http://localhost.
    cookie_secure: bool = False
    session_hours: int = 12  # how long a login stays valid

    # --- Snwolley Agent API Toolkit ---
    # Only the key is needed: the endpoint defaults to the one in the assignment brief.
    # Without a key the platform still runs and shows the labelled template summary.
    snwolley_api_key: SecretStr | None = None
    snwolley_base_url: str | None = "https://v1.snwolley.ai/v1/chat/completions/assignment"

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("snwolley_api_key", "snwolley_base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        """A blank line in .env (SNWOLLEY_API_KEY=) arrives as '' - treat it as not set."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("app_secret_key")
    @classmethod
    def _secret_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError(
                "APP_SECRET_KEY must be at least 32 characters. Generate one with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}")
        return value

    @property
    def snwolley_configured(self) -> bool:
        return self.snwolley_api_key is not None

    def safe_summary(self) -> dict[str, str]:
        """Settings in a form that is safe to print or log (no secret values)."""
        return {
            "postgres": f"{self.postgres_user}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}",
            "postgres_password": "set",
            "app_secret_key": "set",
            "snwolley_api_key": "set" if self.snwolley_api_key else "NOT SET (template summary only)",
            "snwolley_base_url": self.snwolley_base_url or "default endpoint",
            "log_level": self.log_level,
        }


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them everywhere."""
    return Settings()
