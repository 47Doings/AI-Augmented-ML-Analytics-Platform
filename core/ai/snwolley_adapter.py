"""The ONE place the Snwolley Agent API Toolkit is called.

STATUS: placeholder. The Snwolley documentation has not been provided yet, so this
file deliberately does NOT guess at endpoints, headers, request or response
shapes. Until it is filled in, SnwolleyAdapter.generate() raises
ProviderUnavailable, and the platform falls back to the template summary and says
so on the page. Nothing else in the platform needs to change when this is done:

    TODO(snwolley-docs): implement SnwolleyAdapter.generate()
      - send `system` and `user` using the toolkit's own call
      - ask for JSON matching `json_schema` if the toolkit supports structured output
      - return the model's reply as a string (guardrails parse and validate it)
      - authenticate with self._api_key; never log it or put it in an error message
      - map toolkit errors/timeouts to ProviderUnavailable (short message)
    ASSUMPTION TO CHECK against the docs: that the toolkit can take a prompt and
    return text. If it is agent/tool-based instead, call it from here and still
    return text.

Credentials come from environment variables only (SNWOLLEY_API_KEY,
SNWOLLEY_BASE_URL via core.config); no key is ever written in code.
"""
from __future__ import annotations

from typing import Protocol

from core.config import Settings


class ProviderUnavailable(Exception):
    """The AI provider cannot answer right now (not configured, not implemented, network error)."""


class Provider(Protocol):
    name: str

    def generate(self, system: str, user: str, json_schema: dict) -> str: ...


class SnwolleyAdapter:
    name = "snwolley"

    def __init__(self, api_key: str, base_url: str | None):
        self._api_key = api_key  # never log or print this
        self._base_url = base_url

    def generate(self, system: str, user: str, json_schema: dict) -> str:
        raise ProviderUnavailable(
            "Snwolley adapter is not implemented yet (TODO(snwolley-docs) in core/ai/snwolley_adapter.py)."
        )

    def __repr__(self) -> str:  # so the key can never leak through a log line or traceback
        return "SnwolleyAdapter(api_key=***)"


def get_provider(settings: Settings) -> Provider | None:
    """The configured provider, or None when no key is set."""
    if not settings.snwolley_configured:
        return None
    return SnwolleyAdapter(settings.snwolley_api_key.get_secret_value(), settings.snwolley_base_url)
